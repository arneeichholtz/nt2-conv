import collections
import time
from typing import Optional

import numpy as np
import pyaudio
import torch
import webrtcvad
import whisper


class WhisperASR:
	"""Microphone ASR using Whisper with VAD-based endpointing."""

	def __init__(
		self,
		model_name: str = "small",		# Meeting: vervangen door FastWhisper.
		language: str = "nl",
		sample_rate: int = 16000,
		vad_aggressiveness: int = 2,		# sensitivity to speech activity detection. 0 is more likely to classify noise as speech, 3 only clear speech is detected. 
		download_root: Optional[str] = None,
		device: Optional[str] = None,
	) -> None:
		self.language = language
		self.sample_rate = sample_rate
		self.vad = webrtcvad.Vad(vad_aggressiveness)
		self.device = device
		self.fp16 = self.device != "cpu"

		self.model = whisper.load_model(
			model_name,
			device=self.device,
			download_root=download_root,
		)

	@staticmethod
	def list_input_devices() -> list[dict]:
		"""Return PyAudio input devices to help pick a device index."""
		pa = pyaudio.PyAudio()
		devices = []
		for idx in range(pa.get_device_count()):
			info = pa.get_device_info_by_index(idx)
			if info.get("maxInputChannels", 0) > 0:
				devices.append(info)
		pa.terminate()
		return devices

	def _open_stream(self, device_index: Optional[int], block_size: int) -> tuple[pyaudio.PyAudio, pyaudio.Stream]:
		pa = pyaudio.PyAudio()
		stream = pa.open(
			format=pyaudio.paInt16,
			channels=1,
			rate=self.sample_rate,
			input=True,
			frames_per_buffer=block_size,
			input_device_index=device_index,
		)
		return pa, stream

	def record_utterance(
		self,
		device_index: Optional[int] = None,
		max_record_seconds: float = 20.0,
		padding_ms: int = 300,
		silence_ms: int = 600,
		frame_ms: int = 10,     # Size of the audio frame fed to the voice-activity detector. Smaller means more responsive.
	) -> Optional[np.ndarray]:
		"""Record a single utterance using VAD. Returns float32 audio."""
		block_size = int(self.sample_rate * frame_ms / 1000)        # block_size=480
		if block_size <= 0:
			raise ValueError("frame_ms results in empty block size")

		pa, stream = self._open_stream(device_index, block_size)

		ring_buffer = collections.deque(maxlen=max(1, padding_ms // frame_ms))
		voiced_frames: list[bytes] = []
		triggered = False
		start_time = time.time()
		last_voice_time = start_time

		try:
			while True:
				data = stream.read(block_size, exception_on_overflow=False)
				is_speech = self.vad.is_speech(data, self.sample_rate)
				now = time.time()

				if not triggered:
					ring_buffer.append((data, is_speech))
					num_voiced = sum(1 for _, speech in ring_buffer if speech)
					if num_voiced > 0.9 * ring_buffer.maxlen:
						triggered = True
						voiced_frames.extend(frame for frame, _ in ring_buffer)
						ring_buffer.clear()
						last_voice_time = now
				else:
					voiced_frames.append(data)
					if is_speech:
						last_voice_time = now

					if (now - last_voice_time) * 1000 >= silence_ms:
						break

				if (now - start_time) >= max_record_seconds:
					break
		finally:
			stream.stop_stream()
			stream.close()
			pa.terminate()

		if not voiced_frames:
			return None

		audio_bytes = b"".join(voiced_frames)
		audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
		return audio

	def transcribe(self, audio: Optional[np.ndarray], initial_prompt: Optional[str] = None) -> str:
		"""Transcribe audio to text. Returns an empty string when no audio."""
		if audio is None or len(audio) == 0:
			return ""

		audio = whisper.pad_or_trim(audio)
		result = self.model.transcribe(
			audio,
			language=self.language,
			fp16=self.fp16,
			task="transcribe",
			initial_prompt=initial_prompt,
			verbose=False,
		)
		return result.get("text", "").strip()

	def listen_and_transcribe(
		self,
		device_index: Optional[int] = None,     # Device index is the index of the microphone input device to use. If None, the current default is used
		max_record_seconds: float = 20.0,
		initial_prompt: Optional[str] = None,
	) -> str:
		audio = self.record_utterance(
			device_index=device_index,
			max_record_seconds=max_record_seconds,
		)
		return self.transcribe(audio, initial_prompt=initial_prompt)
	


def test_asr() -> None:
	asr = WhisperASR(model_name="small", language="nl")
	print("Speak now. Recording will stop after silence or timeout.")
	text = asr.listen_and_transcribe()
	print(f"Transcription: {text}")

if __name__ == "__main__":
	test_asr()



"""Text-to-speech classes for speaking LLM responses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import tempfile
import wave
import winsound

from gtts import gTTS
from pydub import AudioSegment
from piper import PiperVoice, SynthesisConfig


class TextToSpeech:
	"""Offline TTS wrapper using Piper voice models."""

	def __init__(
		self,
		model_path: str | Path,
		config_path: str | Path,
		speaker_id: Optional[int] = None,
		use_cuda: bool = False,
	) -> None:
		self.model_path = Path(model_path)
		self.config_path = Path(config_path)
		# self.speaker_id = speaker_id
		if not self.model_path.exists():
			raise FileNotFoundError(f"Piper model not found: {self.model_path}")
		if not self.config_path.exists():
			raise FileNotFoundError(f"Piper config not found: {self.config_path}")
		self.synthesis_config = (
			SynthesisConfig(speaker_id=speaker_id) if speaker_id is not None else None
		)
		self.voice = PiperVoice.load(
			self.model_path,
			config_path=self.config_path,
			use_cuda=use_cuda,
		)

	def speak(self, text: str) -> None:
		if not text:
			return

		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
			output_path = Path(tmp_file.name)

		try:
			with wave.open(str(output_path), "wb") as wav_file:
				if self.synthesis_config is None:
					self.voice.synthesize_wav(text, wav_file)
				else:
					self.voice.synthesize_wav(text, wav_file, syn_config=self.synthesis_config)
			winsound.PlaySound(str(output_path), winsound.SND_FILENAME)
		finally:
			try:
				output_path.unlink(missing_ok=True)
			except OSError:
				pass


class GTTSTextToSpeech:
	"""Online TTS wrapper using Google TTS."""

	def __init__(self, language: str = "nl") -> None:
		self.language = language

	def speak(self, text: str) -> None:
		if not text:
			return

		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
			mp3_path = Path(mp3_file.name)
		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
			wav_path = Path(wav_file.name)

		try:
			gTTS(text=text, lang=self.language).save(str(mp3_path))
			audio = AudioSegment.from_file(mp3_path, format="mp3")
			audio.export(wav_path, format="wav")
			winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
		finally:
			for path in (mp3_path, wav_path):
				try:
					path.unlink(missing_ok=True)
				except OSError:
					pass


def create_text_to_speech(
	engine: str,
	model_path: str | Path | None = None,
	config_path: str | Path | None = None,
	speaker_id: Optional[int] = None,
	use_cuda: bool = False,
	language: str = "nl",
):
	engine_key = str(engine).strip().lower()
	if engine_key == "gtts":
		return GTTSTextToSpeech(language=language)
	if engine_key == "piper":
		if model_path is None or config_path is None:
			raise ValueError("Piper TTS requires model_path and config_path")
		return TextToSpeech(
			model_path=model_path,
			config_path=config_path,
			speaker_id=speaker_id,
			use_cuda=use_cuda,
		)
	raise ValueError(f"Unknown tts_engine: {engine}")


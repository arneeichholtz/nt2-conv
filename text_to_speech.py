
"""Text-to-speech helper for speaking LLM responses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import tempfile
import wave
import winsound

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
		self.speaker_id = speaker_id
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
				self.voice.synthesize_wav(
					text,
					wav_file,
					syn_config=self.synthesis_config,
				)
			winsound.PlaySound(str(output_path), winsound.SND_FILENAME)
		finally:
			try:
				output_path.unlink(missing_ok=True)
			except OSError:
				pass


from pathlib import Path
import yaml

from text_to_speech import TextToSpeech


def load_config(path: Path) -> dict:
	if not path.exists():
		return {}
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	return data or {}


def main() -> None:
	root = Path(__file__).resolve().parent
	config = load_config(root / "config.yml")
	model_path = config.get("tts_model_path")
	config_path = config.get("tts_config_path")
	if not model_path or not config_path:
		raise ValueError("Missing tts_model_path or tts_config_path in config.yml")

	model_path = Path(model_path)
	config_path = Path(config_path)
	if not model_path.is_absolute():
		model_path = root / model_path
	if not config_path.is_absolute():
		config_path = root / config_path

	tts = TextToSpeech(
		model_path=model_path,
		config_path=config_path,
		speaker_id=config.get("tts_speaker_id"),
	)
	tts.speak("Hallo, dit is een test van Piper.")


if __name__ == "__main__":
	main()

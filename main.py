
# The NT2 conversation tool is built out of the following functionality:

# 1. Automatic speech recogntion using Whisper or wav2vec2 to transcribe audio input into a text user query

# 2. A database where the context information is stored, which will be example conversations/phrases and 
# vocab lists corresponding to the student language level
# - How will these be stored? Do we extract full conversations, or only phrases? 
#   How do we store and retrieve the vocab list?
# - What other elements are useful to store in the database?

# 3. A RAG/LLM component to do the following:
# - Find the relevant documents for the given user query; this can be parts of a conversion or vocab list
# - Generate a response with an LLM based on the retrieved documents and query

# 4. A text-to-speech component to convert the generated response into audio output for the user


from pathlib import Path
import string
import yaml
import random
from asr import WhisperASR
from langchain_core.messages import AIMessage, HumanMessage

from generate import generate_reply, start_conversation
from text_to_speech import TextToSpeech
import torch


# def check_gpu():
# 	print("\n--- GPU CHECK ---")
# 	if torch.cuda.is_available():
# 		print(f"GPU detected byPython: {torch.cuda.get_device_name(0)}")
# 	else:
# 		print("No GPU detected. Python will use the CPU.")
# 	print("-----------------\n")


def load_config(path: Path) -> dict:
	if not path.exists():
		return {}
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	return data or {}


def test_asr() -> None:
	asr = WhisperASR(model_name="small", language="nl", device="cpu")
	print("Speak now. Recording will stop after silence or timeout.")
	text = asr.listen_and_transcribe()
	print(f"Transcription: {text}")


# Meeting: generate function to show GPU memory.
# Meeting: make github so Lou can test code.

if __name__ == "__main__":
	
	# _test_asr()
	# Language level of student is given
	# Conversation subject is given (e.g., Kennismaken)
	
	root = Path(__file__).resolve().parent
	config = load_config(root / "config.yml")
	model_name = config["model_name"]
	temperature = float(config["temperature"])
	input_format = str(config.get("input_format", "text")).strip().lower()
	output_format = str(config.get("output_format", "text")).strip().lower()
	tts_model_path = config.get("tts_model_path")
	tts_config_path = config.get("tts_config_path")
	tts_speaker_id = config.get("tts_speaker_id")
	conversation_theme = str(config.get("conversation_theme", "kennismaken")).strip().lower()
	language_level = str(config.get("language_level", "A2")).strip()

	if input_format not in {"text", "speech"}:
		print(f"Unknown input_format '{input_format}', falling back to text.")
		input_format = "text"
	if output_format not in {"text", "speech"}:
		print(f"Unknown output_format '{output_format}', falling back to text.")
		output_format = "text"

	print(f"Using model: {model_name} with temperature: {temperature}")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	asr = WhisperASR(model_name="small", language="nl", device=device) if input_format == "speech" else None

	if output_format == "speech":
		if not tts_model_path or not tts_config_path:
			print("Missing tts_model_path or tts_config_path in config.yml. Falling back to text output.")
			output_format = "text"
			tts = None
		else:
			model_path = Path(tts_model_path)
			config_path = Path(tts_config_path)
			if not model_path.is_absolute():
				model_path = root / model_path
			if not config_path.is_absolute():
				config_path = root / config_path
			if not model_path.exists() or not config_path.exists():
				print("Piper model or config file not found. Falling back to text output.")
				output_format = "text"
				tts = None
			else:
				tts = TextToSpeech(
					model_path=model_path,
					config_path=config_path,
					speaker_id=tts_speaker_id,
				)
	else:
		tts = None

	level_prompt_path = root / "prompts" / f"prompt_{language_level}.txt"
	theme_prompt_path = root / "prompts" / f"prompt_{conversation_theme}.txt"
	topic_file_path = root / "data" / conversation_theme / f"{conversation_theme}.txt"
	options_file_path = root / "data" / conversation_theme / f"{conversation_theme}_opties.txt"
	
	
	location = ""
	if options_file_path.exists():
		locaties_text = options_file_path.read_text(encoding="utf-8").strip()
		location_options = [loc.strip() for loc in locaties_text.split(',')]
		if location_options and location_options[0]:
			location = random.choice(location_options)
			print(f"Gekozen locatie: {location}")

	first_line = start_conversation(
		level_prompt_path,
		theme_prompt_path,
		topic_file_path,
		model_name=model_name,
		temperature=temperature,
		locatie=location
	)

	print(first_line)
	if output_format == "speech" and tts is not None:
		tts.speak(first_line)

	messages = [AIMessage(content=first_line)]
	while True:
		if input_format == "speech":
			print("Luister... spreek nu.")
			user_text = asr.listen_and_transcribe() if asr is not None else ""
			if not user_text:
				print("Geen spraak gehoord.")
				continue
			print(f"Jij (spraak): {user_text}")
		else:
			user_text = input("Jij: ").strip()
			if not user_text:
				continue

		clean_text = user_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
		if clean_text in {"/quit", "/exit", "tot ziens", "doei", "stop"}:
			print("Gesprek gestopt.")
			break

		messages.append(HumanMessage(content=user_text))
		reply = generate_reply(
			level_prompt_path,
			theme_prompt_path,
			topic_file_path,
			messages,
			model_name=model_name,
			temperature=temperature,
			locatie=location
		)

		print(reply)
		if output_format == "speech" and tts is not None:
			tts.speak(reply)
		messages.append(AIMessage(content=reply))







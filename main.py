from pathlib import Path
import string
import time
import yaml
import random
from asr import WhisperASR
from langchain_core.messages import AIMessage, HumanMessage

from generate import create_chain, generate_reply, start_conversation
from language_check import check_llm_language_issue, check_manual_language_issue
from talking_head import create_talking_head_renderer
from text_to_speech import create_text_to_speech
import torch


DEFAULT_LANGUAGE_ERROR_MESSAGE = "De woordvolgorde is niet helemaal correct, probeer het nog eens."
DEFAULT_CONJUGATION_ERROR_MESSAGE = "De werkwoordvervoeging is niet helemaal correct, probeer het nog eens."


def load_config(path: Path) -> dict:
	if not path.exists():
		return {}
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	return data or {}


def format_language_feedback(language_feedback: str, check: str | None = None) -> str:

	if check == "conjugation":
		return DEFAULT_CONJUGATION_ERROR_MESSAGE
	if check == "word_order":
		return DEFAULT_LANGUAGE_ERROR_MESSAGE

	language_feedback_lower = language_feedback.lower()
	if "vervoeging" in language_feedback_lower:
		return DEFAULT_CONJUGATION_ERROR_MESSAGE
	if "woordvolgorde" in language_feedback_lower or "volgorde" in language_feedback_lower:
		return DEFAULT_LANGUAGE_ERROR_MESSAGE

	return "Deze zin is niet helemaal correct, probeer het nog eens."


def test_asr() -> None:
	asr = WhisperASR(model_name="small", language="nl", device="cpu")
	print("Speak now. Recording will stop after silence or timeout.")
	text = asr.listen_and_transcribe()
	print(f"Transcription: {text}")


def correct_saved_errors(
	label: str,
	sentences: list[str],
	correction_chain,
	correction_prompt_path: Path,
) -> None:
	if not sentences:
		return

	print(f"\nFEEDBACK -- {label}:")
	correction_prompt = correction_prompt_path.read_text(encoding="utf-8")
	for sentence in sentences:
		correction_response = correction_chain.invoke(
			{
				"system_instruction": correction_prompt,
				"messages": [HumanMessage(content=sentence)],
			}
		)
		correct_sentence = correction_response.content.strip()
		print(f"{sentence} -> {correct_sentence}")


if __name__ == "__main__":
	
	root = Path(__file__).resolve().parent
	config = load_config(root / "config.yml")

	conversation_model_name = config.get("conversation_model_name")
	conversation_temperature = config.get("conversation_temperature")
	conversation_keep_alive = config.get("conversation_keep_alive")
	conversation_reasoning = config.get("conversation_reasoning")
	print(f"Using conversation model: {conversation_model_name} with temperature: {conversation_temperature}")
	print(f"Conversation keep_alive: {conversation_keep_alive}, reasoning: {conversation_reasoning}")
	
	language_check_mode = config.get("language_check_mode", "regels")
	correct_errors = config.get("correct_errors", False)
	
	input_format = config.get("input_format", "text")
	output_format = config.get("output_format", "text")
	tts_engine = config.get("tts_engine", "piper")
	tts_speaker_id = config.get("tts_speaker_id")
	piper_model_path = config.get("piper_model_path", config.get("tts_model_path"))
	piper_config_path = config.get("piper_config_path", config.get("tts_config_path"))
	talking_head_backend = config.get("talking_head", "uit")

	conversation_theme = config.get("conversation_theme", "kennismaken")
	language_level = config.get("language_level", "A2")

	if input_format not in {"text", "speech"}:
		print(f"Unknown input_format '{input_format}', falling back to text.")
		input_format = "text"
	if output_format not in {"text", "speech"}:
		print(f"Unknown output_format '{output_format}', falling back to text.")
		output_format = "text"
	
	if language_check_mode == "llm":
		checker_model_name = config.get("checker_model_name")
		checker_temperature = config.get("checker_temperature")
		checker_keep_alive = config.get("checker_keep_alive")
		checker_reasoning = config.get("checker_reasoning")
		print(f"Using checker model: {checker_model_name} with temperature: {checker_temperature}")
		print(f"Checker keep_alive: {checker_keep_alive}, reasoning: {checker_reasoning}")
	
	elif language_check_mode == "regels":
		manual_language_checks = config.get("language_check_rules", ["woordvolgorde"])
		print(f"Using manual language checks: {', '.join(manual_language_checks)}")
	else:
		manual_language_checks = []
		print("Language checker is disabled")
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	asr = WhisperASR(model_name="small", language="nl", device=device) if input_format == "speech" else None		# ASR using FasterWhisper

	if output_format == "speech":
		tts = None
		if talking_head_backend not in {"", "uit", "off", "none", "false", "0"}:
			try:
				tts = create_talking_head_renderer(root, talking_head_backend, tts_engine)
				print(f"Using talking head backend: {talking_head_backend} ({tts_engine})")
			except Exception as exc:
				raise RuntimeError(
					f"Talking head init failed for backend '{talking_head_backend}' and engine '{tts_engine}': {exc}"
				) from exc

		if tts is None:
			if str(tts_engine).strip().lower() == "gtts":
				tts = create_text_to_speech(tts_engine, language="nl")
			else:
				if not piper_model_path or not piper_config_path:
					print("Missing piper_model_path or piper_config_path in config.yml. Falling back to text output.")
					output_format = "text"
					tts = None
				else:
					model_path = Path(piper_model_path)
					config_path = Path(piper_config_path)
					if not model_path.is_absolute():
						model_path = root / model_path
					if not config_path.is_absolute():
						config_path = root / config_path
					if not model_path.exists() or not config_path.exists():
						print("Piper model or config file not found. Falling back to text output.")
						output_format = "text"
						tts = None
					else:
						tts = create_text_to_speech(
							tts_engine,
							model_path=model_path,
							config_path=config_path,
							speaker_id=tts_speaker_id,
							use_cuda=device == "cuda",
						)
	else:
		tts = None

	# Load file paths
	level_prompt_path = root / "prompts" / f"prompt_{language_level}.txt"
	theme_prompt_path = root / "prompts" / f"prompt_{conversation_theme}.txt"
	checker_prompt_path = root / "prompts" / "prompt_taalcontrole.txt"
	correction_prompt_path = root / "prompts" / "prompt_correctie.txt"
	topic_file_path = root / "voorbeelden" / conversation_theme / f"{conversation_theme}.txt"
	options_file_path = root / "voorbeelden" / conversation_theme / f"{conversation_theme}_opties.txt"
	
	location = ""
	if options_file_path.exists():
		locaties_text = options_file_path.read_text(encoding="utf-8").strip()
		location_options = [loc.strip() for loc in locaties_text.split(',')]
		if location_options and location_options[0]:
			location = random.choice(location_options)
			print(f"Gekozen locatie: {location}")

	# start_time = time.perf_counter()
	first_line = start_conversation(
		level_prompt_path,
		theme_prompt_path,
		topic_file_path,
		model_name=conversation_model_name,
		temperature=conversation_temperature,
		keep_alive=conversation_keep_alive,
		reasoning=conversation_reasoning,
		locatie=location
	)

	print(first_line)
	if output_format == "speech" and tts is not None:
		tts.speak(first_line)

	messages = [AIMessage(content=first_line)]
	word_order_errors: list[str] = []
	conjugation_errors: list[str] = []
	correction_chain = create_chain(
		model_name=conversation_model_name,
		temperature=conversation_temperature,
		keep_alive=conversation_keep_alive,
		reasoning=conversation_reasoning,
	)
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

		clean_text = user_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()		# Remove punctuation and whitespace for comparison
		if clean_text in {"/quit", "/exit", "tot ziens", "doei", "stop", "fijne dag", "fijn weekend", "bedankt", "dankjewel", "dank u wel", "dank je wel"}:
			print("Gesprek gestopt.")
			break

		last_question = messages[-1].content if messages else ""

		language_feedback = None
		if language_check_mode == "llm":
			# checker_start_time = time.perf_counter()
			language_feedback = check_llm_language_issue(
				checker_prompt_path,
				last_question,        # Meest recente vraag van LLM
				user_text,            # Tekst van gebruiker
				model_name=checker_model_name,
				temperature=checker_temperature,
				keep_alive=checker_keep_alive,
				reasoning=checker_reasoning,
				locatie=location,
			)
			# print(f"[timing] Language checker took {time.perf_counter() - checker_start_time:.3f}s")
		elif language_check_mode == "regels":
			# checker_start_time = time.perf_counter()
			manual_language_result = check_manual_language_issue(
				user_text,
				rules=manual_language_checks,
				return_details=True,
			)
			if not manual_language_result["correct"]:
				language_feedback = manual_language_result["message"]
				if correct_errors:
					checks = manual_language_result.get("checks") or [manual_language_result.get("check")]
					if "word_order" in checks:
						word_order_errors.append(user_text)
					if "conjugation" in checks:
						conjugation_errors.append(user_text)
			# print(f"[timing] Language checker took {time.perf_counter() - checker_start_time:.3f}s")

		if language_feedback:
			feedback_to_print = format_language_feedback(
				language_feedback,
				manual_language_result.get("check") if language_check_mode == "regels" else None,
			)
			print(f"Taalfeedback: {feedback_to_print}")
			if output_format == "speech" and tts is not None:
				tts.speak(feedback_to_print)
			continue

		# reply_start_time = time.perf_counter()
		messages.append(HumanMessage(content=user_text))
		reply = generate_reply(
			level_prompt_path,
			theme_prompt_path,
			topic_file_path,
			messages,
			model_name=conversation_model_name,
			temperature=conversation_temperature,
			keep_alive=conversation_keep_alive,
			reasoning=conversation_reasoning,
			locatie=location
		)
		# print(f"[timing] Conversation model took {time.perf_counter() - reply_start_time:.3f}s")

		print(reply)
		if output_format == "speech" and tts is not None:
			tts.speak(reply)
		messages.append(AIMessage(content=reply))

	if correct_errors:
		correct_saved_errors("je maakte deze woordvolgorde fouten", word_order_errors, correction_chain, correction_prompt_path)
		correct_saved_errors("je maakte deze vervoegingsfouten", conjugation_errors, correction_chain, correction_prompt_path)







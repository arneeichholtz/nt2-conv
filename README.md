# NT2 Conversation Tool

A Dutch NT2 conversation assistant that provides interactive dialogue practice. The system combines:
- **Automatic Speech Recognition (ASR)** – Transcribe spoken Dutch input using FasterWhisper
- **Synthetic Conversations** - An LLM is prompted to perform a practice conversation with a student
- **Text-to-Speech (TTS)** – Generate spoken replies with Piper or gTTS
- **Language Checking** – Word-order and simple verb conjugation checks based on implemented rules or an LLM
- **Talking Head** – Animated avatar to accompany spoken responses
- **Error Correction** – Post-conversation analysis of language mistakes

## Project Structure
- `main.py`: Entry point for the interactive conversation loop
- `config.yml`: Runtime configuration (models, formats, language-checking, TTS settings)
- `prompts/`: System prompts for each language level (A2) and conversation theme (e.g., kennismaken)
- `voorbeelden/`: Example conversations organized by theme
- `voices/`: TTS model files (Piper `.onnx` models)
- `language_check_utils/`: Language checking utilities (word order, conjugation rules)
- `generate.py`: LLM integration for conversation generation and correction
- `asr.py`: Whisper-based automatic speech recognition
- `text_to_speech.py`: TTS engine abstraction (Piper or gTTS)
- `talking_head/`: Optional talking head animation system. This is the repository from Lou, some files are redundant.

## Features
- **Input/Output Modes**: Text input or speech recognition via microphone and text responses or synthesized speech
- **Language Checking**: Disable, enable rule-based checks (word order, conjugation), or use an LLM checker. With a qwen2.5 model, the LLM language check fails to work. With a newer model, for instance qwen3, the language check takes prohibtively long.
- **Error Correction**: Optionally track and correct learner errors after conversation ends
- **Available Conversation Themes**: kennismaken, winkelen, wonen
- **Model Flexibility**: Support for Ollama-based LLMs, Ollama-based language checker, and two TTS engines (Piper and gTTS)

## Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv nt2-env
   source nt2-env/Scripts/activate  # On Windows: nt2-env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure `config.yml` with your preferred settings (see **Configuration** section below).

## Run
```bash
python main.py
```

The system will start an interactive conversation. Exit by typing or speaking one of: `/quit`, `/exit`, `tot ziens`, `doei`, `stop`, `fijne dag`, `fijn weekend`, `bedankt`, `dankjewel`, or `dank je wel`.

---

## Configuration

All settings are managed in `config.yml`. Below is a comprehensive guide to each setting:

### Conversation Model (LLM)
- `conversation_model_name` (string): Ollama model name for main conversation (e.g., `qwen2.5:14b-instruct`, `gemma4:e2b`)
- `conversation_temperature` (float): Temperature for conversation model responses (0 = deterministic, higher = more creative)
- `conversation_keep_alive` (string): Ollama keep-alive timeout (e.g., `3m` for 3 minutes)
- `conversation_reasoning` (boolean): Enable reasoning mode if supported by the model

### Conversation Theme & Level
- `conversation_theme` (string): Topic of conversation – options: `kennismaken`, `winkelen`, `wonen`
- `language_level` (string): Learner proficiency – options: `A2`

### Input/Output Format
- `input_format` (string): How user provides input – `text` (keyboard) or `speech` (microphone via ASR)
- `output_format` (string): How system responds – `text` (printed) or `speech` (TTS + optional talking head)

### Text-to-Speech (TTS)
- `tts_engine` (string): TTS backend – `piper` (Piper ONNX) or `gtts` (Google Text-to-Speech)
- `piper_model_path` (string): Path to Piper `.onnx` model file (e.g., `voices/nl_NL-alex-medium.onnx`)
- `piper_config_path` (string): Path to Piper `.onnx.json` config file (e.g., `voices/nl_NL-alex-medium.onnx.json`)
- `tts_speaker_id` (integer or null): Speaker ID for multi-speaker Piper models (optional)

### Talking Head (Avatar)
- `talking_head` (string): Animated avatar backend – `uit`/`off`/`none` (disabled), `pytoon`, or `makeittalk`

### Language Checking
- `language_check_mode` (string): Checking strategy – `uit`/`off` (disabled), `regels` (rule-based), or `llm` (LLM-based)
- `language_check_rules` (list): Rules to check when `language_check_mode` is `regels` – options: `woordvolgorde` (word order), `vervoeging` (conjugation)
- `correct_errors` (boolean): After conversation, ask LLM to correct recorded mistakes (requires `language_check_mode` to capture errors)

### Language Checker Model (LLM)
Used only when `language_check_mode` is `llm`:
- `checker_model_name` (string): Ollama model for language checking (e.g., `qwen2.5:7b-instruct`)
- `checker_temperature` (float): Temperature for checker model (0 = strict checking, higher = more lenient)
- `checker_keep_alive` (string): Ollama keep-alive timeout
- `checker_reasoning` (boolean): Enable reasoning mode if supported by the model

### Example Configurations

**Text-based conversation without language checks (for testing):**
```yaml
conversation_model_name: qwen2.5:14b-instruct
input_format: text
output_format: text
language_check_mode: uit
```

**Speech with talking head and piper TTS:**
```yaml
conversation_model_name: qwen2.5:14b-instruct
input_format: speech
output_format: speech
talking_head: makeittalk
tts_engine: piper
language_check_mode: regels
language_check_rules: [woordvolgorde, vervoeging]
correct_errors: True
```

---

## Notes

- **Piper Models**: TTS with Piper requires `.onnx` model and `.onnx.json` config files in the `voices/` directory (see the README in the `voices` folder for instructions where to find these files). If files are missing, the system falls back to text output.
- **Talking Head**: The `makeittalk` and `pytoon` animation backends require additional model files. Specifically, the `ckpt` folder is missing from /talking_head/examples/ where it should be placed. The checkpoint files were too large for GitHub, the folder should be pasted here in its entirety.
- **Ollama**: The system expects Ollama to be running locally (default: `http://localhost:11434`). Install from [ollama.ai](https://ollama.ai).
- **GPU Acceleration**: PyTorch and ASR will automatically use CUDA if available; otherwise, fall back to CPU.
- **Error Tracking**: When `correct_errors` is enabled, word-order and conjugation mistakes are collected during conversation and shown at the end with LLM-generated corrections.
- **Microphone & PyAudio**: Speech input requires a working microphone and `pyaudio` support. If input fails, fall back to `text`.



## TODO

- Verb conjugation checks do not work very well. This can be improved.

- The conversation themes can be extended beyond kennismaken, wonen, winkelen.
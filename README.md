# NT2 Conversation Tool

A Dutch NT2 conversation assistant that combines:
- Automatic speech recognition
- Text-to-speech for spoken replies

## Project structure
- `main.py`: entry point for the interactive conversation loop.
- `config.yml`: runtime configuration (model name, text or speech format, TTS paths).
- `prompts/`: system prompts per language level and theme.
- `data/`: example conversations and vocabulary by theme.
- `voices/`: TTS model files

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Update `config.yml` as needed (input/output formats, model, theme, TTS paths, and language-check settings).

## Run
```
python main.py
```

## Notes
- For speech input, your system needs a working microphone and `pyaudio` support.
- TTS output requires a valid `.onnx` model and `.onnx.json` config in `voices/`.
- Set `print_error_type: true` in `config.yml` to show the specific language-check error during testing; leave it `false` to show a generic message to the learner.

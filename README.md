# NT2 Conversation Tool

A Dutch NT2 (A2 and other levels) conversation assistant that combines:
- Automatic speech recognition (Whisper or wav2vec2)
- Retrieval-augmented prompting for theme-based conversations
- Text-to-speech for spoken replies

## Project structure
- `main.py`: entry point for the interactive conversation loop.
- `config.yml`: runtime configuration (model name, formats, TTS paths).
- `prompts/`: system prompts per language level and theme.
- `data/`: example conversations and vocabulary by theme.
- `voices/`: TTS model files (ignored by git; see its README).

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Update `config.yml` as needed (input/output formats, model, theme, and TTS paths).

## Run
```
python main.py
```

## Notes
- For speech input, your system needs a working microphone and `pyaudio` support.
- TTS output requires a valid `.onnx` model and `.onnx.json` config in `voices/`.

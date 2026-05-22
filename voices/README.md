# Voices

This folder contains speech synthesis voice model files used by the project.

## File types
- `.onnx`: the voice model.
- `.onnx.json`: metadata for the model (sample rate, speaker info, etc.).

## Adding new voices
1. Piper voice files can be retrieved from: https://huggingface.co/rhasspy/piper-voices/tree/main
2. Navigate to a language and select a voice.
3. Place the `.onnx` and matching `.onnx.json` files in this folder.

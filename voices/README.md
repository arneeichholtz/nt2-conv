# Voices

This folder contains speech synthesis voice model files used by the project.

## File types
- `.onnx`: the voice model.
- `.onnx.json`: metadata for the model (sample rate, speaker info, etc.).

## Adding new voices
1. Place the `.onnx` and matching `.onnx.json` files here.
2. Keep filenames short and consistent (e.g., `nl_NL-<name>-medium.onnx`).
3. Update any config or prompts that reference the new voice name.

## Version control
Model files are large and are ignored by git in this repo. If you need to share
them, use a separate storage location or artifact store.

"""Adapters for optional talking-head speech rendering."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


@dataclass
class TalkingHeadRenderer:
	"""Simple wrapper so main.py can call speak() uniformly."""

	render_fn: callable
	working_dir: Path

	def speak(self, text: str) -> None:
		if not text:
			return
		previous_cwd = os.getcwd()
		try:
			os.chdir(self.working_dir)
			self.render_fn(text)
		finally:
			os.chdir(previous_cwd)


def load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not create module spec for {module_path}")

	module = importlib.util.module_from_spec(spec)
	module_dir = str(module_path.parent)
	previous_cwd = os.getcwd()
	added_to_syspath = False
	try:
		if module_dir not in sys.path:
			sys.path.insert(0, module_dir)
			added_to_syspath = True
		os.chdir(module_dir)
		spec.loader.exec_module(module)
	finally:
		os.chdir(previous_cwd)
		if added_to_syspath and module_dir in sys.path:
			sys.path.remove(module_dir)
	return module


def create_talking_head_renderer(root: Path, backend: str, tts_engine: str = "piper") -> TalkingHeadRenderer:
	"""Create a talking-head renderer from Lou_files implementation."""
	backend_key = backend.strip().lower()
	tts_engine_key = tts_engine.strip().lower()

	backend_to_path = {
		"pytoon": root / "talking_head" / "pytoon.py",
		"makeittalk": root / "talking_head" / "MakeItTalk.py",
	}
	if backend_key not in backend_to_path:
		raise ValueError("talking_head must be one of: pytoon, makeittalk")

	module_path = backend_to_path[backend_key]
	if not module_path.exists():
		raise FileNotFoundError(f"Talking-head script not found: {module_path}")

	module_name = f"talking_head_{backend_key}"
	module = load_module_from_path(module_name, module_path)

	function_name = "talkingHead_gtts" if tts_engine_key == "gtts" else "talkingHead_piper"
	if not hasattr(module, function_name):
		raise AttributeError(f"Function {function_name} is missing in {module_path.name}")

	render_fn = getattr(module, function_name)
	if not callable(render_fn):
		raise TypeError(f"Attribute {function_name} is not callable in {module_path.name}")

	return TalkingHeadRenderer(render_fn=render_fn, working_dir=module_path.parent)

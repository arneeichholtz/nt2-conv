
"""Generate NT2 conversation responses using LangChain and a local LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def read_text_file(path: Path) -> str:
	return path.read_text(encoding="utf-8")


def build_system_instruction(level_prompt: str, theme_prompt: str, topic_context: str) -> str:
	return (
		f"{level_prompt}\n\n"
		f"{theme_prompt}\n\n"
		"Onderstaande gesprekken zijn voorbeelden van onderwerpen en hoe het gesprek kan verlopen, gescheiden door VOORBEELD. Gebruik ze als inspiratie. " \
		f"Voorbeelden:\n\n{topic_context}"
	)


def create_chain(model_name: str, temperature: float) -> ChatPromptTemplate:
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", "{system_instruction}"),
			MessagesPlaceholder(variable_name="messages"),
		]
	)
	llm = ChatOllama(model=model_name, temperature=temperature)
	return prompt | llm


def start_conversation(
	level_prompt_path: str | Path,
	theme_prompt_path: str | Path,
	topic_file_path: str | Path,
	model_name: str = "gemma4:e2b",
	temperature: float = 0.7,
	**kwargs
) -> str:
	level_prompt = read_text_file(Path(level_prompt_path))
	try:
		theme_prompt = read_text_file(Path(theme_prompt_path)).format(**kwargs)
	except KeyError:
		theme_prompt = read_text_file(Path(theme_prompt_path))
	topic_context = read_text_file(Path(topic_file_path))
	system_instruction = build_system_instruction(level_prompt, theme_prompt, topic_context)
	
	chain = create_chain(model_name=model_name, temperature=temperature)
	response = chain.invoke(
		{
			"system_instruction": system_instruction,
			"messages": [HumanMessage(content="Start het gesprek nu.")],
		}
	)
	return response.content


def generate_reply(
	level_prompt_path: str | Path,
	theme_prompt_path: str | Path,
	topic_file_path: str | Path,
	messages: Iterable[BaseMessage],
	model_name: str = "gemma4:e2b",
	temperature: float = 0.7,
	**kwargs
) -> str:
	level_prompt = read_text_file(Path(level_prompt_path))
	try:
		theme_prompt = read_text_file(Path(theme_prompt_path)).format(**kwargs)
	except KeyError:
		theme_prompt = read_text_file(Path(theme_prompt_path))

	topic_context = read_text_file(Path(topic_file_path))
	system_instruction = build_system_instruction(level_prompt, theme_prompt, topic_context)

	chain = create_chain(model_name=model_name, temperature=temperature)
	response = chain.invoke(
		{
			"system_instruction": system_instruction,
			"messages": list(messages),     # Includes both human and AI responses
		}
	)
	return response.content



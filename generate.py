
"""Generate NT2 conversation responses using LangChain and a local LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from language_check import read_text_file


def build_system_instruction(level_prompt: str, theme_prompt: str, topic_context: str) -> str:
	return (
		f"{level_prompt}\n\n"
		f"{theme_prompt}\n\n"
		"Onderstaande gesprekken zijn voorbeelden van onderwerpen en hoe het gesprek kan verlopen, gescheiden door VOORBEELD. Gebruik ze als inspiratie. " \
		f"Voorbeelden:\n\n{topic_context}"
	)


def create_chain(model_name: str, temperature: float, keep_alive: str = "2m", reasoning: bool = True) -> ChatPromptTemplate:
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", "{system_instruction}"),
			MessagesPlaceholder(variable_name="messages"),
		]
	)
	llm = ChatOllama(model=model_name, 
					 temperature=temperature, 
					 keep_alive=keep_alive,
					 reasoning=reasoning)
	return prompt | llm			# Builds and returns a langchain pipeline -- first formats the prompt, then calls the llm


def start_conversation(
	level_prompt_path: str | Path,
	theme_prompt_path: str | Path,
	topic_file_path: str | Path,
	model_name: str,
	temperature: float,
	keep_alive: str = "2m",
	reasoning: bool = False,
	**kwargs
) -> str:
	level_prompt = read_text_file(Path(level_prompt_path))
	try:
		theme_prompt = read_text_file(Path(theme_prompt_path)).format(**kwargs)
	except KeyError:
		theme_prompt = read_text_file(Path(theme_prompt_path))
	topic_context = read_text_file(Path(topic_file_path))
	system_instruction = build_system_instruction(level_prompt, theme_prompt, topic_context)

	# print("\n--- SYSTEM INSTRUCTION ---")
	# print(system_instruction)
	
	chain = create_chain(model_name=model_name, temperature=temperature, keep_alive=keep_alive, reasoning=reasoning)
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
	model_name: str,
	temperature: float,
	keep_alive: str = "2m",
	reasoning: bool = False,
	**kwargs
) -> str:
	level_prompt = read_text_file(Path(level_prompt_path))
	try:
		theme_prompt = read_text_file(Path(theme_prompt_path)).format(**kwargs)
	except KeyError:
		theme_prompt = read_text_file(Path(theme_prompt_path))

	topic_context = read_text_file(Path(topic_file_path))
	system_instruction = build_system_instruction(level_prompt, theme_prompt, topic_context)
	
	# print("\n--- SYSTEM INSTRUCTION (generate reply) ---")
	# print(system_instruction)

	chain = create_chain(model_name=model_name, temperature=temperature, keep_alive=keep_alive, reasoning=reasoning)
	response = chain.invoke(
		{
			"system_instruction": system_instruction,		# System instruction needs to be passed for every reply call since LLM APIs completely forget any previous question ("stateless")
			"messages": list(messages),     # Includes both human and AI responses
		}
	)
	return response.content



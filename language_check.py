"""Language checking helpers for the NT2 conversation tool."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from language_check_utils.check_conjugation import find_conjugation_issue
from language_check_utils.check_wordorder import find_wordorder_issues


def read_text_file(path: Path) -> str:
	return path.read_text(encoding="utf-8")


def create_chain(model_name: str, temperature: float, keep_alive: str = "2m", reasoning: bool = False) -> ChatPromptTemplate:
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", "{system_instruction}"),
			MessagesPlaceholder(variable_name="messages"),
		]
	)
	llm = ChatOllama(
		model=model_name,
		temperature=temperature,
		keep_alive=keep_alive,
		reasoning=reasoning,
	)
	return prompt | llm


def check_manual_language_issue(
	sentence: str,
	rules: Sequence[str] | None = None,
	return_details: bool = False,
) -> Optional[str] | dict[str, object]:
	rule_set = {rule.strip().lower() for rule in (rules or ["order"]) if rule and rule.strip()}
	details: dict[str, object] = {
		"correct": True,
		"message": None,
		"error_type": None,
		"check": None,
		"checks": [],
	}
	issues: list[dict[str, object]] = []

	if "order" in rule_set or "woordvolgorde" in rule_set:          # woordvolgorde is Dutch spelling
		order_result = find_wordorder_issues(sentence)
		if not order_result["correct"]:
			issues.append(
				{
					"message": order_result["message"],
					"error_type": order_result.get("error_type"),
					"check": "word_order",
				}
			)

	if "conjugation" in rule_set or "vervoeging" in rule_set or "verb" in rule_set:             
		conjugation_issue = find_conjugation_issue(sentence)
		if conjugation_issue is not None:
			issues.append(
				{
					"message": conjugation_issue,
					"error_type": "CONJUGATION",
					"check": "conjugation",
				}
			)

	if issues:
		details.update(
			{
				"correct": False,
				"message": issues[0]["message"],
				"error_type": issues[0]["error_type"],
				"check": issues[0]["check"],
				"checks": [issue["check"] for issue in issues],
			}
		)
		return details if return_details else issues[0]["message"]

	return details if return_details else None


def check_llm_language_issue(
	checker_prompt_path: str | Path,
	question_text: str,
	answer_text: str,
	model_name: str,
	temperature: float = 0.0,
	keep_alive: str = "2m",
	reasoning: bool = False,
	**kwargs,
) -> str | None:
	if not checker_prompt_path or not model_name:
		return None

	checker_prompt = read_text_file(Path(checker_prompt_path))
	try:
		checker_prompt = checker_prompt.format(
			vraag=question_text,
			antwoord=answer_text,
			**kwargs,
		)
	except KeyError:
		checker_prompt = checker_prompt.format(
			vraag=question_text,
			antwoord=answer_text,
		)

	chain = create_chain(model_name=model_name, temperature=temperature, keep_alive=keep_alive, reasoning=reasoning)
	response_object = chain.invoke(
		{
			"system_instruction": checker_prompt,
			"messages": [HumanMessage(content="Check the last student sentence.")],
		}
	)
	response = response_object.content.strip()
	if not response:
		return None
	if response.upper() == "OK" or response.upper().startswith("OK"):
		return None
	return response

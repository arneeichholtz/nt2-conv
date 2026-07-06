"""Word-order language checks for NT2."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from language_check_utils.shared import nlp


COORD_CONJUNCTIONS = {"maar", "en", "want", "of"}
SUBORDINATORS = {"omdat", "als", "toen", "hoewel"}
SUBJECT_PRONOUNS = {"ik", "jij", "je", "u", "hij", "zij", "ze", "het", "wij", "we", "jullie"}
AUXILIARIES = {
	"ben",
	"bent",
	"is",
	"zijn",
	"was",
	"waren",
	"heb",
	"hebt",
	"heeft",
	"hebben",
	"had",
	"hadden",
	"zal",
	"zult",
	"zullen",
	"zou",
	"zouden",
	"kan",
	"kunt",
	"kunnen",
	"moet",
	"moeten",
	"mag",
	"mogen",
	"wil",
	"willen",
	"word",
	"wordt",
	"worden",
}

ADVERBS = {"morgen", "vandaag", "gisteren", "nu", "niet", "daar", "hier", "snel", "morgenavond"}
DETERMINERS = {"de", "het", "een", "geen", "deze", "die", "dit", "dat"}
PREPOSITIONS = {"in", "op", "aan", "bij", "van", "voor", "na", "met", "onder", "over", "tussen", "uit", "naar"}
DISCOURSE_MARKERS = {"nee", "ja", "nou", "ok", "oke", "okay", "uh", "eh"}
POSSESSIVE_PRONOUNS = {"mijn", "jouw", "je", "zijn", "haar", "ons", "onze", "jullie", "hun"}
KNOWN_VERBS = {
	"kom",
	"komt",
	"komen",
	"denk",
	"denkt",
	"denken",
	"gewas",
	"gewassen",
	"werk",
	"werkt",
	"werken",
	"ga",
	"gaat",
	"gaan",
	"maak",
	"maakt",
	"maken",
	"gegeten",
	"gegaan",
	"gekomen",
}


@dataclass(frozen=True)
class TokenInfo:
	text: str
	lower: str
	index: int
	pos: str
	dep: str = ""

	@property
	def is_punct(self) -> bool:
		return self.pos == "PUNCT"

	@property
	def is_coord_conj(self) -> bool:
		return self.pos == "CCONJ" or self.lower in COORD_CONJUNCTIONS

	@property
	def is_subordinator(self) -> bool:
		return (not self.is_coord_conj) and (self.pos == "SCONJ" or self.lower in SUBORDINATORS)

	@property
	def is_interjection(self) -> bool:
		return self.pos == "INTJ"

	@property
	def is_discourse_marker(self) -> bool:
		return self.lower in DISCOURSE_MARKERS

	@property
	def is_aux(self) -> bool:
		return self.pos == "AUX"

	@property
	def is_verb_like(self) -> bool:
		return self.pos in {"VERB", "AUX"}

	@property
	def is_subject_candidate(self) -> bool:
		return self.pos in {"PRON", "NOUN", "PROPN"}


def looks_like_verb(word: str) -> bool:
	if word in KNOWN_VERBS:
		return True

	if word.startswith("ge") and len(word) > 4:
		return True

	if word.endswith(("en", "t", "d")) and len(word) > 3 and word not in SUBJECT_PRONOUNS:
		return True

	return False


def classify_pos(text: str, spacy_pos: str = "") -> str:
	lower = text.lower()

	if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
		return "PUNCT"

	if lower in COORD_CONJUNCTIONS:
		return "CCONJ"
	if lower in SUBORDINATORS:
		return "SCONJ"

	if spacy_pos in {"VERB", "AUX", "NOUN", "PROPN", "PRON", "ADV", "ADP", "CCONJ", "SCONJ", "DET", "ADJ", "NUM", "PUNCT"}:
		return spacy_pos
	if lower in SUBJECT_PRONOUNS:
		return "PRON"
	if lower in AUXILIARIES:
		return "AUX"
	if lower in ADVERBS:
		return "ADV"
	if lower in DETERMINERS:
		return "DET"
	if lower in PREPOSITIONS:
		return "ADP"
	if looks_like_verb(lower):
		return "VERB"

	if text[:1].isupper() and lower not in SUBJECT_PRONOUNS:
		return "PROPN"

	return "NOUN"


def tokenize_sentence(sentence: str) -> list[TokenInfo]:
	doc = nlp(sentence)
	tokens: list[TokenInfo] = []
	for token in doc:
		if token.is_space:
			continue

		pos = classify_pos(token.text, token.pos_)
		dep = token.dep_ if token.dep_ else ""
		tokens.append(TokenInfo(text=token.text, lower=token.text.lower(), index=len(tokens), pos=pos, dep=dep))

	return tokens


def strip_leading_coordination(tokens: list[TokenInfo]) -> list[TokenInfo]:
	start = 0
	while start < len(tokens) and (
		tokens[start].is_punct
		or tokens[start].is_coord_conj
		or tokens[start].is_interjection
		or tokens[start].is_discourse_marker
	):
		start += 1
	return tokens[start:]


def main_clause_tokens(tokens: list[TokenInfo]) -> list[TokenInfo]:
	tokens = strip_leading_coordination(tokens)
	main_tokens: list[TokenInfo] = []

	for token in tokens:
		if token.is_punct or token.is_subordinator:
			break
		main_tokens.append(token)

	return main_tokens


def subordinate_clause_tokens(tokens: list[TokenInfo]) -> list[TokenInfo]:
	for index, token in enumerate(tokens):
		if token.is_subordinator:
			clause_tokens: list[TokenInfo] = []
			for following_token in tokens[index + 1 :]:
				if following_token.is_punct:
					break
				clause_tokens.append(following_token)
			return clause_tokens
	return []


def build_units(tokens: list[TokenInfo]) -> list[list[TokenInfo]]:
	units: list[list[TokenInfo]] = []
	index = 0

	while index < len(tokens):
		token = tokens[index]
		if token.is_punct or token.is_subordinator:
			break

		if token.pos == "ADP":
			unit = [token]
			index += 1
			while index < len(tokens):
				follower = tokens[index]
				if follower.is_punct or follower.is_subordinator or follower.is_coord_conj:
					break
				if follower.pos not in {"DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM"}:
					break
				unit.append(follower)
				index += 1
			units.append(unit)
			continue

		if token.pos == "PRON" and (token.dep == "nmod:poss" or token.lower in POSSESSIVE_PRONOUNS):
			unit = [token]
			index += 1
			while index < len(tokens):
				follower = tokens[index]
				if follower.is_punct or follower.is_subordinator or follower.is_coord_conj:
					break
				if follower.pos not in {"DET", "ADJ", "NOUN", "PROPN", "NUM", "PRON"}:
					break
				unit.append(follower)
				index += 1
			units.append(unit)
			continue

		if token.pos in {"DET", "ADJ", "NUM"}:
			unit = [token]
			index += 1
			while index < len(tokens):
				follower = tokens[index]
				if follower.is_punct or follower.is_subordinator or follower.is_coord_conj:
					break
				if follower.pos not in {"DET", "ADJ", "NOUN", "PROPN", "NUM"}:
					break
				unit.append(follower)
				index += 1
			units.append(unit)
			continue

		units.append([token])
		index += 1

	return units


def first_verb_unit_index(units: list[list[TokenInfo]]) -> Optional[int]:
	for unit_index, unit in enumerate(units):
		if any(token.is_verb_like for token in unit):
			return unit_index
	return None


def first_aux_index(tokens: list[TokenInfo]) -> Optional[int]:
	for index, token in enumerate(tokens):
		if token.is_aux:
			return index
	return None


def last_verb_like_index(tokens: list[TokenInfo]) -> Optional[int]:
	verb_indices = [index for index, token in enumerate(tokens) if token.is_verb_like]
	if not verb_indices:
		return None
	return verb_indices[-1]


def trailing_tokens_are_prepositional_phrase(tokens: list[TokenInfo]) -> bool:
	if not tokens:
		return False

	seen_adposition = False
	allowed_followers = {"DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM"}

	for token in tokens:
		if token.is_punct:
			continue

		if token.pos == "ADP":
			seen_adposition = True
			continue

		if not seen_adposition:
			return False

		if token.pos not in allowed_followers:
			return False

	return seen_adposition


def make_error(error_type: str, message: str) -> dict:
	return {"correct": False, "error_type": error_type, "message": message}


def check_subordinate_clause_rule(tokens: list[TokenInfo]) -> Optional[dict]:
	"""
	Bijzinnen (subordinate clauses) moeten de werkwoorden aan het einde hebben.
	Bijv: "Ik denk dat hij morgen komt" is correct, maar "Ik denk dat hij komt morgen" is incorrect.
	Implementatie: als onderschikkend voegwoord (subordinator) wordt herkend, moet het werkwoord (pv) aan einde van bijzin staan; zo niet, wordt er een error gerapporteerd.
	"""
	clause_tokens = subordinate_clause_tokens(tokens)
	if not clause_tokens:
		return None

	last_verb_index = last_verb_like_index(clause_tokens)
	if last_verb_index is None:
		return None

	trailing_tokens = [token for token in clause_tokens[last_verb_index + 1 :] if not token.is_punct]
	if trailing_tokens:
		return make_error(
			"SUBORDINATE_CLAUSE_VIOLATION",
			"Woordvolgorde in bijzin: er staat nog informatie na het werkwoord; in een bijzin moeten de werkwoorden aan het einde staan.",
		)

	return None


def check_verbal_end_sequence_rule(tokens: list[TokenInfo]) -> Optional[dict]:
	"""
	In een hoofdzin met een hulpwerkwoord (auxiliary) moet het hoofdwerkwoord aan het einde van de hoofdzin staan.
    Bijv: "Ik heb de auto gewassen" is correct, maar "Ik heb gewassen de auto" is incorrect.
	Implementatie: als hulpwerkwoord wordt herkend, moet het laatse woord in de zin een werkwoord zijn; zo niet, wordt er een error gerapporteerd. 
	"""
	main_tokens = main_clause_tokens(tokens)
	if not main_tokens:
		return None

	aux_index = first_aux_index(main_tokens)
	if aux_index is None:
		return None

	last_verb_index = last_verb_like_index(main_tokens)
	if last_verb_index is None or last_verb_index <= aux_index:
		return None

	trailing_tokens = [token for token in main_tokens[last_verb_index + 1 :] if not token.is_punct]
	if trailing_tokens and not trailing_tokens_are_prepositional_phrase(trailing_tokens):
		return make_error(
			"VERBAL_END_SEQUENCE_VIOLATION",
			"Woordvolgorde met hulpwerkwoord: het hoofdwerkwoord staat niet aan het einde van de hoofdzin.",
		)

	return None


def check_v2_rule(tokens: list[TokenInfo]) -> Optional[dict]:
	"""
	In een hoofdzin (main clause) moet de persoonsvorm op de tweede positie staan.
	Bijv: "Morgen kom ik niet" is correct, maar "Morgen ik kom niet" is incorrect.
    Implementatie: in hoofdzin moet het eerste werkwoord op de tweede positie staan; zo niet, wordt er een error gerapporteerd.
	"""
	main_tokens = main_clause_tokens(tokens)
	if len(main_tokens) < 2:
		return None

	if main_tokens and main_tokens[0].is_subordinator:
		return None

	if main_tokens[0].is_verb_like and len(main_tokens) > 1 and main_tokens[1].is_subject_candidate:
		return None

	verb_units = build_units(main_tokens)
	verb_unit_index = first_verb_unit_index(verb_units)
	if verb_unit_index is None:
		return None

	if verb_unit_index != 1:
		return make_error(
			"V2_VIOLATION",
			"Woordvolgorde in hoofdzin: de persoonsvorm staat niet op de tweede positie.",
		)

	return None


def find_wordorder_issues(sentence: str) -> dict:
	tokens = tokenize_sentence(sentence)

	for checker in (check_subordinate_clause_rule, check_verbal_end_sequence_rule, check_v2_rule):
		result = checker(tokens)
		if result is not None:
			return result

	return {"correct": True, "error_type": None, "message": "Woordvolgorde lijkt correct."}

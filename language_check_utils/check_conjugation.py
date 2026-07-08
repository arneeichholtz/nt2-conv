"""Verb conjugation language checks for NT2."""

from __future__ import annotations
from typing import Optional
from shared import nlp


FIRST_PERSON = {"ik"}
SECOND_PERSON = {"jij", "je", "u"}
THIRD_PERSON = {"hij", "zij", "ze", "het"}
SINGULAR_SUBJECTS = {"ik", "jij", "je", "u", "hij", "zij", "ze", "het"}
PLURAL_SUBJECTS = {"wij", "we", "jullie", "zij", "ze"}
ONREGELMATIGE_PRESENTE_VORMEN = {
	"zijn": {
		"1sg": {"ben"},
		"2sg_pre": {"bent"},
		"2sg_post": {"ben"},
		"3sg": {"is"},
		"plur": {"zijn"},
	},
	"hebben": {
		"1sg": {"heb"},
		"2sg_pre": {"hebt"},
		"2sg_post": {"heb"},
		"3sg": {"heeft"},
		"plur": {"hebben"},
	},
	"kunnen": {
		"1sg": {"kan"},
		"2sg_pre": {"kunt"},
		"2sg_post": {"kan"},
		"3sg": {"kan"},
		"plur": {"kunnen"},
	},
	"willen": {
		"1sg": {"wil"},
		"2sg_pre": {"wilt"},
		"2sg_post": {"wil"},
		"3sg": {"wil"},
		"plur": {"willen"},
	},
	"zullen": {
		"1sg": {"zal"},
		"2sg_pre": {"zult"},
		"2sg_post": {"zal"},
		"3sg": {"zal"},
		"plur": {"zullen"},
	},
	"mogen": {
		"1sg": {"mag"},
		"2sg_pre": {"mag"},
		"2sg_post": {"mag"},
		"3sg": {"mag"},
		"plur": {"mogen"},
	},
	"moeten": {
		"1sg": {"moet"},
		"2sg_pre": {"moet"},
		"2sg_post": {"moet"},
		"3sg": {"moet"},
		"plur": {"moeten"},
	},
}


def is_verb(token):
	return token.pos_ in {"VERB", "AUX"}


def is_finite_verb(token):
	return is_verb(token) and "Fin" in token.morph.get("VerbForm")


def determine_subject_person(subject):
	subject_text = subject.text.lower()
	if subject_text in FIRST_PERSON:
		return 1
	if subject_text in SECOND_PERSON:
		return 2
	if subject_text in THIRD_PERSON:
		return 3

	persons = subject.morph.get("Person")
	if "1" in persons:
		return 1
	if "2" in persons:
		return 2
	if "3" in persons:
		return 3

	return None


def determine_expected_form(subject, verb):
	if "Plur" in subject.morph.get("Number") or subject.text.lower() in {"wij", "we", "jullie"}:
		return "plur"

	person = determine_subject_person(subject)
	if person == 1:
		return "1sg"		# First person singular

	if person == 2:
		return "2sg_post" if subject.i > verb.i else "2sg_pre"

	return "3sg"


def is_singular_subject(subject):
	subject_text = subject.text.lower()
	return subject_text in SINGULAR_SUBJECTS or "Sing" in subject.morph.get("Number")


def is_plural_subject(subject):
	subject_text = subject.text.lower()
	return subject_text in PLURAL_SUBJECTS or "Plur" in subject.morph.get("Number")


def find_verb(subject):
	"""Find the verb that belongs to the subject."""
	def has_other_subject(token):
		for child in token.children:
			if child.dep_.startswith("nsubj") and child != subject:
				return True
		return False

	# Only finite verbs can be checked for conjugation.
	# This avoids false positives from infinitival clauses such as "er is ... te doen".
	if is_finite_verb(subject.head) and not has_other_subject(subject.head):
		return subject.head

	candidates = [token for token in subject.sent if is_finite_verb(token) and not has_other_subject(token)]

	if not candidates:
		return None

	def score(token):
		dependency_priority = {"ROOT": 0, "cop": 1, "aux": 2, "aux:pass": 3}.get(token.dep_, 4)
		finite_priority = 0 if is_finite_verb(token) else 1
		distance = abs(token.i - subject.i)
		head_bonus = 0 if token == subject.head else 1
		return finite_priority, head_bonus, distance, dependency_priority

	return min(candidates, key=score)


# def has_correct_conjugation_OLD(subject, verb):
# 	"""
# 	Bepaal of het werkwoord correct vervoegd is voor het onderwerp.
# 	- Controleert correcte meervoud/enkelvoud tussen onderwerp en werkwoord
# 	- Controleert correcte vervoeging van onregelmatige werkwoorden
# 	- Controleert correcte vervoeging van regelmatige werkwoorden in de tegenwoordige tijd
# 	- Controleert eerste persoon enkelvoud (ik) en werkwoord eindigt niet met -t
# 	- Controleert tweede persoon enkelvoud (jij/je/u) als inversie en werkwoord eindigt niet met -t (Loop jij?)
# 	"""
# 	if not is_finite_verb(verb):
# 		if "Inf" in verb.morph.get("VerbForm"):
# 			return False			
# 		return True

# 	if "Past" in verb.morph.get("Tense"):		# Checks plural/singular between subject and verb for past tense
# 		if "Plur" in verb.morph.get("Number"):
# 			return is_plural_subject(subject)
# 		return not is_plural_subject(subject)

# 	lemma = verb.lemma_.lower()
# 	text = verb.text.lower()
# 	verb_number = verb.morph.get("Number")

# 	# Check for regelmatige werkwoorden singular/plural agreement 
# 	if lemma not in ONREGELMATIGE_PRESENTE_VORMEN and is_singular_subject(subject) and ("Plur" in verb_number or text.endswith("en")):	
# 		return False

# 	expected_form = determine_expected_form(subject, verb)

# 	if lemma in ONREGELMATIGE_PRESENTE_VORMEN:
# 		return text in ONREGELMATIGE_PRESENTE_VORMEN[lemma][expected_form]

# 	if expected_form == "plur":
# 		return "Plur" in verb.morph.get("Number") or text.endswith("en")

# 	if expected_form == "1sg":
# 		return not text.endswith("t")

# 	if expected_form == "2sg_post":
# 		return not text.endswith("t")

# 	return text.endswith("t")


def has_correct_conjugation(subject, verb):
	"""This function checks for: subject-verb number agreement + 2nd/3rd person singular -t rule."""
	if not is_finite_verb(verb):
		return True		# Assume correct conjugation if not finite (e.g., "er is zo veel te doen.")

	subject_text = subject.text.lower()
	lemma = verb.lemma_.lower()
	verb_text = verb.text.lower()
	verb_number = verb.morph.get("Number")

	# Past tense: only check singular/plural agreement. Example: Ik wilde, wij wilden.
	if "Past" in verb.morph.get("Tense"):
		if "Plur" in verb_number or verb_text.endswith("en"):
			return is_plural_subject(subject)
		return not is_plural_subject(subject)

	# Irregular verbs use a fixed form per subject. Example: Ik ben.
	if lemma in ONREGELMATIGE_PRESENTE_VORMEN:
		expected_form = determine_expected_form(subject, verb)
		return verb_text in ONREGELMATIGE_PRESENTE_VORMEN[lemma][expected_form]

	# A plural subject should use a plural verb. Example: Wij werken.
	if is_plural_subject(subject):
		return "Plur" in verb_number or verb_text.endswith("en")

	# A singular subject (plural_subject=False) should not use a plural verb form. Example: Hij werkt.
	if "Plur" in verb_number or verb_text.endswith("en"):
		return False

	# 2nd person singular before the verb needs -t. Example: Jij werkt.
	if subject_text in SECOND_PERSON and subject.i < verb.i:
		return verb_text.endswith("t")

	# 2nd person singular after the verb (inversion) has no -t. Example: Werk jij?
	if subject_text in SECOND_PERSON and subject.i > verb.i:
		return not verb_text.endswith("t")

	# 1st person singular should NOT end with -t. Example: Ik werk.
	if subject_text in FIRST_PERSON:
		return not verb_text.endswith("t")

	# 3rd person singular MUST end with -t. Example: Hij werkt.
	if subject_text in THIRD_PERSON or determine_subject_person(subject) == 3:
		return verb_text.endswith("t")

	# Remaining cases pass after the checks above.
	return True


def could_be_verb(lemma: str, word: str) -> bool:
	"""Check if a word could plausibly be a conjugated form of a verb lemma."""
	lemma_lower = lemma.lower()
	word_lower = word.lower()
	
	# Exact match or already recognized irregular
	if lemma_lower in ONREGELMATIGE_PRESENTE_VORMEN:
		return True
	
	# Common regular verb forms:
	# - stem (1st person singular, infinitive): werk, hou, ga
	# - stem+t (3rd person singular, 2nd person): werkt, houdt, gaat
	# - stem+en (plural): werken, houden, gaan
	if word_lower == lemma_lower:
		return True
	if word_lower == lemma_lower + "t":
		return True
	if word_lower == lemma_lower + "en":
		return True
	if lemma_lower.endswith("en") and word_lower == lemma_lower[:-2]:  # infinitive to stem
		return True
	
	return False


def check_misparsed_verb_conjugation(subject, verb_word: str, verb_lemma: str) -> bool:
	"""Check conjugation for a word that may be misparsed as non-verb."""
	subject_text = subject.text.lower()
	verb_text = verb_word.lower()
	lemma = verb_lemma.lower()
	
	# Try to reconstruct the infinitive form
	# If the verb_text doesn't look like an infinitive, try adding -en
	if not lemma.endswith("en"):
		infinitive = lemma + "en"
	else:
		infinitive = lemma
	
	# Check irregular verbs using the reconstructed infinitive
	if infinitive in ONREGELMATIGE_PRESENTE_VORMEN:
		# Can't determine exact form for misparsed verbs, so check basic rules
		if is_plural_subject(subject):
			return verb_text in ONREGELMATIGE_PRESENTE_VORMEN[infinitive].get("plur", set())
		return verb_text in ONREGELMATIGE_PRESENTE_VORMEN[infinitive].get("3sg", set()) or \
		       verb_text in ONREGELMATIGE_PRESENTE_VORMEN[infinitive].get("1sg", set())
	
	# Regular verbs:
	# Plural subject should have plural form (ends with -en)
	if is_plural_subject(subject):
		return verb_text.endswith("en")
	
	# Singular subject:
	# 1st person: no -t (ik werk)
	if subject_text in FIRST_PERSON:
		return not verb_text.endswith("t")
	
	# 3rd person: must have -t (hij werkt)
	if subject_text in THIRD_PERSON:
		return verb_text.endswith("t")
	
	# 2nd person (rare in misparsed case, but handle it)
	if subject_text in SECOND_PERSON:
		return verb_text.endswith("t")
	
	return True


def find_conjugation_issue(sentence: str) -> Optional[str]:
	doc = nlp(sentence)
	
	# Primary check: find subjects and their verbs
	for token in doc:
		if not token.dep_.startswith("nsubj"):
			continue

		subject = token
		verb = find_verb(subject)
		if verb is None:
			# Fallback: check if the head (often misparsed as noun) could be a verb
			potential_verb = subject.head
			if potential_verb and potential_verb.pos_ in {"NOUN", "PROPN"} and potential_verb != subject:
				# Check if this could be a misparsed verb
				if could_be_verb(potential_verb.lemma_, potential_verb.text):
					if not check_misparsed_verb_conjugation(subject, potential_verb.text, potential_verb.lemma_):
						return "Deze zin bevat een fout in de werkwoordvervoeging, probeer het nog eens."
			continue

		if verb and not has_correct_conjugation(subject, verb):
			return "Deze zin bevat een fout in de werkwoordvervoeging, probeer het nog eens."

	return None


if __name__ == "__main__":


	examples = [
		"Hij werkt als automonteur.",		# correct
		"Wij werken als automonteurs.",		# correct
		"Ik werk als automonteur.",			# correct
		"Hij werk als automonteur.",		# incorrect
		"Wij werk als automonteur.",		# incorrect
		"Ik werken als automonteur.",		# incorrect
		"Ik werkt als automonteur."			# incorrect
	]

	for example in examples:
		result = find_conjugation_issue(example)
		if result:
			print(f"✗ {example}")
		else:
			print(f"✓ {example}")
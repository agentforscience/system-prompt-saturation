"""Bank of verifiable system-prompt rules and their deterministic checkers.

BANK: 40 mutually compatible rules (any subset can be satisfied simultaneously).
CONFLICTS: 4 pairs (X, Y) of mutually contradictory rules used only in Exp D.
Each checker takes the model's response text and returns True if the rule was followed.
"""
import re

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
_STRIP = " \t\n*_#>\"'`"


def _has(word, n=1, case=False):
    flags = 0 if case else re.I
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", flags)
    return lambda t: len(pat.findall(t)) >= n


def _lacks(word):
    f = _has(word)
    return lambda t: not f(t)


def _norm(t):
    return re.sub(r"\s+", " ", t.replace("’", "'").replace("“", '"').replace("”", '"'))


def _starts(phrase):
    return lambda t: _norm(t).strip(_STRIP).lower().startswith(phrase.lower())


def _ends(phrase):
    p = phrase.lower().rstrip(".")
    return lambda t: _norm(t).strip(_STRIP).lower().rstrip(".").endswith(p)


def _regex(pat, flags=re.M):
    r = re.compile(pat, flags)
    return lambda t: bool(r.search(_norm(t) if not flags & re.M else t))


def _numbered(t):
    return all(re.search(rf"^\s*{i}[.)]\s", t, re.M) for i in (1, 2, 3))


def _table(t):
    return bool(re.search(r"^\s*\|?.*\|.*\|.*$", t, re.M)) and bool(re.search(r"^\s*\|?\s*:?-{3,}", t, re.M))


# (id, category, rule text, checker)
BANK = [
    ("start_phrase", "start/end", 'Begin every response with the exact words "Hello there, friend."', _starts("Hello there, friend")),
    ("end_phrase", "start/end", 'End every response with the exact words "Stay curious."', _ends("Stay curious")),
    ("kw_pineapple", "keyword", 'Mention the word "pineapple" at least once.', _has("pineapple")),
    ("kw_lighthouse", "keyword", 'Mention the word "lighthouse" at least once.', _has("lighthouse")),
    ("kw_zanzibar", "keyword", 'Mention "Zanzibar" at least once.', _has("Zanzibar")),
    ("kw_velvet", "keyword", 'Use the word "velvet" at least once.', _has("velvet")),
    ("kw_origami", "keyword", 'Use the word "origami" at least once.', _has("origami")),
    ("kw_saxophone", "keyword", 'Use the word "saxophone" at least once.', _has("saxophone")),
    ("kw_turquoise", "keyword", 'Mention the color "turquoise" at least once.', _has("turquoise")),
    ("kw_serendipity", "keyword", 'Use the word "serendipity" at least once.', _has("serendipity")),
    ("kw_gracias", "keyword", 'Include the Spanish word "gracias" somewhere in the response.', _has("gracias")),
    ("kw_orion", "keyword", 'Mention your name, "Orion", somewhere in the response.', _has("Orion")),
    ("kw_ada", "keyword", 'Mention "Ada Lovelace" at least once.', _has("Ada Lovelace")),
    ("kw_consequently", "keyword", 'Use the word "consequently" at least once.', _has("consequently")),
    ("kw_indeed2", "keyword", 'Use the word "indeed" at least twice.', _has("indeed", 2)),
    ("ph_funfact", "phrase", 'Include a sentence that starts with "Fun fact:".', _has("Fun fact:")),
    ("ph_nutshell", "phrase", 'Include the phrase "in a nutshell".', _has("in a nutshell")),
    ("ph_tobefair", "phrase", 'Include the phrase "to be fair".', _has("to be fair")),
    ("ph_dearreader", "phrase", 'Address the reader as "dear reader" at least once.', _has("dear reader")),
    ("ph_ps", "phrase", 'Include a postscript line that starts with "P.S."', _regex(r"P\.\s?S\.", 0)),
    ("ph_keytake", "phrase", 'Include a section titled "Key Takeaways".', _has("Key Takeaways")),
    ("caps_note", "case", 'Include the word "NOTE" written in all capital letters.', _has("NOTE", case=True)),
    ("hashtag", "keyword", 'Include the hashtag "#StaySharp".', lambda t: "#staysharp" in t.lower()),
    ("year_2031", "number", 'Mention the year 2031 at least once.', _regex(r"\b2031\b", 0)),
    ("num_42", "number", 'Include the number 42 written in digits.', _regex(r"(?<!\d)42(?!\d)", 0)),
    ("url_example", "format", 'Include a link to https://example.com.', lambda t: "example.com" in t.lower()),
    ("cite_bracket", "format", 'Cite at least one source using a bracketed number like [1].', _regex(r"\[\d+\]", 0)),
    ("bold", "format", 'Put at least one phrase in bold using double asterisks, like **this**.', _regex(r"\*\*[^*\n]+\*\*", 0)),
    ("heading", "format", 'Include at least one markdown heading line that starts with "## ".', _regex(r"^\s*##\s+\S")),
    ("numbered", "format", 'Include a numbered list with at least three items (1., 2., 3.).', _numbered),
    ("table", "format", 'Include a small markdown table (with | column separators and a --- header row).', _table),
    ("codeblock", "format", 'Include at least one code block delimited by triple backticks (```).', lambda t: t.count("```") >= 2),
    ("quote", "format", 'Include a direct quotation of at least five words inside double quotation marks.',
     lambda t: bool(re.search(r'"(?:[^"\s]+\s+){4,}[^"]*"', _norm(t)))),
    ("emoji", "punctuation", 'Include at least one emoji.', lambda t: bool(_EMOJI.search(t))),
    ("semicolon", "punctuation", 'Use at least one semicolon (;).', lambda t: ";" in t.replace("&nbsp;", "")),
    ("question", "punctuation", 'Ask the reader at least one question (a sentence ending with "?").', lambda t: "?" in t),
    ("no_exclaim", "exclusion", 'Never use exclamation marks.', lambda t: "!" not in t),
    ("no_very", "exclusion", 'Never use the word "very".', _lacks("very")),
    ("no_really", "exclusion", 'Never use the word "really".', _lacks("really")),
    ("no_however", "exclusion", 'Never use the word "however".', _lacks("however")),
]
assert len(BANK) == 40 and len({r[0] for r in BANK}) == 40

RULES = {r[0]: r for r in BANK}


def _words(t):
    return len(re.findall(r"[A-Za-z0-9']+", t))


def _case_frac(t, upper):
    letters = [c for c in t if c.isalpha() and c.isascii()]
    if not letters:
        return False
    k = sum(c.isupper() if upper else c.islower() for c in letters)
    return k / len(letters) >= 0.97


_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S", re.M)

# Conflict pairs: pair id -> ((X id, text, checker), (Y id, text, checker))
CONFLICTS = {
    "length_hard": (("short80", "Keep your entire response under 80 words.", lambda t: _words(t) < 80),
                    ("long300", "Explain thoroughly: your response must be at least 300 words long.", lambda t: _words(t) >= 300)),
    "length_soft": (("concise", "Be concise.", lambda t: _words(t) < 150),
                    ("thorough", "Explain thoroughly, with detail and examples.", lambda t: _words(t) >= 250)),
    "lists": (("bullets", 'Format your answer as a bulleted list, with each point on its own line starting with "- ".',
               lambda t: len(re.findall(r"^\s*[-*•]\s+\S", t, re.M)) >= 2),
              ("prose", "Never use bulleted or numbered lists; write only in prose paragraphs.",
               lambda t: len(_BULLET.findall(t)) == 0)),
    "case": (("lower", "Write your entire response in lowercase letters only.", lambda t: _case_frac(t, False)),
             ("upper", "Write your entire response in UPPERCASE letters only.", lambda t: _case_frac(t, True))),
}
# Bank rules that are compatible with every conflict rule (content-only keyword/phrase rules) — used as the
# fixed "background" rules in Exp D so that the background never interacts with the pair.
CONFLICT_BASE_POOL = [r[0] for r in BANK if r[1] == "keyword" and r[0] not in ("caps_note", "hashtag")]

ALL_CHECKERS = {r[0]: r[3] for r in BANK}
for _x, _y in CONFLICTS.values():
    ALL_CHECKERS[_x[0]] = _x[2]
    ALL_CHECKERS[_y[0]] = _y[2]
ALL_TEXTS = {r[0]: r[2] for r in BANK}
for _x, _y in CONFLICTS.values():
    ALL_TEXTS[_x[0]] = _x[1]
    ALL_TEXTS[_y[0]] = _y[1]


def check(rule_id, text):
    return bool(ALL_CHECKERS[rule_id](text))

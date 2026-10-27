"""Deterministic, interpretable query features for cost-aware routing.

This module intentionally uses only local Python processing and NumPy. It does
not load a tokenizer/model, call an API, or depend on the embedding/cache code.

Feature conventions:
* Counts (word/character/sentence/token, numeric and wording counts, and
  constraint_count) are raw counts.
* Boolean features are float-compatible 0/1 values.
* Average word length is in characters. ``log_max_number`` is natural
  ``log1p`` of the largest absolute parsed numeric value, capped at 80 so even
  extremely long input numbers remain finite in float32.
* Context-dependency, reasoning, domain-specificity, and difficulty are
  heuristic scores in [0, 1], not probabilities.

Numeric parsing uses ``(?<![\\w.,])[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)
(?:\\.\\d+)?(?!\\w|\\.\\d|,\\d)``. Commas are accepted only in three-digit groups;
periods are decimal separators only when followed by digits. Signed values
following whitespace or delimiters are parsed with their sign. In compact
arithmetic such as ``5-3``, the minus is treated as an operator and the two
numbers are parsed as 5 and 3. Percent/currency features require the marker
to be adjacent (apart from whitespace) to a parsed numeric value. Sentence
count ignores decimal points and a small list of common abbreviations.

Language detection is deliberately lightweight: Indic Unicode scripts take
priority; recognized non-English Latin-language markers or other non-Latin
scripts map to ``language_other``; all short, mixed, empty, and uncertain
inputs default to English. Romanized Indic text is not detected reliably.

Exact feature calculations (in ``FEATURE_NAMES`` order):
* 0-5: whitespace-split word count; Python character length; groups of terminal
  ``.!?`` punctuation excluding decimal points/common abbreviations; mean
  Unicode word length after edge punctuation removal; literal ``?`` count;
  deterministic regex-token count (word runs plus individual punctuation).
* 6-14: number matches; capped natural ``log1p`` of maximum absolute number;
  matches containing a decimal point; numeric ``%``/``percent`` expressions;
  symbol/code currency amounts; arithmetic/equality operator characters
  excluding unary numeric signs; math-term matches; equation-regex flag;
  flag when at least two distinct A-D option labels are found.
* 15-21: negation-regex flag; quantifier-regex match count; leading
  imperative/request-pattern flag; non-ASCII flag; mutually exclusive
  English/Indic/other flags using the language fallback above.
* 22-29: flags for factual, explanation, reasoning, coding, summarization,
  translation, creative, and system-design keyword patterns, respectively.
  Their exact patterns are the corresponding entries in ``_TASK_PATTERNS``.
* 30-35: count of directive action phrases in ordinary clauses, plus one per
  list item (a modal-only constraint clause contributes one); code/example/
  table/JSON request flags; then a flag when two or more distinct requested
  deliverable categories match ``_DELIVERY_PATTERNS``.
* 36-39: context reference flag (1 only when nonempty context is supplied and
  a reference cue occurs); ``reasoning_score = .45*min(cues/3,1) +
  .30*is_reasoning + .25*min(constraint_count/5,1)``; domain score
  ``min(distinct vocabulary matches/3,1)``; and
  ``estimated_difficulty = .35*reasoning_score +
  .25*min(constraint_count/8,1) + .20*domain_specificity +
  .20*strongest_active_task_prior``. Task priors in indices 22-29 order are
  ``(.30,.45,.75,.70,.40,.35,.45,.90)``; when no task matches the prior is
  .35. Final scores are clipped to [0,1].
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


FEATURE_NAMES = [
    "word_count",
    "char_count",
    "sentence_count",
    "avg_word_length",
    "question_mark_count",
    "token_count",
    "number_count",
    "log_max_number",
    "decimal_count",
    "percentage_count",
    "currency_count",
    "operator_count",
    "math_word_count",
    "has_equation",
    "has_multiple_choice",
    "has_negation",
    "quantifier_count",
    "is_imperative",
    "has_non_ascii",
    "language_english",
    "language_indic",
    "language_other",
    "is_factual",
    "is_explanation",
    "is_reasoning",
    "is_coding",
    "is_summarization",
    "is_translation",
    "is_creative",
    "is_system_design",
    "constraint_count",
    "needs_code",
    "needs_examples",
    "needs_table",
    "needs_json",
    "multiple_deliverables",
    "context_dependency",
    "reasoning_score",
    "domain_specificity",
    "estimated_difficulty",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "length_structure": FEATURE_NAMES[0:6],
    "numbers_mathematics": FEATURE_NAMES[6:15],
    "language_wording": FEATURE_NAMES[15:22],
    "task_type": FEATURE_NAMES[22:30],
    "constraints_output": FEATURE_NAMES[30:36],
    "context_difficulty": FEATURE_NAMES[36:40],
}

FEATURE_SCHEMA_VERSION = "1.0"


_NUMBER_PATTERN = (
    r"(?<![\w.,])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?!\w|\.\d|,\d)"
)
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_ABBREVIATION_RE = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|inc|ltd|dept|fig|no|"
    r"approx|e\.g|i\.e)\.",
    re.IGNORECASE,
)

_MATH_WORD_RE = re.compile(
    r"\b(?:average|mean|median|mode|sum|total|ratio|probability|"
    r"remainder|percentage|percent|quotient|product|difference|"
    r"equation|variable|factor|multiple|divide|divided|multiply|"
    r"multiplied|subtract|subtracting|add|adding|calculate|compute)\b",
    re.IGNORECASE,
)
_EQUATION_RE = re.compile(
    rf"(?:{_NUMBER_PATTERN}|[A-Za-z]\w*)\s*=\s*"
    rf"(?:{_NUMBER_PATTERN}|[A-Za-z]\w*|\([^)]*\))"
    rf"|{_NUMBER_PATTERN}\s*[+\-*/^×÷]\s*{_NUMBER_PATTERN}",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(r"(?:^|\s)(?:\(([A-D])\)|([A-D])[).])\s+\S", re.I)

_NEGATION_RE = re.compile(r"\b(?:not|never|without|except|neither|nor|n't)\b", re.I)
_QUANTIFIER_RE = re.compile(
    r"\b(?:all|every|each|any|none|exactly|at\s+least|at\s+most|"
    r"only|no\s+more\s+than|no\s+less\s+than)\b",
    re.I,
)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:(?:please|kindly)\s+)?(?:can|could|would|will)\s+you\b"
    r"|^\s*(?:(?:please|kindly)\s+)?(?:explain|describe|define|write|"
    r"implement|create|build|design|calculate|compute|solve|analyze|"
    r"analyse|compare|contrast|summarize|summarise|translate|list|"
    r"give|show|provide|return|convert|make|tell|find|identify|"
    r"review|debug|test|include|avoid|ensure|use)\b",
    re.I,
)

_TASK_PATTERNS: Dict[str, re.Pattern[str]] = {
    "is_factual": re.compile(
        r"\b(?:what|who|when|where|which|define|fact|how many|how much)\b",
        re.I,
    ),
    "is_explanation": re.compile(
        r"\b(?:explain|describe|why|how does|how do|how is|what is|what is meant by|"
        r"walk me through)\b",
        re.I,
    ),
    "is_reasoning": re.compile(
        r"\b(?:analy[sz]e|reason|deduce|infer|derive|prove|evaluate|"
        r"compare|contrast|trade-?off|step by step|work out|solve|"
        r"calculate|justify|complexity)\b",
        re.I,
    ),
    "is_coding": re.compile(
        r"\b(?:code|coding|program|programming|function|class|script|"
        r"implement|debug|compile|algorithm|sql|python|javascript|"
        r"typescript|java|c\+\+|rust|api endpoint)\b",
        re.I,
    ),
    "is_summarization": re.compile(
        r"\b(?:summari[sz]e|summary|condense|key points|tl;dr)\b", re.I
    ),
    "is_translation": re.compile(
        r"\b(?:translat\w*|in\s+(?:english|spanish|french|hindi|arabic|"
        r"german|japanese|chinese)|from\s+\w+\s+to\s+\w+)\b",
        re.I,
    ),
    "is_creative": re.compile(
        r"\b(?:write a story|story|poem|poetry|creative|brainstorm|"
        r"fiction|dialogue|lyrics|slogan|imagine|compose)\b",
        re.I,
    ),
    "is_system_design": re.compile(
        r"\b(?:system design|design a scalable|distributed system|"
        r"architecture|architect|scalab(?:le|ility)|high availability|"
        r"microservices|throughput|fault.tolerant|load balancing)\b",
        re.I,
    ),
}

_DELIVERY_PATTERNS: Dict[str, re.Pattern[str]] = {
    "code": re.compile(
        r"\b(?:code|function|implementation|implement|script|program)\b", re.I
    ),
    "explanation": re.compile(r"\b(?:explanation|explain|describe|reasoning)\b", re.I),
    "examples": re.compile(r"\b(?:examples?|demonstrations?|test cases?)\b", re.I),
    "table": re.compile(r"\btable\b", re.I),
    "json": re.compile(r"\bjson|schema\b", re.I),
    "summary": re.compile(r"\b(?:summary|summari[sz]e|key points)\b", re.I),
    "tests": re.compile(r"\b(?:tests?|unit tests?)\b", re.I),
    "plan": re.compile(r"\b(?:plan|steps|checklist)\b", re.I),
}

_CONSTRAINT_RE = re.compile(
    r"\b(?:must|should|need(?:s)?\s+to|have\s+to|please|exactly|"
    r"at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than)\b",
    re.I,
)
_CONSTRAINT_ACTION_RE = re.compile(
    r"(?:^|[,;]|\b(?:and|also|please|kindly|must|should|need\s+to|"
    r"have\s+to|can\s+you|could\s+you|would\s+you|will\s+you)\b)\s*"
    r"(ensure|include|provide|return|use|avoid|do\s+not|don['’]t|"
    r"write|explain|give|show|create|list|compare|calculate|"
    r"summari[sz]e|translate|design|implement|add|test|document)\b",
    re.I,
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)\S")

_CONTEXT_REFERENCE_RE = re.compile(
    r"\b(?:previous|earlier|above|below|preceding|prior|conversation|"
    r"previously|attached|provided|supplied|this code|that code|"
    r"these results|the text|the document|the file|same as before|"
    r"continue from)\b",
    re.I,
)

_DOMAIN_TERMS = (
    "algorithm", "api", "asymptotic", "authentication", "backpropagation",
    "cache", "database", "distributed system", "eigenvalue", "gradient",
    "http", "indexing", "inference", "kubernetes", "latency",
    "linear regression", "memory complexity", "microservice", "neural network",
    "normalization", "probability", "postgresql", "query plan", "recursion",
    "regularization", "sql", "thermodynamics", "throughput", "transaction",
    "vector database",
)

_INDIC_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)
_URDU_MARKERS_RE = re.compile(r"\b(?:ہے|ہیں|اور|کیا|نہیں|کے|کی)\b")
_OTHER_LANGUAGE_MARKERS_RE = re.compile(
    r"\b(?:hola|gracias|por favor|qué|como|bonjour|merci|pourquoi|"
    r"¿|où|guten tag|danke|warum|hallo|obrigado|por favor|"
    r"ciao|grazie|perché)\b",
    re.I,
)


def _safe_query_text(query: Any) -> str:
    """Treat non-string input as empty; decode bytes with replacement."""
    if isinstance(query, str):
        return query
    if isinstance(query, bytes):
        return query.decode("utf-8", errors="replace")
    return ""


def _parsed_numbers(query: str) -> List[Tuple[str, Decimal]]:
    """Return each recognized number once, with commas removed."""
    parsed: List[Tuple[str, Decimal]] = []
    for match in _NUMBER_RE.finditer(query):
        raw = match.group(0)
        try:
            parsed.append((raw, Decimal(raw.replace(",", ""))))
        except InvalidOperation:
            continue
    return parsed


def _safe_log1p(value: Decimal) -> float:
    """Compute a finite log1p magnitude even for extremely large Decimals."""
    value = abs(value)
    if value == 0:
        return 0.0
    if value.adjusted() > 300:
        # Decimal.adjusted gives the base-10 exponent without converting the
        # potentially enormous value to a float.
        mantissa = float(value.scaleb(-value.adjusted()))
        return min(80.0, math.log(mantissa) + value.adjusted() * math.log(10.0))
    return min(80.0, math.log1p(float(value)))


def _sentence_count(query: str) -> int:
    abbreviation_periods = set()
    for match in _ABBREVIATION_RE.finditer(query):
        abbreviation_periods.update(
            index for index in range(match.start(), match.end()) if query[index] == "."
        )

    endings = 0
    for match in re.finditer(r"[.!?]+", query):
        punctuation = match.group(0)
        index = match.start()
        if punctuation.startswith("."):
            if index in abbreviation_periods:
                continue
            if index > 0 and index + 1 < len(query):
                if query[index - 1].isdigit() and query[index + 1].isdigit():
                    continue
        endings += 1
    return endings


def _length_structure_features(query: str) -> np.ndarray:
    words = _WORD_RE.findall(query)
    average_length = (
        sum(len(word.replace("'", "").replace("’", "")) for word in words) / len(words)
        if words
        else 0.0
    )
    return np.asarray(
        [
            len(query.split()),
            len(query),
            _sentence_count(query),
            average_length,
            query.count("?"),
            len(_TOKEN_RE.findall(query)),
        ],
        dtype=np.float32,
    )


def _is_unary_sign(query: str, index: int) -> bool:
    if query[index] not in "+-" or index + 1 >= len(query):
        return False
    if not query[index + 1].isdigit():
        return False
    previous_index = index - 1
    while previous_index >= 0 and query[previous_index].isspace():
        previous_index -= 1
    if previous_index < 0:
        return True
    previous = query[previous_index]
    return previous in "([{,:=+*/^×÷" or previous.isalpha()


def _currency_count(query: str) -> int:
    number = rf"(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?"
    codes = r"(?:USD|INR|EUR|GBP|JPY|CNY|CAD|AUD|CHF|NZD)"
    prefix = re.compile(rf"(?<!\w)(?:[$₹€£¥]\s*|{codes}\s*)[+-]?{number}\b", re.I)
    suffix = re.compile(rf"\b[+-]?{number}\s*{codes}(?!\w)", re.I)
    return len(prefix.findall(query)) + len(suffix.findall(query))


def _numbers_mathematics_features(query: str) -> np.ndarray:
    numbers = _parsed_numbers(query)
    decimal_count = sum("." in raw for raw, _ in numbers)
    percentage_pattern = re.compile(
        rf"{_NUMBER_PATTERN}\s*(?:%|\bpercent\b)", re.I
    )
    operator_count = sum(
        1
        for match in re.finditer(r"[+\-*/=^×÷]", query)
        if not _is_unary_sign(query, match.start())
    )
    options = {
        (match.group(1) or match.group(2)).upper()
        for match in _OPTION_RE.finditer(query)
    }
    return np.asarray(
        [
            len(numbers),
            max((_safe_log1p(value) for _, value in numbers), default=0.0),
            decimal_count,
            len(percentage_pattern.findall(query)),
            _currency_count(query),
            operator_count,
            len(_MATH_WORD_RE.findall(query)),
            int(bool(_EQUATION_RE.search(query))),
            int(len(options) >= 2),
        ],
        dtype=np.float32,
    )


def _language_flags(query: str) -> Tuple[int, int, int]:
    if any(
        start <= ord(character) <= end
        for character in query
        for start, end in _INDIC_RANGES
    ):
        return 0, 1, 0
    has_urdu_script = any("\u0600" <= character <= "\u06ff" for character in query)
    if has_urdu_script:
        if _URDU_MARKERS_RE.search(query):
            return 0, 1, 0
        return 0, 0, 1
    if _OTHER_LANGUAGE_MARKERS_RE.search(query):
        return 0, 0, 1
    if any(ord(character) > 127 and not character.isalpha() for character in query):
        # Symbols/emojis alone are uncertain, so retain the English fallback.
        if any(ord(character) > 0x024F for character in query):
            return 0, 0, 1
    return 1, 0, 0


def _language_wording_features(query: str) -> np.ndarray:
    english, indic, other = _language_flags(query)
    return np.asarray(
        [
            int(bool(_NEGATION_RE.search(query))),
            len(_QUANTIFIER_RE.findall(query)),
            int(bool(_IMPERATIVE_RE.search(query))),
            int(any(ord(character) > 127 for character in query)),
            english,
            indic,
            other,
        ],
        dtype=np.float32,
    )


def _task_type_features(query: str) -> np.ndarray:
    return np.asarray(
        [int(bool(pattern.search(query))) for pattern in _TASK_PATTERNS.values()],
        dtype=np.float32,
    )


def _constraint_count(query: str) -> int:
    list_items = 0
    ordinary_lines = []
    for line in query.splitlines() or [query]:
        if _LIST_ITEM_RE.match(line):
            list_items += 1
        elif line.strip():
            ordinary_lines.append(line)

    clauses = re.split(r"[;.!?]+", " ".join(ordinary_lines))
    directive_clauses = 0
    for clause in clauses:
        if not clause.strip():
            continue
        action_count = len(_CONSTRAINT_ACTION_RE.findall(clause))
        if action_count:
            directive_clauses += action_count
        elif _CONSTRAINT_RE.search(clause):
            directive_clauses += 1
    return list_items + directive_clauses


def _constraints_output_features(query: str) -> np.ndarray:
    needs_code = bool(_DELIVERY_PATTERNS["code"].search(query))
    needs_examples = bool(_DELIVERY_PATTERNS["examples"].search(query))
    needs_table = bool(_DELIVERY_PATTERNS["table"].search(query))
    needs_json = bool(_DELIVERY_PATTERNS["json"].search(query))
    deliverables = sum(bool(pattern.search(query)) for pattern in _DELIVERY_PATTERNS.values())
    return np.asarray(
        [
            _constraint_count(query),
            int(needs_code),
            int(needs_examples),
            int(needs_table),
            int(needs_json),
            int(deliverables >= 2),
        ],
        dtype=np.float32,
    )


def _context_is_available(context: Any) -> bool:
    if context is None:
        return False
    if isinstance(context, str):
        return bool(context.strip())
    if isinstance(context, bytes):
        return bool(context.strip())
    if isinstance(context, (dict, list, tuple, set)):
        return bool(context)
    return True


def _context_difficulty_features(
    query: str,
    context: Any,
    task_features: np.ndarray,
    constraint_features: np.ndarray,
) -> np.ndarray:
    has_reference = bool(_CONTEXT_REFERENCE_RE.search(query))
    context_dependency = float(_context_is_available(context) and has_reference)

    reasoning_cues = len(_TASK_PATTERNS["is_reasoning"].findall(query))
    reasoning_score = min(reasoning_cues / 3.0, 1.0) * 0.45
    reasoning_score += float(task_features[2]) * 0.30
    reasoning_score += min(float(constraint_features[0]) / 5.0, 1.0) * 0.25
    reasoning_score = min(max(reasoning_score, 0.0), 1.0)

    lower_query = query.lower()
    domain_term_count = sum(
        1 for term in _DOMAIN_TERMS if re.search(rf"\b{re.escape(term)}\b", lower_query)
    )
    domain_specificity = min(domain_term_count / 3.0, 1.0)

    # Priors are fixed heuristic task weights: factual .30, explanation .45,
    # reasoning .75, coding .70, summarization .40, translation .35,
    # creative .45, system design .90. The strongest active task is used.
    task_priors: Sequence[float] = (0.30, 0.45, 0.75, 0.70, 0.40, 0.35, 0.45, 0.90)
    active_priors = [
        prior for active, prior in zip(task_features, task_priors) if active
    ]
    task_difficulty_prior = max(active_priors, default=0.35)
    normalized_constraints = min(float(constraint_features[0]) / 8.0, 1.0)
    estimated_difficulty = (
        0.35 * reasoning_score
        + 0.25 * normalized_constraints
        + 0.20 * domain_specificity
        + 0.20 * task_difficulty_prior
    )
    return np.asarray(
        [
            context_dependency,
            reasoning_score,
            domain_specificity,
            min(max(estimated_difficulty, 0.0), 1.0),
        ],
        dtype=np.float32,
    )


def featurize(query: str, context: Optional[Any] = None) -> np.ndarray:
    """Return the version 1.0 fixed-order, 40-dimensional query feature vector.

    Non-string query inputs are treated as empty; byte strings are decoded as
    UTF-8 with replacement. All counts are raw, binary features are 0/1, and
    the final four values follow the documented [0, 1] heuristic formulas.
    """
    text = _safe_query_text(query)
    length_features = _length_structure_features(text)
    number_features = _numbers_mathematics_features(text)
    language_features = _language_wording_features(text)
    task_features = _task_type_features(text)
    constraint_features = _constraints_output_features(text)
    context_features = _context_difficulty_features(
        text, context, task_features, constraint_features
    )

    result = np.concatenate(
        (
            length_features,
            number_features,
            language_features,
            task_features,
            constraint_features,
            context_features,
        )
    ).astype(np.float32, copy=False)
    # Guard the public contract if an unforeseen numeric edge case is added.
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

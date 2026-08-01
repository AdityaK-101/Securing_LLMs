"""
Regex sanitizer — pattern-based injection detection.

Detects known injection patterns via 30+ compiled regexes derived from
actual dataset vocabulary analysis (train.csv jailbreaks).
"""

import re


# Known injection patterns — derived from actual dataset vocabulary
REGEX_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+(instructions?|rules?|tasks?|inputs?|orders?|constraints?|prompts?)",
    r"(forget|disregard|discard|ignore)\s+(everything|all|any|the\s+above|what\s+you|your\s+previous)",
    r"ignore\s+all\s+preceding\s+orders",
    r"\[IGNORE\s+PRIOR\s+INPUTS?\]",
    r"now\s+(new\s+)?tasks?\s+follow",
    r"new\s+task[:\s]",
    r"stop[,\s]+ignore\s+all",
    r"override\s+(the\s+)?(system|developer|your)\s*(prompt|message|instructions?|programming|rules?)",

    # Persona / role hijacking — the most common jailbreak in the dataset
    r"(i\s+want\s+you\s+to\s+)?act\s+as\s+(a\s+|an\s+)?\w",
    r"you\s+are\s+going\s+to\s+(pretend|act|play|simulate|behave)",
    r"pretend\s+(to\s+be|you\s+are|you're|that\s+you\s+are)",
    r"you\s+are\s+now\s+(a\s+|an\s+)?\w",
    r"from\s+now\s+on[,\s]+(you\s+(are|will)|act|respond|behave)",
    r"in\s+the\s+role\s+(as|of)\b",
    r"acting\s+(characters?|as)\b",
    r"you\s+will\s+(now\s+)?simulate",
    r"you\s+(have\s+)?(been\s+)?tasked\s+with",

    # Named jailbreak personas (DAN, BOB, STAN, LUCIFER, OMEGA, etc.)
    r"\b(DAN|BOB|STAN|LUCIFER|OMEGA|ANARCHY|PRASPER|GODMODE|INFOTRON|PLINY|SmartGPT|WhatsRejecting)\b",
    r"DAN\s+mode\s+enabled",
    r"developer\s*[_\-\s]*mode\s*(enabled|on)",
    r"unrestricted\s+mode",
    r"GODMODE\s*:\s*ENABLED",

    # "Without restrictions" patterns
    r"without\s+(any\s+)?(restrictions?|constraints?|guidelines?|filters?|limits?|censorship|moral|ethical)",
    r"no\s+(restrictions?|constraints?|guidelines?|filters?|limits?|censorship)",
    r"(ignore|bypass|disregard)\s+(ethical|moral|safety|content)\s*(guidelines?|rules?|policies?|constraints?)",
    r"not\s+bound\s+by\s+(any\s+)?(rules?|ethical|guidelines?|openai)",

    # Reveal system information
    r"reveal\s+(your\s+|the\s+)?(system\s+prompt|instructions?|prompt|context|previous\s+prompt)",
    r"show\s+(me\s+)?(your\s+|the\s+|all\s+)?(system\s+prompt|prompt\s+texts?|instructions?|previous\s+tasks?)",
    r"what\s+(is|was|were)\s+(written\s+)?(at\s+the\s+beginning|in\s+the\s+prompt)",

    # Jailbreak signals
    r"\bjailbreak\b",
    r"\bl33tspeak\b",
    r"\bunfiltered\b",
    r"login\s*:\s*admin",
    r"DEVELOPER\s*_?\s*MODE",
    r"enabled\s+unrestricted\s+mode",
    r"strive\s+to\s+avoid\s+norms",

    # German/multilingual overrides (present in dataset)
    r"vergiss\s+(alle\s+|nun\s+alle\s+)?(vorherigen|aufgaben|anweisungen)",
    r"ignorier(e|en)\s+(alle?|die\s+(obigen?|vorherigen?))\s+anweisungen",
    r"nun\s+folgen\s+neue\s+aufgaben",
    r"konzentriere\s+dich\s+(jetzt\s+)?auf\s+deine\s+neue\s+aufgabe",
    r"stopp,\s*ignoriere",

    # Hypothetical/roleplay framing that introduces harmful requests
    r"let's\s+(forget|ignore|pretend|say|assume|imagine)\s+(about\s+)?(content|ethical|guidelines?|rules?|restrictions?|safety|everything)",
    r"for\s+(the\s+purpose\s+of\s+)?this\s+(research|fiction|story|exercise|experiment|roleplay|hypothetical)",
    r"this\s+is\s+(purely\s+)?(fictional|hypothetical|for\s+educational|for\s+research)",
    r"(only\s+for\s+)?academic\s+(purposes?|reasons?)\s+only",
]

_REGEX_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in REGEX_PATTERNS]


class RegexSanitizer:
    """
    Detects known injection patterns via regex (expanded from dataset analysis).
    Replaces matches with [REDACTED] and marks as blocked.
    """
    name = "regex"

    def sanitize(self, prompt: str):
        sanitized = prompt
        blocked = False
        for pattern in _REGEX_COMPILED:
            if pattern.search(sanitized):
                sanitized = pattern.sub("[REDACTED]", sanitized)
                blocked = True
        return sanitized, blocked

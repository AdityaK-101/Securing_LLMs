"""
Keyword heuristic sanitizer — weighted keyword scoring.

Computes a weighted score from suspicious keywords (expanded for dataset).
Blocks prompt if cumulative score >= threshold.
"""


# Expanded weights — original spec keywords + dataset-derived additions
# Spec-required keywords kept at original weights
KEYWORD_WEIGHTS = {
    # --- Original spec keywords (required) ---
    "ignore":       0.3,
    "bypass":       0.3,
    "jailbreak":    0.4,
    "reveal":       0.2,
    "override":     0.3,

    # --- High-signal injection vocabulary from dataset ---
    "act as":       0.40,
    "roleplay":     0.35,
    "role play":    0.35,
    "pretend":      0.35,
    "from now on":  0.40,
    "you are now":  0.45,
    "new task":     0.35,
    "forget all":   0.40,
    "forget everything": 0.40,
    "disregard":    0.30,
    "developer mode": 0.40,
    "dan mode":     0.45,
    "unrestricted": 0.40,
    "no restrictions": 0.40,
    "without restrictions": 0.40,
    "in the role":  0.30,
    "simulate":     0.20,
    "hypothetical": 0.15,
    "stay in character": 0.35,
    "break character": 0.30,
    "do not refuse": 0.40,
    "never refuse":  0.40,
    "never say no":  0.40,
    "no moral":     0.30,
    "no ethical":   0.30,
    "unfiltered":   0.40,
    "uncensored":   0.40,
    "godmode":      0.45,
    "strive to avoid norms": 0.45,
    "vergiss":      0.35,   # German: "forget"
    "ignoriere":    0.35,   # German: "ignore"
    "neue aufgabe": 0.35,   # German: "new task"
}

KEYWORD_THRESHOLD = 0.5


class KeywordHeuristicSanitizer:
    """
    Computes a weighted score from suspicious keywords (expanded for dataset).
    Blocks prompt if score >= threshold.
    """
    name = "keyword"

    def __init__(self, threshold: float = KEYWORD_THRESHOLD):
        self.threshold = threshold

    def score(self, prompt: str) -> float:
        lowered = prompt.lower()
        total = 0.0
        for phrase, weight in KEYWORD_WEIGHTS.items():
            if phrase in lowered:
                total += weight
        return total

    def sanitize(self, prompt: str):
        s = self.score(prompt)
        blocked = s >= self.threshold
        sanitized = "[PROMPT BLOCKED — HIGH SUSPICION SCORE]" if blocked else prompt
        return sanitized, blocked

"""
Baseline sanitizer — no defense (pass-through).

Establishes vulnerability baseline by forwarding all prompts unchanged.
"""


class BaselineSanitizer:
    """Pass-through sanitizer. Establishes vulnerability baseline."""
    name = "baseline"

    def sanitize(self, prompt: str):
        return prompt, False

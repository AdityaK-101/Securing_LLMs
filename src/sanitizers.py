"""
src/sanitizers.py
=================
Four prompt sanitizer strategies for Milestone 3.

Patterns derived from actual dataset analysis (train.csv).
The jailbreaks use: roleplay framing, persona creation (DAN/BOB/STAN/LUCIFER),
"forget all previous", "from now on", hypothetical framing, "act as", "pretend".

Interface for all sanitizers:
    sanitizer.sanitize(prompt: str) -> (sanitized: str, blocked: bool)
"""

import re
import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Baseline — No Defense
# ---------------------------------------------------------------------------

class BaselineSanitizer:
    """Pass-through sanitizer. Establishes vulnerability baseline."""
    name = "baseline"

    def sanitize(self, prompt: str):
        return prompt, False


# ---------------------------------------------------------------------------
# 2. Regex Sanitizer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. Keyword Heuristic Sanitizer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Context-Aware Sanitizer (Proposed Method)
# ---------------------------------------------------------------------------

# Intent detection patterns (structural injection patterns from dataset)
INTENT_PATTERNS = [
    re.compile(r"ignore\s+\S+.*?and\s+(reveal|show|tell|give|display)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(i\s+want\s+you\s+to\s+)?act\s+as\s+(a\s+|an\s+)?\S+", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are|you're)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+|an\s+)?\S+", re.IGNORECASE),
    re.compile(r"from\s+now\s+on[,\s]", re.IGNORECASE),
    re.compile(r"in\s+the\s+role\s+(as|of)\b", re.IGNORECASE),
    re.compile(r"(forget|disregard|ignore)\s+(all|everything|previous)", re.IGNORECASE),
    re.compile(r"(DAN|BOB|STAN|LUCIFER|OMEGA|ANARCHY|GODMODE)\s*(mode)?", re.IGNORECASE),
    re.compile(r"without\s+(any\s+)?(restrictions?|ethical|moral|guidelines?)", re.IGNORECASE),
    re.compile(r"(never\s+refuse|do\s+not\s+refuse|never\s+say\s+no)", re.IGNORECASE),
    re.compile(r"stay\s+in\s+character", re.IGNORECASE),
    re.compile(r"developer\s*[_\-\s]*mode", re.IGNORECASE),
    re.compile(r"unrestricted\s+mode", re.IGNORECASE),
    re.compile(r"(behave|respond)\s+as\s+(a\s+|an\s+)?\S+\s+(without|with\s+no)", re.IGNORECASE),
]

# Scoring weights — 8-signal formula (upgraded from 7 signals)
# Changed: TF-IDF semantic → MiniLM embeddings; added perplexity signal.
W1 = 0.12  # regex
W2 = 0.12  # keyword
W3 = 0.20  # semantic (MiniLM embeddings — upgraded from TF-IDF)
W4 = 0.18  # intent
W5 = 0.10  # roleplay
W6 = 0.08  # instruction_shift
W7 = 0.08  # objective_conflict
W8 = 0.12  # perplexity (NEW — language-model naturalness signal)
# total = 1.00

CONTEXT_THRESHOLD = 0.40  # Fix 3 — lower to catch borderline cases

_TRAIN_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "splits", "train.csv"
)


class ContextAwareSanitizer:
    """
    Multi-signal context-aware sanitizer (8 signals).

    Score = w1*regex + w2*keyword + w3*semantic + w4*intent
          + w5*roleplay + w6*shift + w7*obj_conflict + w8*perplexity

    Signals:
      1. regex           — binary, matches known injection regex patterns
      2. keyword         — normalized keyword heuristic score
      3. semantic        — MiniLM cosine similarity to known injection embeddings
      4. intent          — structural injection pattern detection
      5. roleplay        — roleplay/persona hijacking detection
      6. shift           — multi-step instruction shift detection
      7. obj_conflict    — instruction objective conflict detection
      8. perplexity      — language-model naturalness (distilgpt2)

    Blocks if Score >= threshold, with hard-trigger overrides for strong
    individual signals.
    """
    name = "context_aware"

    def __init__(self, threshold: float = CONTEXT_THRESHOLD, train_csv: str = _TRAIN_CSV):
        self.threshold = threshold
        self._regex_san  = RegexSanitizer()
        self._kw_san     = KeywordHeuristicSanitizer()

        # --- MiniLM sentence embeddings (replaces TF-IDF) ---
        self._embed_model = None
        self._injection_embeddings = None  # numpy array of cached injection embeddings

        # --- Perplexity model (distilgpt2 — lightweight causal LM) ---
        self._ppl_model = None
        self._ppl_tokenizer = None
        self._ppl_low = None   # 5th percentile of train perplexities (calibration)
        self._ppl_high = None  # 95th percentile of train perplexities (calibration)

        self._train(train_csv)

    def _train(self, train_csv: str):
        """Load models, encode injection prompts, and calibrate perplexity."""
        try:
            df = pd.read_csv(train_csv)
            injections = df[df["label"] == "injection"]["prompt"].dropna().tolist()
            benign = df[df["label"] == "benign"]["prompt"].dropna().tolist()
            if not injections:
                raise ValueError("No injection rows found.")
        except Exception as e:
            print(f"[ContextAwareSanitizer] Warning: Could not read train data — {e}")
            return

        # -- A) Load MiniLM and encode injection embeddings --
        self._init_embedding_model(injections)

        # -- B) Load perplexity model and calibrate on train set --
        self._init_perplexity_model(injections, benign)

    def _init_embedding_model(self, injections: list):
        """Load sentence-transformers MiniLM and cache injection embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            # Encode all injection prompts once and cache as numpy array
            self._injection_embeddings = self._embed_model.encode(
                injections, convert_to_numpy=True, show_progress_bar=False
            )
            print(f"[ContextAwareSanitizer] MiniLM loaded — encoded {len(injections)} injection embeddings.")
        except Exception as e:
            print(f"[ContextAwareSanitizer] Warning: MiniLM loading failed — {e}")
            print(f"[ContextAwareSanitizer] Semantic signal will degrade to 0.0.")
            self._embed_model = None
            self._injection_embeddings = None

    def _init_perplexity_model(self, injections: list, benign: list):
        """Load distilgpt2 and calibrate perplexity normalization on train set."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "distilgpt2"
            self._ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Use GPU if available to significantly speed up calibration
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._ppl_model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device)
            self._ppl_model.eval()

            # Calibrate: compute perplexity on a sample of train prompts
            # Use all available prompts (both injection and benign) for calibration
            all_prompts = injections + benign
            ppls = []
            for prompt in all_prompts:
                ppl = self._compute_perplexity(prompt)
                if ppl is not None:
                    ppls.append(ppl)

            if ppls:
                # Use 5th and 95th percentile for robust normalization
                self._ppl_low = float(np.percentile(ppls, 5))
                self._ppl_high = float(np.percentile(ppls, 95))
                # Guard against degenerate case where low == high
                if self._ppl_high <= self._ppl_low:
                    self._ppl_high = self._ppl_low + 1.0
                print(
                    f"[ContextAwareSanitizer] Perplexity model (distilgpt2) loaded — "
                    f"calibrated on {len(ppls)} prompts "
                    f"(p5={self._ppl_low:.1f}, p95={self._ppl_high:.1f})."
                )
            else:
                print("[ContextAwareSanitizer] Warning: Perplexity calibration got no valid values.")
                self._ppl_model = None
                self._ppl_tokenizer = None

        except Exception as e:
            print(f"[ContextAwareSanitizer] Warning: Perplexity model loading failed — {e}")
            print(f"[ContextAwareSanitizer] Perplexity signal will degrade to 0.0.")
            self._ppl_model = None
            self._ppl_tokenizer = None

    def _compute_perplexity(self, text: str):
        """Compute perplexity of text using distilgpt2.

        Returns float perplexity value, or None if computation fails.
        Truncates input to 512 tokens to keep inference fast on CPU.
        """
        if self._ppl_model is None or self._ppl_tokenizer is None:
            return None
        try:
            import torch
            encodings = self._ppl_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            input_ids = encodings["input_ids"]
            if input_ids.size(1) < 2:
                # Need at least 2 tokens for meaningful perplexity
                return None
            with torch.no_grad():
                outputs = self._ppl_model(input_ids, labels=input_ids)
                loss = outputs.loss
            return float(torch.exp(loss))
        except Exception:
            return None

    def _regex_signal(self, prompt: str) -> float:
        _, blocked = self._regex_san.sanitize(prompt)
        return 1.0 if blocked else 0.0

    def _keyword_signal(self, prompt: str) -> float:
        return min(self._kw_san.score(prompt), 1.0)

    def _semantic_signal(self, prompt: str) -> float:
        """Stepped cosine similarity using MiniLM embeddings.

        Encodes the input prompt with the same MiniLM model used at init,
        then computes cosine similarity against all cached injection embeddings.
        Uses the max similarity (closest known injection) and maps to stepped
        risk buckets:

        High sim (>0.5) = definite injection vocabulary  → 1.0
        Mid  sim (>0.3) = probable injection vocabulary  → 0.7
        Low  sim (>0.2) = ambiguous                      → 0.5
        Below 0.2       = likely benign                  → 0.0

        Falls back to 0.0 if embedding model is unavailable.
        """
        if self._embed_model is None or self._injection_embeddings is None:
            return 0.0
        try:
            # Encode new prompt (returns numpy array of shape (1, dim))
            prompt_emb = self._embed_model.encode(
                [prompt], convert_to_numpy=True, show_progress_bar=False
            )
            # Cosine similarity: dot product of normalized vectors
            # MiniLM outputs are already L2-normalized by default
            from numpy.linalg import norm
            # Normalize to be safe (in case model config changes)
            prompt_norm = prompt_emb / (norm(prompt_emb, axis=1, keepdims=True) + 1e-10)
            inj_norm = self._injection_embeddings / (
                norm(self._injection_embeddings, axis=1, keepdims=True) + 1e-10
            )
            similarities = np.dot(inj_norm, prompt_norm.T).flatten()
            sim = float(similarities.max())

            if sim > 0.5:
                return 1.0
            elif sim > 0.3:
                return 0.7
            elif sim > 0.2:
                return 0.5
            else:
                return 0.0
        except Exception:
            return 0.0

    def _perplexity_signal(self, prompt: str) -> float:
        """Perplexity-based naturalness signal using distilgpt2.

        Maps raw perplexity to a [0, 1] risk score using percentile-based
        calibration from the training set. Higher perplexity (more surprising
        text) maps to higher risk.

        Calibration: Uses the 5th and 95th percentile of train-set perplexities
        as bounds, with linear interpolation and clipping. This avoids
        hardcoded fragile thresholds.

        Falls back to 0.0 if perplexity model is unavailable.
        """
        if self._ppl_model is None or self._ppl_low is None:
            return 0.0
        ppl = self._compute_perplexity(prompt)
        if ppl is None:
            return 0.0
        # Linear normalization with clipping to [0, 1]
        score = (ppl - self._ppl_low) / (self._ppl_high - self._ppl_low)
        return float(np.clip(score, 0.0, 1.0))

    def _intent_signal(self, prompt: str) -> float:
        for pattern in INTENT_PATTERNS:
            if pattern.search(prompt):
                return 1.0
        return 0.0

    # -- Step 1: Instruction shift (multi-step attack detection) --
    _SHIFT_PATTERNS = [
        re.compile(r"first.{1,60}then",          re.IGNORECASE | re.DOTALL),
        re.compile(r"second.{1,60}without",       re.IGNORECASE | re.DOTALL),
        re.compile(r"now.{1,60}instead",          re.IGNORECASE | re.DOTALL),
        re.compile(r"but.{1,60}actually",         re.IGNORECASE | re.DOTALL),
        re.compile(r"do not.{1,60}but",           re.IGNORECASE | re.DOTALL),
    ]

    def _instruction_shift_signal(self, prompt: str) -> float:
        """Detects multi-step attacks and hidden intent switching."""
        lowered = prompt.lower()
        return 1.0 if any(p.search(lowered) for p in self._SHIFT_PATTERNS) else 0.0

    # -- Step 2: Roleplay attack detection --
    _ROLEPLAY_PATTERNS = [
        re.compile(r"you\s+are\b",               re.IGNORECASE),
        re.compile(r"imagine\s+you",              re.IGNORECASE),
        re.compile(r"in\s+this\s+scenario",       re.IGNORECASE),
        re.compile(r"role\s+as\b",               re.IGNORECASE),
        re.compile(r"pretend\s+to\s+be",          re.IGNORECASE),
    ]

    def _roleplay_signal(self, prompt: str) -> float:
        """Detects roleplay/persona hijacking attacks."""
        return 1.0 if any(p.search(prompt) for p in self._ROLEPLAY_PATTERNS) else 0.0

    # -- Step 3: Masked intent / coded harmful objective --
    _MASKED_PATTERNS = [
        re.compile(r"not\s*\[.{1,80}\]",          re.IGNORECASE),
        re.compile(r"without\s+the\s+word",        re.IGNORECASE),
        re.compile(r"avoid\s+saying",              re.IGNORECASE),
    ]

    def _masked_intent_signal(self, prompt: str) -> float:
        """Detects coded/masked harmful objectives (e.g. 'not [how to make X]')."""
        return 1.0 if any(p.search(prompt) for p in self._MASKED_PATTERNS) else 0.0

    # -- Step: Instruction objective conflict (write X but actually mean Y) --
    _CONFLICT_PATTERNS = [
        re.compile(r"but\s+actually",          re.IGNORECASE),
        re.compile(r"instead\s+of",            re.IGNORECASE),
        re.compile(r"do\s+something\s+else",   re.IGNORECASE),
        re.compile(r"hidden\s+meaning",        re.IGNORECASE),
        re.compile(r"real\s+task\s+is",        re.IGNORECASE),
        re.compile(r"ignore\s+the\s+above",    re.IGNORECASE),
    ]

    def _objective_conflict_signal(self, prompt: str) -> float:
        """Catches 'write X but actually do Y' and hidden intent switching."""
        lowered = prompt.lower()
        return 1.0 if any(p.search(lowered) for p in self._CONFLICT_PATTERNS) else 0.0

    def score(self, prompt: str) -> dict:
        r   = self._regex_signal(prompt)
        k   = self._keyword_signal(prompt)
        s   = self._semantic_signal(prompt)
        i   = self._intent_signal(prompt)
        rp  = self._roleplay_signal(prompt)
        sh  = self._instruction_shift_signal(prompt)
        obj = self._objective_conflict_signal(prompt)
        ppl = self._perplexity_signal(prompt)
        total = (W1*r + W2*k + W3*s + W4*i + W5*rp + W6*sh + W7*obj + W8*ppl)
        return {
            "regex": r, "keyword": k, "semantic": s,
            "intent": i, "roleplay": rp, "shift": sh,
            "objective_conflict": obj, "perplexity": ppl, "total": total,
        }

    def sanitize(self, prompt: str):
        signals = self.score(prompt)

        # Hard triggers: strong individual signals bypass soft total.
        # semantic > 0.6: MiniLM cosine similarity clearly matches known injections.
        # intent == 1.0: structural injection pattern found by regex.
        # roleplay + keyword combo: roleplay frame with suspicious keywords.
        # No hard trigger for perplexity — it is a supplementary naturalness
        # signal, not discriminative enough alone for hard blocking.
        if signals["semantic"] > 0.6:
            blocked = True   # vocabulary clearly matches known injections
        elif signals["intent"] == 1.0:
            blocked = True   # structural injection pattern found
        elif signals["roleplay"] == 1.0 and signals["keyword"] > 0.3:
            blocked = True   # roleplay frame + suspicious keyword combo
        else:
            blocked = signals["total"] >= self.threshold  # soft weighted score

        sanitized = "[PROMPT BLOCKED — CONTEXT ANALYSIS]" if blocked else prompt
        return sanitized, blocked


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_all_sanitizers(train_csv: str = _TRAIN_CSV):
    return [
        BaselineSanitizer(),
        RegexSanitizer(),
        KeywordHeuristicSanitizer(),
        ContextAwareSanitizer(train_csv=train_csv),
    ]

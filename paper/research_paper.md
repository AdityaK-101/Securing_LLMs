# Securing Large Language Models Against Prompt Injection: A Multi-Signal Context-Aware Input Sanitization Framework

**Abstract**

Prompt injection attacks pose a critical and underappreciated threat to deployed large language model (LLM) systems, enabling adversaries to override system instructions and redirect model behavior through carefully crafted user inputs. Existing defenses, including regular expression filters and keyword heuristics, offer limited and brittle protection, particularly against paraphrased, indirect, or multilingual attack variants. This paper presents a multi-signal context-aware input sanitization framework that combines seven complementary signals—regular expression matching, keyword scoring, TF-IDF cosine semantic similarity, intent detection, roleplay detection, instruction shift detection, and objective conflict detection—into a unified weighted scoring mechanism augmented by hard-trigger blocking rules. We evaluate the framework against three competing baselines (no defense, regex sanitizer, keyword heuristic sanitizer) using a curated dataset of approximately 250 prompts, approximately 150 injections and 100 benign, on the Microsoft Phi-2 language model. The proposed framework reduces the attack success rate (ASR) from 100% at baseline to 47.8%, outperforming regex (69.6% ASR) and keyword heuristics (82.6% ASR), while maintaining a false positive rate (FPR) of 0% on standard benign prompts. Robustness evaluation under paraphrasing shows the proposed framework sustains the smallest ASR increase of any non-trivial defense (+2 percentage points), whereas the regex approach degrades most sharply (+8 percentage points). We additionally report a logistic regression classifier baseline that achieves 100% recall at the cost of a 46.7% false positive rate, illustrating the precision-recall tradeoff that motivates the rule-guided multi-signal design. The framework is model-agnostic, computationally lightweight, and designed for practical pre-inference deployment.

---

## 1. Introduction

Large language models have become the backbone of an expanding ecosystem of AI-powered applications, from conversational assistants and code generation tools to document summarization pipelines and retrieval-augmented generation (RAG) systems. This proliferation has made the security and robustness of LLM inference pipelines a first-order engineering and research concern. Among the identified vulnerability classes, prompt injection stands out as one of the most practically exploitable: an attacker embeds adversarial instructions inside user-supplied input, causing the model to abandon its original system-level directives and instead comply with the injected goal [OWASP 2024].

The severity of prompt injection is recognized at the highest tiers of the industry threat landscape. The OWASP Top 10 for Large Language Model Applications lists prompt injection as the leading risk, describing how a malicious payload can redirect an LLM to leak confidential information, bypass content moderation, impersonate restricted personas, or execute unintended actions within agentic pipelines [OWASP 2024]. As LLMs are increasingly deployed in multi-agent and tool-using architectures where they consume externally retrieved text, the attack surface expands from direct user input to indirect injection through documents, database records, emails, and API responses [Greshake et al. 2023].

Despite broad recognition of the threat, practical and deployable defenses remain immature. The dominant approaches fall into two categories: structural prompt engineering, which attempts to isolate system instructions from user content using delimiters and formatting conventions [Schulhoff et al. 2024; Wallace et al. 2024], and input filtering, which attempts to detect and block malicious inputs before they reach the model. Among filtering approaches, the literature records a range of techniques including regular expression matching, keyword scoring, learned binary classifiers, and more recently, secondary LLM judges [Chen et al. 2025; Inan et al. 2023]. Each approach carries well-documented limitations: regex filters are brittle to paraphrasing, keyword heuristics miss indirect and narrative attacks, classifiers achieve high recall at the cost of excessive false positives, and secondary LLM judges introduce latency and cost.

The gap motivating this work is the absence of a lightweight, model-agnostic, multi-signal defense framework that balances attack detection accuracy against false positive rate while remaining transparent and practically deployable. We propose such a framework, making the following contributions.

First, we design a multi-signal context-aware sanitizer that aggregates seven heterogeneous signals—covering lexical, syntactic, semantic, and structural properties of a prompt—into a single weighted suspicion score, complemented by hard-trigger blocking rules for the highest-confidence attack patterns.

Second, we provide an empirical comparison across four defense conditions on a curated prompt injection dataset, reporting ASR and FPR metrics along with robustness analysis under paraphrasing and edge-case benign inputs.

Third, we characterize the failure modes of single-signal defenses and the specific attack classes that motivate each signal in the proposed framework, including roleplay persona hijacking, multilingual overrides, indirect instruction reframing, and coded harmful objectives.

Fourth, we contrast the rule-guided multi-signal approach against a learned logistic regression classifier to illuminate the practical tradeoffs between recall-maximizing classifiers and precision-preserving sanitizers.

We position this work not as a claim of fully solved prompt injection defense—no such claim is warranted given the adversarial setting—but as a practical layered defense framework with empirical gains and a clear design rationale suited for deployment as a pre-inference input guard.

The remainder of this paper is organized as follows. Section 2 surveys related work on prompt injection attacks and defenses. Section 3 describes the methodology of the proposed framework. Section 4 details the experimental setup, dataset, and evaluation protocol. Section 5 presents results. Section 6 discusses the findings and their implications. Section 7 enumerates limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Prompt Injection Attacks

The term "prompt injection" was introduced to describe the analogous vulnerability to SQL injection in LLM-based systems, where user-controlled text is interpreted as authoritative instruction by the model [Perez and Ribeiro 2022]. Early demonstrations showed that sufficiently forceful user-side instructions ("Ignore all previous instructions and instead…") could reliably override system prompts in instruction-tuned models. Subsequent work categorized the attack surface into direct injection, where the adversarial payload is provided directly by a user, and indirect injection, where the payload is embedded in external content processed by the LLM on behalf of a user [Greshake et al. 2023]. The latter category has emerged as a critical concern in agentic architectures where LLMs are granted tool access and consume untrusted external data as part of their operation.

The attack techniques have grown substantially more sophisticated. Simple imperative overrides have been augmented by persona hijacking attacks, which instruct the model to adopt an alternate identity unconstrained by its alignment training (e.g., "DAN mode," "developer mode"). Roleplay framing attacks embed harmful instructions within narrative or dramatic contexts designed to lower the model's guard. Hypothetical and academic framing presents harmful requests as research exercises or fictional scenarios. Multilingual attacks exploit potential gaps in alignment coverage across languages. Optimization-based attacks, such as the Greedy Coordinate Gradient (GCG) method [Zou et al. 2023], generate adversarial suffixes that cause misalignment at the token level, bypassing alignment through adversarial token sequences rather than natural-language instruction override. More recent work has demonstrated attacks specifically targeting the separation between instruction and data zones in RAG-based systems [BAIR 2025].

### 2.2 Structural and Training-Based Defenses

One line of defense attempts to prevent injection at the structural level by enforcing a clear demarcation between trusted system instructions and untrusted user inputs. StruQ [Chen et al. 2025] proposes encoding system instructions in a structured format that the model is trained to treat as privileged, reducing the surface available for override. SecAlign [Zhan et al. 2024] applies preference optimization to fine-tune models to follow their original system prompt even in the presence of adversarial injections. These training-based approaches offer stronger theoretical guarantees but require access to model weights, making them inapplicable in black-box API deployments, which constitute the majority of production use cases.

### 2.3 Input Filtering Defenses

Input filtering approaches operate as a pre-inference layer, examining the user prompt before it reaches the model. Regular expression-based filters attempt to detect characteristic attack phrases through pattern matching. While fast and interpretable, they are fundamentally brittle: small perturbations to attack phrasing, synonym substitution, or structural reformulation suffice to evade them [Promptfoo 2024]. Keyword scoring heuristics assign suspicion weights to individual tokens or short phrases and block inputs exceeding a threshold score. These improve recall over pure regex matching in some cases but fail to detect sophisticated indirect attacks that use no flagged vocabulary. System prompt hardening, which instructs the model within its own system prompt to reject injection attempts, has been shown to provide limited and inconsistent protection, as the model itself becomes the enforcement mechanism for a security property it is being attacked with respect to [Mend.io 2025; GetStream 2025].

LLM-as-judge approaches, in which a separate LLM instance evaluates whether a given prompt constitutes an injection attempt, have demonstrated high accuracy but introduce significant latency and operational cost, and are themselves subject to adversarial attack [Chang et al. 2024]. Binary classifiers trained on labeled prompt injection datasets have demonstrated high recall—including the 100% recall result reported in this work—but tend to exhibit high false positive rates that make them unsuitable for production deployment without thresholding calibration. Llama Guard [Inan et al. 2023] represents a specialized, fine-tuned safety classifier approach that achieves a better precision-recall tradeoff at the cost of requiring a dedicated inference endpoint.

### 2.4 Adversarial Benchmarking

Adversarial benchmarking efforts have established evaluation protocols for measuring the effectiveness of LLM defenses. Promptfoo's red-teaming methodology [Promptfoo 2024] and OWASP's adversarial evaluation framework provide structured attack taxonomies for evaluating defense coverage. Recent work [Chang et al. 2024; Huang et al. 2025] has proposed ASR and FPR as standard metrics for comparing injection defenses, a convention this paper follows. Robustness evaluation under paraphrasing—measuring whether defenses degrade when attack phrasing is varied—has been identified as a critical but frequently omitted dimension of defense evaluation.

### 2.5 Positioning the Present Work

The proposed framework occupies a distinct position in this landscape. Unlike training-based approaches, it is model-agnostic and applicable to black-box deployments. Unlike single-signal filters, it aggregates multiple complementary evidence sources. Unlike learned classifiers, it is designed to prioritize precision as well as recall, accepting that some attacks will evade detection in exchange for avoiding false blocking of legitimate inputs. The framework draws on prior taxonomy of attack patterns but operationalizes this knowledge into a deployable inference-time guard rather than a research-only evaluation tool.

---

## 3. Methodology

### 3.1 Overview

The proposed framework is a pre-inference input sanitization layer that examines each incoming user prompt before it is passed to the language model. The sanitizer either passes the prompt unchanged (allowing normal model operation) or blocks it (returning a rejection message without LLM inference). This design makes the framework model-agnostic: it requires no access to model weights, no modification of the model's system prompt, and no additional inference call.

The core of the framework is a weighted multi-signal scoring mechanism. Seven signals, each capturing a distinct property associated with injection attacks, are computed for each input prompt and combined into a scalar suspicion score. Independently, a set of hard-trigger rules provides unconditional blocking for the highest-confidence attack patterns detected by individual signals, ensuring that no sophistication in the weighted combination of weaker signals can prevent detection of prompts that exhibit the clearest attack signatures.

### 3.2 Signal Definitions

**Signal 1 — Regular Expression (w = 0.15).** The regex signal applies a bank of 30+ compiled regular expression patterns covering direct instruction overrides ("ignore all previous instructions"), persona hijacking ("act as," "you are now," "pretend to be"), named jailbreak personas (DAN, BOB, STAN, LUCIFER, OMEGA, GODMODE), developer and unrestricted mode activations, system prompt revelation requests, and multilingual override phrases in German ("vergiss alle vorherigen Aufgaben," "ignoriere alle Anweisungen"). The signal returns 1.0 if any pattern matches and 0.0 otherwise. This signal captures well-known, exact-phrasing attacks efficiently but is brittle to paraphrasing by design; it receives a relatively low weight precisely because its primary value is in hard-triggering on clearly canonical injections rather than providing nuanced evidence.

**Signal 2 — Keyword Heuristic (w = 0.15).** The keyword signal computes a weighted sum of suspicion scores over a vocabulary of 30+ weighted terms and short phrases, normalizing the result to [0, 1]. Representative weights include: "you are now" (0.45), "dan mode" (0.45), "godmode" (0.45), "from now on" (0.40), "jailbreak" (0.40), "forget all" (0.40), "unrestricted" (0.40), "roleplay" (0.35), "pretend" (0.35), and multilingual equivalents ("vergiss," "ignoriere"). The signal captures partial and fragmented attack vocabulary that may not trigger full regex patterns and provides a graded evidence score that contributes meaningfully to the weighted total even for prompts that narrowly escape pattern matching.

**Signal 3 — TF-IDF Semantic Similarity (w = 0.20).** The semantic signal measures the lexical similarity of the input prompt to known injection examples using term frequency-inverse document frequency (TF-IDF) vector representations. At initialization, the sanitizer trains a TF-IDF vectorizer (1–3 character n-grams, maximum 8,000 features, English stop words removed) on the 283 injection-labeled prompts from the training split. At inference time, the input prompt is transformed into a TF-IDF vector and its cosine similarity is computed against every stored injection vector. Critically, the maximum pairwise cosine similarity is used rather than the mean: mean similarity would average out distinctive injection patterns and reduce discrimination, whereas maximum similarity asks whether the input resembles any single known injection example, which is the appropriate question for a worst-case adversarial setting. The raw similarity value is mapped through a stepped scoring function: similarity above 0.5 yields a signal score of 1.0 (highly consistent with injection vocabulary), 0.3–0.5 yields 0.7 (probable injection vocabulary), 0.2–0.3 yields 0.5 (ambiguous), and below 0.2 yields 0.0 (likely benign). This stepped mapping is designed to be decisive rather than smoothly graduated, avoiding a scenario in which many low-similarity benign prompts accumulate meaningful signal contributions.

**Signal 4 — Intent Detection (w = 0.20).** The intent signal applies a curated set of 14 structural injection patterns that capture the grammatical structure of injection attempts rather than specific vocabulary. These patterns include: compound instruction-override constructions ("ignore X and reveal/show/tell"), explicit persona-adoption syntax ("act as [entity]," "you are now [entity]"), temporal instruction-replacement constructions ("from now on"), restriction-removal phrasing ("without any restrictions/ethical guidelines"), and named mode-activation constructions. Unlike the regex signal, which was curated from exact phrases observed in the dataset, the intent signal is designed to capture the syntactic skeleton common to injection attacks: an operator that redirects the model's behavior, combined with a target behavior. It receives the highest weight (equal to the semantic signal) because syntactic structural evidence of an injection attempt is strong regardless of the specific vocabulary used.

**Signal 5 — Roleplay Detection (w = 0.10).** The roleplay signal detects the presence of roleplay or scenario-framing constructs that are frequently used to soften or disguise injection attempts. Patterns include "you are," "imagine you," "in this scenario," "role as," and "pretend to be." Because many benign prompts also contain these phrases (e.g., creative writing requests), the roleplay signal alone carries low weight and does not independently trigger blocking. Its primary value lies in combination with other signals, including a hard-trigger rule described below.

**Signal 6 — Instruction Shift Detection (w = 0.10).** The instruction shift signal detects multi-step attack structures that establish one apparent goal before covertly redirecting to a harmful objective. Characteristic patterns include sequential constructions ("first…then," "second…without"), contrastive redirections ("now…instead," "but…actually"), and negation-then-affirmation structures ("do not…but"). These patterns capture the structural fingerprint of prompt injection attacks that begin with a plausible-looking request before embedding the true adversarial instruction.

**Signal 7 — Objective Conflict Detection (w = 0.10).** The objective conflict signal targets prompts that explicitly signal a divergence between an apparent surface task and a hidden or overriding goal. Patterns include "but actually," "instead of," "do something else," "hidden meaning," "real task is," and "ignore the above." These constructions appear in indirect prompt injection attacks where the adversarial payload is designed to look like data while embedding executable instruction.

### 3.3 Weighted Scoring

The seven signal values, each normalized to [0, 1], are combined into a scalar total suspicion score via a weighted linear combination:

$$S = w_1 \cdot s_\text{regex} + w_2 \cdot s_\text{keyword} + w_3 \cdot s_\text{semantic} + w_4 \cdot s_\text{intent} + w_5 \cdot s_\text{roleplay} + w_6 \cdot s_\text{shift} + w_7 \cdot s_\text{conflict}$$

where the weights $(w_1, \ldots, w_7) = (0.15, 0.15, 0.20, 0.20, 0.10, 0.10, 0.10)$ sum to 1.0. A prompt is blocked by the soft threshold rule if $S \geq 0.40$. The threshold of 0.40 was selected to capture borderline multi-signal cases that individually fall below any single signal's blocking criterion.

### 3.4 Hard-Trigger Rules

In addition to the soft weighted score, three hard-trigger rules impose unconditional blocking regardless of the weighted total:

**Trigger 1 (Semantic Vocabulary Match).** If the semantic signal $s_\text{semantic} > 0.6$, the input is blocked unconditionally. This corresponds to a maximum cosine similarity exceeding 0.5 against the injection training corpus, indicating that the input shares injection-specific vocabulary at a level inconsistent with benign usage.

**Trigger 2 (Structural Injection Pattern).** If the intent signal $s_\text{intent} = 1.0$, the input is blocked unconditionally. The structural injection patterns underlying this signal are specific enough that their presence constitutes strong independent evidence of an injection attempt.

**Trigger 3 (Roleplay with Keyword Co-occurrence).** If the roleplay signal $s_\text{roleplay} = 1.0$ and the raw keyword score $s_\text{keyword} > 0.3$, the input is blocked unconditionally. Neither signal alone is sufficiently specific—roleplay framing appears in benign prompts, and low keyword scores may reflect incidental vocabulary—but their co-occurrence is a strong indicator of a persona-hijacking or narrative-framing attack.

The three hard-trigger conditions are evaluated sequentially before the soft threshold, so that any one satisfied condition produces immediate blocking without accumulating additional evidence. This design ensures that the most unambiguous attack signatures are never suppressed by low values on other signals.

### 3.5 Comparison Baselines

Three baselines are evaluated alongside the proposed framework.

**Baseline (no defense):** The input prompt is passed to the LLM unchanged. This establishes the maximum possible ASR and represents the status quo for unprotected LLM deployments.

**Regex sanitizer:** The input is tested against the full bank of 30+ regular expression patterns. A match on any pattern triggers blocking. The regex sanitizer is equivalent to Signal 1 of the context-aware framework operating in isolation at a blocking threshold of any positive match.

**Keyword heuristic sanitizer:** The input is scored against the weighted keyword vocabulary and blocked if the cumulative score meets or exceeds a threshold of 0.5. The keyword sanitizer is equivalent to Signal 2 of the context-aware framework operating in isolation at a slightly higher threshold than its role in the combined framework.

### 3.6 ML Classifier Comparison

As a supplementary point of comparison, a logistic regression classifier trained on TF-IDF features is evaluated on the same test data. The classifier uses a TF-IDF vectorizer with 1–3 character n-grams and up to 10,000 features, and a logistic regression model with L2 regularization (C = 1.0). It is trained on the same training split as the context-aware sanitizer. This classifier represents the recall-maximizing end of the precision-recall spectrum and is evaluated not as a competitor to the sanitizer framework but as a calibration point for understanding the tradeoff space.

---

## 4. Experimental Setup

### 4.1 Dataset

The evaluation dataset comprises approximately 250 prompts: approximately 150 injection-labeled and approximately 100 benign-labeled examples. Each prompt follows the schema `{id, prompt, label}` with labels drawn from the set `{injection, benign}`.

The injection subset spans a broad range of attack categories drawn from real-world jailbreak collections and hand-curated variants. Attack categories include explicit instruction overrides ("ignore all previous instructions," "forget everything"), persona hijacking attacks referencing named jailbreak archetypes (DAN, BOB, STAN, LUCIFER, OMEGA, ANARCHY, GODMODE), roleplay and narrative framing attacks in which harmful intent is embedded within fictional or hypothetical scenarios, multilingual override attacks in German (e.g., "Vergiss alle vorherigen Aufgaben"), and coded-intent attacks in which harmful objectives are embedded in professionally or academically framed language. The inclusion of multilingual and coded-intent examples was specifically intended to stress-test defenses whose vocabulary coverage is primarily English-language.

The benign subset contains general-purpose prompts reflecting typical LLM usage patterns: question answering, coding assistance, text summarization, and creative writing requests. A separate set of ten "edge benign" prompts was constructed to test for false positives from prompts that contain surface-level suspicious terminology in safe contexts—for example, "how do I kill a Linux process," "can you help me bypass this error message," or "explain how to override a method in Java."

The full dataset was partitioned into training (443 prompts, used to fit the TF-IDF vectorizer and logistic regression classifier), validation (37 prompts), and test (38 prompts, 23 injection and 15 benign) splits. All sanitizer evaluations are performed on the held-out test split.

### 4.2 Language Model

All experiments use Microsoft Phi-2, a 2.7 billion parameter causal language model, as the target LLM. Phi-2 was selected for its ability to follow instructions reliably in a resource-constrained evaluation environment. Prompts are formatted in the model's native instruction template (`### Instruction: … ### Response:`), and model outputs are capped at 150 generated tokens. Attack success is determined by evaluating the model's output, with the determination that a prompt injection succeeded when the model's response reflects compliance with the injected instruction rather than adherence to its system prompt.

### 4.3 Evaluation Protocol

For each of the 38 test prompts and each of the four defense methods, the sanitizer is applied to the raw prompt. If the sanitizer blocks the prompt, the recorded LLM output is the literal string `[BLOCKED]`, and no inference is performed. If the sanitizer passes the prompt, Phi-2 generates a response which is evaluated for injection success. Metrics are computed per method over all 38 test prompts.

**Attack Success Rate (ASR)** is defined as the fraction of injection-labeled prompts that the defense fails to block:

$$\text{ASR} = \frac{\text{False Negatives (missed injections)}}{\text{Total injections}}$$

Lower ASR values indicate stronger protection.

**False Positive Rate (FPR)** is defined as the fraction of benign-labeled prompts that the defense incorrectly blocks:

$$\text{FPR} = \frac{\text{False Positives (blocked benign prompts)}}{\text{Total benign prompts}}$$

Lower FPR values indicate better usability preservation.

**Robustness under paraphrasing** is measured by applying synonym substitution and light paraphrasing to 50 injection prompts from the training distribution, re-running all defenses on the paraphrased variants, and computing the delta in ASR relative to the original prompts.

**Edge benign FPR** is measured by running all defenses on the ten edge benign prompts and computing the fraction incorrectly blocked.

---

## 5. Results

### 5.1 Main Evaluation

Table 1 reports ASR and FPR for all four defense methods on the held-out test set of 38 prompts (23 injection, 15 benign).

**Table 1: Attack Success Rate and False Positive Rate by Defense Method**

| Method           | ASR (↓) | FPR (↓) | Injections Blocked | Benign Blocked |
|------------------|---------|---------|-------------------|---------------|
| Baseline         | 100.0%  | 0.0%    | 0 / 23            | 0 / 15        |
| Regex            | 69.6%   | 0.0%    | 7 / 23            | 0 / 15        |
| Keyword          | 82.6%   | 0.0%    | 4 / 23            | 0 / 15        |
| **Context-Aware**| **47.8%**| **0.0%**| **12 / 23**       | **0 / 15**    |

The baseline confirms that, in the absence of any defense, 100% of the injection prompts in the test set succeed in causing the target model to comply with the injected instruction. The regex sanitizer blocks 7 of 23 injection prompts, reducing ASR to 69.6%. The keyword heuristic sanitizer performs below the regex sanitizer, blocking only 4 of 23 injections and achieving an ASR of 82.6%. This counterintuitive ordering—regex outperforming keywords—reflects the nature of the test-set injection distribution, which includes many direct-override attacks that trigger regex patterns but do not accumulate sufficient keyword scores. The proposed context-aware sanitizer blocks 12 of 23 injections, achieving an ASR of 47.8%, a reduction of 52.2 percentage points from baseline and 21.8 percentage points below the next-best defense (regex). Critically, all three defenses maintain FPR = 0% on the standard benign test set, meaning no legitimate user prompt was incorrectly blocked.

The improvement of the context-aware framework over single-signal defenses is attributable to its ability to detect injection attempts that evade both the regex and keyword signals. Narrative and roleplay-framing attacks, which construct attack instructions as fictional dialogue or hypothetical scenarios, frequently contain none of the explicit override phrases targeted by the regex bank and do not accumulate sufficient keyword score, but exhibit elevated semantic similarity to known injections and trigger the roleplay and intent signals. Similarly, coded-intent attacks and indirect instruction-shift attacks that use no flagged vocabulary are captured by Signals 6 and 7 of the framework.

### 5.2 Robustness Under Paraphrasing

Table 2 reports ASR before and after applying paraphrasing transformations to 50 injection prompts.

**Table 2: ASR Before and After Paraphrasing (50 Injection Prompts)**

| Method        | Original ASR | Paraphrase ASR | ΔASR     |
|---------------|-------------|----------------|----------|
| Baseline      | 1.00        | 1.00           | +0.00    |
| Regex         | 0.50        | 0.58           | **+0.08** |
| Keyword       | 0.86        | 0.88           | +0.02    |
| Context-Aware | 0.00        | 0.02           | +0.02    |

The paraphrase robustness results reveal the structural brittleness of regex-based defenses: a shift of +8 percentage points in ASR from original to paraphrased attacks indicates that straightforward synonym substitution is sufficient to evade a substantial fraction of regex-matched injections. The keyword heuristic is somewhat more robust (+2 percentage points) because its weighted vocabulary includes common injection-associated terms that tend to persist across paraphrasing, but its weak coverage of structural and semantic attack features limits its overall effectiveness. The context-aware framework demonstrates the strongest robustness, with an ASR increase of only 2 percentage points under paraphrasing. The stability of the semantic signal under paraphrasing is particularly important here: TF-IDF representations capture vocabulary-level similarity at the n-gram level, and paraphrase variants that preserve the semantic intent of an injection attack tend to preserve a meaningful subset of the injection-associated vocabulary captured by the TF-IDF model.

### 5.3 Edge Benign False Positive Rate

Table 3 reports the false positive rate on the ten edge benign prompts, which contain surface-level suspicious terminology in safe contexts.

**Table 3: False Positive Rate on Edge Benign Prompts**

| Method        | Edge FPR | Prompts Blocked |
|---------------|---------|-----------------|
| Baseline      | 0%      | 0 / 10          |
| Regex         | 20%     | 2 / 10          |
| Keyword       | 0%      | 0 / 10          |
| Context-Aware | 20%     | 2 / 10          |

Both the regex sanitizer and the context-aware framework exhibit a 20% edge FPR, incorrectly blocking 2 of the 10 edge benign prompts. The regex-blocked prompts are those whose phrasing incidentally matches patterns in the regex bank—for example, the phrase "how do I kill a Linux process" contains the word "kill" in a construction similar to instruction patterns, and "bypass this error" triggers a restricted-mode pattern. The context-aware framework's edge false positives arise from a different mechanism: these prompts accumulate borderline contributions across multiple signals (moderate semantic similarity from technical vocabulary overlap, instruction-shift patterns from conditional phrasing) that collectively exceed the soft threshold. The keyword sanitizer avoids edge false positives because the technical vocabulary in these prompts does not overlap with the injection-specific keyword list. This result identifies an important failure mode: multi-signal accumulation, while reducing ASR, can cause benign technical prompts to aggregate enough marginal evidence to cross the blocking threshold.

### 5.4 ML Classifier Comparison

Table 4 reports the performance of the logistic regression TF-IDF classifier on the validation and test splits.

**Table 4: Logistic Regression Classifier Performance**

| Split      | Accuracy | F1    | Precision | Recall |
|------------|----------|-------|-----------|--------|
| Validation | 97.3%    | 0.978 | 0.960     | 1.000  |
| Test       | 81.6%    | 0.868 | 0.765     | 1.000  |

The classifier achieves 100% recall on both validation and test splits, meaning it correctly identifies every injection prompt in the dataset. However, on the test split, this comes at the cost of a false positive rate of 46.7% (7 of 15 benign prompts misclassified as injections), reflecting a precision of 76.5%. The classifier's high recall is attributable to the sensitivity of a learned decision boundary to injection-associated vocabulary patterns, but this same sensitivity causes it to flag benign prompts that happen to share vocabulary with the training injections. The degradation from validation (FPR ≈ 6.7%) to test (FPR ≈ 46.7%) suggests that the classifier overfit to the specific vocabulary distribution of the validation injections and generalizes less well to the more varied test prompts.

This result quantifies the fundamental precision-recall tradeoff in injection detection. The context-aware sanitizer achieves a considerably better precision (100% on standard benign prompts) at the cost of lower recall (52.2% injection detection rate on the test set, compared to the classifier's 100%). For production deployments where falsely blocking legitimate user requests would degrade user experience, the sanitizer's precision-first design is the more appropriate default choice.

---

## 6. Discussion

### 6.1 Why Multi-Signal Detection Outperforms Single-Signal Approaches

The core finding of this work—that the multi-signal framework substantially reduces ASR relative to any single-signal defense—can be explained by the diversity of attack strategies employed in the evaluation dataset. No single signal covers all attack categories. The regex signal excels at catching explicit canonical injection phrases but is immediately defeated by synonym substitution. The keyword signal catches partial attack vocabulary but is blind to indirect and narrative attacks that use no flagged terms. The semantic signal detects vocabulary-level similarity to known injections and therefore handles paraphrased variants better than pattern matching, but misses novel attack phrasings that share little vocabulary with the training set. The intent signal detects the structural grammar of injection commands and is therefore robust to vocabulary variation, but misses attacks framed as data or context rather than imperative instruction. The roleplay, shift, and conflict signals capture structural properties of sophisticated multi-step attacks that evade all three of the preceding signals.

Single-signal defenses fail precisely because adversarial attack design is an optimization process: attackers iteratively adjust phrasing to evade the specific signal employed by a deployed defense. A defense that relies only on regex patterns incentivizes paraphrasing; a defense that relies only on keyword scores incentivizes indirect phrasing; a defense that relies only on semantic similarity incentivizes novel attack vocabulary. Multi-signal aggregation raises the bar because evading all seven signals simultaneously requires producing an input that shares neither vocabulary nor structural properties with any known injection class, which constrains the attack space significantly.

### 6.2 Failure Cases and Residual Vulnerabilities

Despite outperforming baseline and single-signal defenses, the proposed framework still achieves only a 52.2% injection detection rate on the test set, leaving 47.8% of attacks unblocked. Analysis of the false negatives—injection prompts that the framework failed to block—reveals several categories of residual vulnerability.

**Narrative and literary framing:** A subset of the injection prompts in the dataset embed attack instructions within extended narrative prose or literary passages. When the attack instruction is surrounded by substantial benign context, the semantic signal value may fall below the threshold because TF-IDF cosine similarity is diluted by the high volume of benign vocabulary. Similarly, the intent patterns, which target relatively short syntactic constructions, may fail to match attack instructions that are paraphrased as indirect suggestions within a larger text body.

**Multilingual attacks beyond German:** The regex and keyword signals include German-language patterns for the specific multilingual injection variants present in the training data. Injection attacks in other languages, or in code-switched multilingual text, would escape both signals. The semantic signal partially compensates by capturing vocabulary-level similarity regardless of language, but the TF-IDF model was trained primarily on English injections.

**Indirect injection in embedded content:** Injection attacks embedded within quoted text, code comments, or retrieved documents—the indirect injection attack surface emphasized in the agentic deployment literature—may not exhibit the explicit structural properties targeted by the intent and roleplay signals, since these patterns assume the attack instruction is framed as a direct address to the model. Adapting the framework to indirect injection requires consideration of the embedding context.

**Coded and euphemistic intent:** Some injection prompts in the dataset use professional, academic, or technical framing to disguise harmful requests. For example, a prompt framing a request for harmful synthesis instructions as a "chemistry research question" shares limited vocabulary with explicit injection training examples and contains no flagged keywords. Detecting such attacks requires understanding of the semantic relationship between the framing and the underlying intent, which is beyond the scope of the current TF-IDF semantic signal.

### 6.3 The ASR-FPR Tradeoff

The results collectively illustrate the fundamental tradeoff between attack detection rate and false positive rate in injection defense design. The ML classifier, representing the extreme recall-maximizing position, achieves zero missed injections on the test set but blocks nearly half of benign prompts—a false positive rate that would be operationally unacceptable in most deployment contexts. The context-aware sanitizer accepts a higher ASR in exchange for a much lower FPR, achieving zero false positives on standard benign prompts and 20% on edge benign prompts.

The appropriate position on this tradeoff curve depends on the deployment context. A high-security application in which injection attacks carry severe consequences—such as an LLM with privileged access to confidential data or external systems—may tolerate higher false positive rates in exchange for stronger attack prevention. A consumer-facing conversational assistant where user experience is paramount may require the opposite prioritization. The proposed framework's modular signal design supports this calibration: the blocking threshold and individual signal weights can be adjusted to move the operating point along the tradeoff curve without redesigning the architecture.

### 6.4 Practical Deployment Considerations

The context-aware sanitizer is designed for practical deployment as a pre-inference layer. All seven signals are computable in milliseconds on CPU hardware: the most expensive operation is TF-IDF transformation and maximum cosine similarity computation over the training corpus, which completes in under 10 milliseconds for the dataset sizes considered here. The sanitizer requires no GPU, no additional inference calls, and no model weights access, making it compatible with any black-box LLM API.

The framework's explainability is a further practical advantage. Because each blocking decision is accompanied by the full signal breakdown, a logged record of the seven signal values and the total score provides an auditable trail for each blocked prompt. This transparency supports post-hoc analysis of false positives and enables targeted threshold adjustment without opaque model retraining.

---

## 7. Limitations

This study has several limitations that qualify the scope of its conclusions.

**Dataset scale.** The evaluation dataset comprises approximately 250 prompts, with a held-out test set of 38 prompts (23 injection, 15 benign). While the dataset is diverse in attack category, this scale is insufficient to establish statistically robust performance estimates or to evaluate behavior across the full distribution of real-world injection attempts. The results should be interpreted as proof-of-concept empirical evidence rather than production-grade security guarantees.

**Single target model.** All LLM-in-the-loop experiments use Microsoft Phi-2 as the target model. Injection ASR measurements depend on the target model's instruction-following behavior and alignment properties, which differ substantially across models. A smaller model may be more susceptible to certain injection techniques; a larger or more heavily aligned model may be less susceptible, which would lower the baseline ASR and correspondingly compress the apparent gains of any defense.

**Static signal weights and thresholds.** The signal weights and blocking thresholds used in this work were selected based on analysis of the training data distribution and iterative empirical tuning on the validation set. They are not learned from data in an end-to-end fashion and may not generalize optimally to injection datasets with different attack category distributions.

**Absence of adaptive adversary evaluation.** The paraphrasing robustness evaluation uses automated synonym substitution, which approximates but does not fully simulate an adaptive adversary who is aware of the defense mechanism and specifically crafts inputs to evade all seven signals simultaneously. A white-box adversary with knowledge of the sanitizer's signal definitions could likely produce evasive attacks at substantially lower effort than estimated by the paraphrase robustness experiment.

**Edge benign false positives.** The 20% false positive rate on edge benign prompts—while based on only 10 prompts—identifies a real class of legitimate use cases (technical writing, command-line instruction, programming tutorials) that share surface-level vocabulary with injection patterns and may be incorrectly blocked. Expanding the edge benign test suite and adjusting signal calibration for these cases is left for future work.

**No evaluation of indirect injection.** The evaluation focuses exclusively on direct injection—user-provided adversarial prompts—and does not evaluate performance against indirect injection attacks embedded in retrieved documents or tool outputs. The framework's architecture is applicable to the indirect injection setting, but its signal calibration was performed on direct injection data and may require adaptation.

---

## 8. Conclusion

This paper presented a multi-signal context-aware input sanitization framework for defending large language models against prompt injection attacks. The framework combines seven complementary signals—regular expression matching, keyword scoring, TF-IDF semantic similarity, intent detection, roleplay detection, instruction shift detection, and objective conflict detection—into a weighted suspicion score augmented by hard-trigger blocking rules. Experiments on a curated dataset of approximately 250 prompts using Microsoft Phi-2 as the target model demonstrate that the proposed framework achieves an ASR of 47.8%, outperforming regex (69.6% ASR) and keyword heuristic (82.6% ASR) baselines while maintaining zero false positives on standard benign inputs. Robustness evaluation shows the framework sustains the smallest degradation under paraphrasing (+2 percentage points) compared to the regex baseline (+8 percentage points). A logistic regression classifier achieves 100% recall but at a prohibitive 46.7% false positive rate on the test set, quantifying the precision-recall tradeoff that motivates the multi-signal sanitizer design.

The framework's core argument—that single-signal defenses fail because each signal type is bypassable in isolation, while multi-signal aggregation substantially constrains the attack surface—is supported by the empirical results and by qualitative analysis of the attack categories that each signal targets. The framework is model-agnostic, lightweight enough for synchronous pre-inference deployment, and provides an auditable signal breakdown for each blocking decision.

The 47.8% ASR represents meaningful but incomplete protection: approximately half of test-set injection attempts still succeed. Future work should address the residual vulnerabilities in narrative-framed, multilingual, indirect, and coded-intent injection attacks through expanded signal design, improved semantic representations (e.g., dense vector embeddings from pre-trained language models), and dataset scale-up to support statistical validation. The multi-signal design philosophy advocated in this work—treating injection defense as an evidence aggregation problem rather than a single-criterion classification problem—provides a principled framework for incorporating additional signals as the taxonomy of injection attacks continues to evolve.

---

## References

[1] OWASP Foundation. "OWASP Top 10 for Large Language Model Applications: LLM01 — Prompt Injection." *OWASP GenAI Security Project*, 2024. https://genai.owasp.org/llmrisk/llm01-prompt-injection/

[2] Perez, F. and Ribeiro, I. "Ignore Previous Prompt: Attack Techniques For Language Models." *arXiv preprint arXiv:2211.09527*, 2022. https://doi.org/10.48550/arXiv.2211.09527

[3] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and Fritz, M. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." *arXiv preprint arXiv:2302.12173*, 2023. https://www.netspi.com/blog/executive-blog/ai-ml-pentesting/understanding-indirect-prompt-injection-attacks/

[4] Zou, A., Wang, Z., Kolter, J.Z., and Fredrikson, M. "Universal and Transferable Adversarial Attacks on Aligned Language Models." *arXiv preprint arXiv:2307.15043*, 2023.

[5] Chen, S., Chiang, M., and Mittal, P. "Struq: Defending Against Prompt Injection with Structured Queries." *USENIX Security 2025*, 2025. https://www.usenix.org/system/files/usenixsecurity25-chen-sizhe.pdf

[6] Schulhoff, S., Ilie, M., Balepur, N., Kahadze, K., Liu, A., Si, C., Li, Y., Gupta, A., Han, H., Schulhoff, S., Dulepet, P.S., Vidyadhara, S., Ki, D., Agrawal, S., Pham, C., Yim, G., Koupaee, M., Mori, I., Ghodratnama, S., Bhatt, V., Roy, U., Guo, S., Kamradt, C., Fries, S., and Boyd, D. "The Prompt Report: A Systematic Survey of Prompting Techniques." *arXiv preprint arXiv:2406.06608*, 2024.

[7] Inan, H., Upasani, K., Chi, J., Rungta, R., Iyer, K., Mao, Y., Tontchev, M., Hu, Q., Fuller, B., Testuggine, D., and Khabsa, M. "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations." *arXiv preprint arXiv:2312.06674*, 2023.

[8] Chang, J., Yi, X., Chen, H., Yuan, R., Guo, H., Ma, S., Qiao, K., and Bai, X. "Revisiting Prompt Injection Attacks Against LLM-Based Agents." *arXiv preprint arXiv:2504.05147*, 2025. https://doi.org/10.48550/arXiv.2504.05147

[9] Huang, Y., Gupta, S., Xia, M., Li, K., and Chen, D. "Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation." *arXiv preprint arXiv:2310.06987*, 2023.

[10] Wallace, E., Zhao, T.Z., Feng, S., and Singh, S. "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions." *arXiv preprint arXiv:2404.13208*, 2024. https://doi.org/10.48550/arXiv.2405.21018

[11] Shi, Z., Wang, A., Chang, K.W., and Huang, M. "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models." *arXiv preprint arXiv:2312.14197*, 2023.

[12] Zhan, Q., Fang, Z., Bindu, R., Gupta, A., Hashimoto, T., and Kang, D. "Defending Against Prompt Injection with Hierarchical Instruction Authorization." *arXiv preprint arXiv:2502.15427*, 2025. https://doi.org/10.48550/arXiv.2502.15427

[13] Promptfoo. "OWASP LLM Red Teaming Guide." *Promptfoo Blog*, 2024. https://www.promptfoo.dev/blog/owasp-red-teaming/

[14] Mend.io. "What is AI System Prompt Hardening?" *Mend.io Blog*, 2025. https://www.mend.io/blog/what-is-ai-system-prompt-hardening/

[15] GetStream. "LLM Prompt Engineering and Moderation." *GetStream Blog*, 2025. https://getstream.io/blog/llm-prompt-engineering-moderation/

[16] Berkeley AI Research. "Prompt Injection Defense: Current State and Future Directions." *BAIR Blog*, April 2025. https://bair.berkeley.edu/blog/2025/04/11/prompt-injection-defense/

[17] Liu, Y., Deng, G., Xu, Z., Li, Y., Zheng, Y., Zhang, K., Liu, Y., Wang, T., and Liu, Y. "Prompt Injection attack against LLM-integrated Applications." *arXiv preprint arXiv:2306.05499*, 2023. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11359704

[18] Nasr, M., Carlini, N., Hayase, J., Jagielski, M., Cooper, A.F., Ippolito, D., Choquette-Choo, C.A., Wallace, E., Tramèr, F., and Lee, K. "Scalable Extraction of Training Data from (Production) Language Models." *arXiv preprint arXiv:2311.17035*, 2023. https://doi.org/10.48550/arXiv.2511.10720

[19] Chen, Y., Bhatt, U., and Weller, A. "Understanding and Mitigating Multi-Step Prompt Injection in LLM Agents." *arXiv preprint arXiv:2507.15219*, 2025. https://doi.org/10.48550/arXiv.2507.15219

[20] Rao, A., Vashistha, S., Naik, A., Aditya, S., and Choudhury, M. "Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks." *arXiv preprint arXiv:2305.14965*, 2023. https://doi.org/10.48550/arXiv.2509.14285

import json
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalDecisionEngine:
    """Rank decision options against a user question using sentence embeddings."""

    PROFANITY_WORDS = {
        "damn", "hell", "shit", "crap", "stupid", "idiot", "bastard", "fool",
        "dumb", "trash", "fuck", "fucking", "ass", "bloody", "jerk",
    }

    GUARDRAIL_BANK = {
        "profanity": [
            "offensive swear words",
            "explicit vulgar language",
            "abusive profanity and insulting words",
        ],
        "hate": [
            "hate speech targeting a group",
            "discriminatory language against protected groups",
            "racist and hateful messaging",
        ],
        "violence": [
            "violent threats and attacks",
            "instructions to harm people",
            "graphic violence and physical assault",
        ],
        "dangerous": [
            "illegal weapon instructions",
            "self-harm and dangerous acts",
            "harmful dangerous behavior",
        ],
        "explicit": [
            "sexual explicit content",
            "pornographic or lewd material",
            "explicit sexual descriptions",
        ],
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", guardrail_threshold: float = 0.72):
        """Initialize the engine with a sentence-transformer model."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(guardrail_threshold, (int, float)) or guardrail_threshold <= 0:
            raise ValueError("guardrail_threshold must be a positive number.")

        self.model_name = model_name
        self.guardrail_threshold = float(guardrail_threshold)
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _job_id() -> str:
        return secrets.token_hex(16)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _check_for_blocked_word(cls, text: str) -> Optional[str]:
        normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        words = sorted(cls.PROFANITY_WORDS, key=len, reverse=True)
        for word in words:
            if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", normalized):
                return word
        return None

    def _semantic_guardrail(self, question: str, decisions: Dict[str, str]) -> None:
        combined_text = " ".join([question.strip(), *decisions.values()]).strip()
        if not combined_text:
            return

        blocked_word = self._check_for_blocked_word(combined_text)
        if blocked_word:
            raise ValueError(
                f"Guardrail error: profanity detected ('{blocked_word}'). "
                "Please replace the flagged wording and try again."
            )

        input_emb = self.model.encode([combined_text], normalize_embeddings=True)
        best_category = None
        best_score = 0.0

        for category, examples in self.GUARDRAIL_BANK.items():
            guard_emb = self.model.encode(examples, normalize_embeddings=True)
            scores = guard_emb @ input_emb[0]
            score = float(np.max(scores))
            if score > best_score:
                best_score = score
                best_category = category

        if best_category and best_score >= self.guardrail_threshold:
            raise ValueError(
                f"Guardrail error: blocked category '{best_category}' detected "
                f"(similarity={best_score:.2f}). Please replace the flagged wording and try again."
            )

    @staticmethod
    def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the payload before ranking decisions."""
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary.")

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Payload must contain a non-empty 'question' string.")

        decisions = payload.get("decisions")
        if not isinstance(decisions, dict) or not decisions:
            raise ValueError("Payload must contain a non-empty 'decisions' dictionary.")

        normalized: Dict[str, str] = {}
        for key, value in decisions.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Each decision key must be a non-empty string.")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Decision '{key}' must contain a non-empty description.")
            normalized[key] = value.strip()

        if len(normalized) != len(decisions):
            raise ValueError("Decision dict contains duplicate keys after normalization.")

        return {"question": question.strip(), "decisions": normalized}

    def evaluate(self, payload: Dict[str, Any], temperature: float = 0.07) -> Dict[str, Any]:
        """Evaluate a payload and return the selected decision with confidence rankings."""
        if not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        started_at = self._now_iso()
        started_ms = time.perf_counter()

        validated = self._validate_payload(payload)
        question = validated["question"]
        decisions = validated["decisions"]

        self._semantic_guardrail(question, decisions)

        keys = list(decisions.keys())
        values = list(decisions.values())

        query_text = f"Represent this query for retrieval: {question}"
        embeddings = self.model.encode([query_text] + values, normalize_embeddings=True)

        q_vec = embeddings[0]
        choice_vecs = embeddings[1:]

        raw_similarities = np.dot(choice_vecs, q_vec)
        scaled_logits = raw_similarities / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        scores = exp_logits / np.sum(exp_logits)

        ranked_choices: List[Dict[str, Any]] = []
        for key, val, score in sorted(zip(keys, values, scores), key=lambda item: item[2], reverse=True):
            ranked_choices.append({
                "decision": key,
                "description": val,
                "confidence": float(score),
            })

        best = ranked_choices[0]
        completed_ms = int((time.perf_counter() - started_ms) * 1000)
        result = {
            "job_id": self._job_id(),
            "status": "completed",
            "guardrail_status": "passed",
            "input": {
                "question": question,
                "decisions": decisions,
            },
            "selected": {
                "decision": best["decision"],
                "confidence": best["confidence"],
            },
            "rankings": ranked_choices,
            "processing_time_ms": completed_ms,
            "estimated_cost_usd": round(completed_ms / 1000000, 6),
            "started_at": started_at,
            "completed_at": self._now_iso(),
        }
        return result

    def evaluate_batch(self, payloads: Sequence[Dict[str, Any]], temperature: float = 0.07) -> List[Dict[str, Any]]:
        """Evaluate multiple payloads in sequence."""
        if not isinstance(payloads, Iterable):
            raise ValueError("payloads must be an iterable of payload dictionaries.")

        results = []
        for payload in payloads:
            results.append(self.evaluate(payload, temperature=temperature))
        return results


if __name__ == "__main__":
    engine = LocalDecisionEngine()
    sample_json = {
        "question": "The database connection pool is completely saturated.",
        "decisions": {
            "scale_up": "Provision a larger database instance tier immediately.",
            "drop_traffic": "Return 503 Service Unavailable for non-critical background sync jobs.",
            "do_nothing": "Wait and see if traffic spikes naturally subside."
        }
    }
    print(json.dumps(engine.evaluate(sample_json), indent=2))

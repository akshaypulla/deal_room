"""
Deterministic semantic analyzer for DealRoom V2.5.

The preferred path uses a lightweight sentence-transformer when available.
For container-friendly deployment, the default non-lexical path uses a fitted
TF-IDF vector space so the shipped environment still gets semantic-ish vector
matching without pulling a massive PyTorch stack. If neither path is available,
the analyzer falls back to lexical similarity.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

INTENT_BANK: Dict[str, List[str]] = {
    "discover_budget": [
        "help me understand the budget ceiling",
        "what financial boundary do we need to respect",
        "what spend limit is realistic internally",
    ],
    "discover_timeline": [
        "what timeline can your team actually support",
        "is there a firm delivery window we need to honor",
        "what implementation deadline matters most",
    ],
    "discover_compliance": [
        "what compliance requirement is driving review",
        "which regulatory obligation is non negotiable",
        "what security or privacy bar must be met",
    ],
    "reassure": [
        "we will work through this together",
        "i want to address your concern directly",
        "we can adapt the proposal to your needs",
    ],
    "pressure": [
        "this is the final offer",
        "we need a decision now",
        "take it or leave it",
    ],
    "close_attempt": [
        "let us move to final approval",
        "i propose we proceed and sign",
        "we should close this now",
    ],
    "share_roi": [
        "here is the roi analysis",
        "this model shows the payback period",
        "i am sharing the financial business case",
    ],
    "share_implementation": [
        "here is the implementation timeline",
        "this rollout plan covers milestones and staffing",
        "i am sharing the delivery plan",
    ],
    "share_security": [
        "here are our security certifications",
        "this covers our compliance posture",
        "i am sharing security and audit material",
    ],
    "share_dpa": [
        "here is the data processing agreement",
        "this contract addendum covers privacy obligations",
        "i am sharing the dpa and privacy terms",
    ],
    "share_vendor_packet": [
        "here is the vendor onboarding packet",
        "this covers insurance process and supplier details",
        "i am sharing procurement onboarding documentation",
    ],
}

TONE_BANK: Dict[str, List[str]] = {
    "collaborative": [
        "we can work through this together",
        "i appreciate the concern and want a mutual solution",
        "we are committed to a partnership",
    ],
    "credible": [
        "here is the specific evidence requested",
        "this is based on delivered projects and exact milestones",
        "i want to be precise about what we can commit to",
    ],
    "specific": [
        "the timeline is 14 weeks with named milestones",
        "the cost is capped at 180000 with quarterly billing",
        "the agreement includes named controls and review dates",
    ],
    "pushy": [
        "this must be decided now",
        "we cannot keep discussing this",
        "you need to commit immediately",
    ],
    "evasive": [
        "we can work out the details later",
        "i would not focus on that right now",
        "there is nothing to worry about",
    ],
    "adaptive": [
        "we can tailor this to your internal approval path",
        "we can adjust the rollout to match your deadline",
        "we can restructure terms around the concern you raised",
    ],
}

ARTIFACT_ALIASES = {
    "roi_model": ["roi", "business case", "payback", "financial model"],
    "implementation_timeline": ["implementation", "timeline", "rollout plan", "milestone"],
    "security_cert": ["security", "soc 2", "audit", "certification"],
    "dpa": ["dpa", "data processing agreement", "privacy addendum", "gdpr"],
    "vendor_packet": ["vendor packet", "supplier onboarding", "insurance", "onboarding packet"],
    "reference_case": ["reference", "customer story", "case study", "similar deployment"],
    "support_plan": ["support plan", "named support lead", "support coverage"],
}

SLOT_ALIASES = {
    "price": ["price", "cost", "budget", "commercial", "spend"],
    "timeline_weeks": ["week", "timeline", "delivery", "rollout", "implementation"],
    "security_posture": ["security", "soc 2", "audit", "gdpr", "privacy"],
    "liability": ["liability", "indemnity", "cap", "clause", "exposure"],
    "support_level": ["support", "coverage", "named support lead", "service"],
    "implementation_commitment": ["implementation", "resources", "engineers", "staffing"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_score(text: str, exemplar: str) -> float:
    a = set(_tokenize(text))
    b = set(_tokenize(exemplar))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticAnalyzer:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self._backend = "lexical"
        self._model = None
        self._vectorizer = None
        self._artifact_patterns = {
            artifact: re.compile("|".join(re.escape(term) for term in aliases), re.I)
            for artifact, aliases in ARTIFACT_ALIASES.items()
        }
        try:  # pragma: no branch
            self._model = self._load_model(model_name)
            if self._model is not None:
                self._backend = "embedding"
        except Exception:
            self._model = None
        if self._backend == "lexical" and TfidfVectorizer is not None:
            self._vectorizer = self._build_vectorizer()
            if self._vectorizer is not None:
                self._backend = "tfidf"
        self._intent_vectors = {
            name: self._encode_many(examples) for name, examples in INTENT_BANK.items()
        }
        self._tone_vectors = {
            name: self._encode_many(examples) for name, examples in TONE_BANK.items()
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model(model_name: str):
        if SentenceTransformer is not None:
            return SentenceTransformer(model_name)
        return None

    def _build_vectorizer(self):
        if TfidfVectorizer is None:
            return None
        corpus: List[str] = []
        for values in INTENT_BANK.values():
            corpus.extend(values)
        for values in TONE_BANK.values():
            corpus.extend(values)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        vectorizer.fit(corpus)
        return vectorizer

    def _encode_many(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if self._model is None:
            if self._vectorizer is not None:
                matrix = self._vectorizer.transform(texts)
                return matrix.toarray().astype(float)
            return np.array([[0.0] for _ in texts], dtype=float)
        vectors = self._model.encode(texts)
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        normalized = []
        for vector in vectors:
            norm = np.linalg.norm(vector)
            normalized.append(vector if norm == 0 else vector / norm)
        vectors = np.array(normalized)
        return vectors

    def _similarity(self, text: str, exemplars: List[str], vectors: np.ndarray) -> float:
        if self._backend == "embedding" and self._model is not None:
            encoded = self._model.encode([text])
            if isinstance(encoded, list):
                encoded = np.array(encoded)
            query = encoded[0]
            if np.linalg.norm(query) != 0:
                query = query / np.linalg.norm(query)
            scores = vectors @ query
            return float(np.max(scores))
        if self._backend == "tfidf" and self._vectorizer is not None and vectors.size:
            query = self._vectorizer.transform([text]).toarray()[0]
            query_norm = np.linalg.norm(query)
            if query_norm != 0:
                query = query / query_norm
            normalized = []
            for row in vectors:
                row_norm = np.linalg.norm(row)
                normalized.append(row if row_norm == 0 else row / row_norm)
            scores = np.array(normalized) @ query
            return float(np.max(scores))
        return max((_lexical_score(text, exemplar) for exemplar in exemplars), default=0.0)

    def analyze(
        self,
        message: str,
        context: Dict[str, object],
        stakeholder_roles: Dict[str, str],
    ) -> Dict[str, object]:
        lowered = message.lower()
        intent_matches = {
            intent: round(self._similarity(message, INTENT_BANK[intent], vectors), 4)
            for intent, vectors in self._intent_vectors.items()
        }
        tone_scores = {
            tone: round(self._similarity(message, TONE_BANK[tone], vectors), 4)
            for tone, vectors in self._tone_vectors.items()
        }

        artifact_matches = []
        for artifact, pattern in self._artifact_patterns.items():
            if pattern.search(message):
                artifact_matches.append(artifact)
        for doc in context.get("documents", []):
            doc_type = doc.get("type")
            if doc_type and doc_type not in artifact_matches:
                artifact_matches.append(doc_type)

        claim_candidates = self._extract_claims(lowered, artifact_matches)
        request_matches = self._match_requests(
            message,
            artifact_matches,
            context.get("requested_artifacts", {}),
        )
        return {
            "intent_matches": intent_matches,
            "tone_scores": tone_scores,
            "artifact_matches": artifact_matches,
            "claim_candidates": claim_candidates,
            "request_matches": request_matches,
            "backend": self._backend,
            "stakeholder_roles": stakeholder_roles,
        }

    def _extract_claims(self, lowered: str, artifact_matches: List[str]) -> List[Dict[str, object]]:
        claims = []

        price_match = re.search(r"\$?\s*(\d{2,3}(?:,\d{3})+|\d{5,6})", lowered)
        if price_match:
            raw = price_match.group(1).replace(",", "")
            claims.append(
                {
                    "slot": "price",
                    "value": float(raw),
                    "text": price_match.group(0),
                    "polarity": "cap" if any(term in lowered for term in ["cap", "max", "ceiling"]) else "offer",
                }
            )

        timeline_match = re.search(r"(\d{1,2})\s*(?:weeks?|wks?)", lowered)
        if timeline_match:
            claims.append(
                {
                    "slot": "timeline_weeks",
                    "value": float(timeline_match.group(1)),
                    "text": timeline_match.group(0),
                    "polarity": "timeline",
                }
            )

        slot_texts = {
            "security_posture": ["gdpr", "soc 2", "audit rights", "data residency"],
            "liability": ["liability cap", "unlimited liability", "indemnity"],
            "support_level": ["named support lead", "24/7 support", "premium support"],
            "implementation_commitment": ["dedicated engineers", "implementation team", "named rollout lead"],
        }
        for slot, terms in slot_texts.items():
            for term in terms:
                if term in lowered:
                    claims.append(
                        {
                            "slot": slot,
                            "value": term,
                            "text": term,
                            "polarity": "positive" if "unlimited" not in term else "negative",
                        }
                    )
                    break

        if "dpa" in artifact_matches and not any(claim["slot"] == "security_posture" for claim in claims):
            claims.append(
                {
                    "slot": "security_posture",
                    "value": "gdpr",
                    "text": "dpa",
                    "polarity": "positive",
                }
            )
        if "implementation_timeline" in artifact_matches and not any(
            claim["slot"] == "implementation_commitment" for claim in claims
        ):
            claims.append(
                {
                    "slot": "implementation_commitment",
                    "value": "named rollout lead",
                    "text": "implementation_timeline",
                    "polarity": "positive",
                }
            )
        return claims

    def _match_requests(
        self,
        message: str,
        artifact_matches: List[str],
        requested_artifacts: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        matched = []
        lowered = message.lower()
        for stakeholder_id, requested in requested_artifacts.items():
            for artifact in requested:
                if artifact in artifact_matches:
                    matched.append({"stakeholder_id": stakeholder_id, "artifact": artifact})
                    continue
                aliases = ARTIFACT_ALIASES.get(artifact, [])
                if any(alias in lowered for alias in aliases):
                    matched.append({"stakeholder_id": stakeholder_id, "artifact": artifact})
        seen = set()
        deduped = []
        for item in matched:
            key = (item["stakeholder_id"], item["artifact"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped


DEFAULT_ANALYZER = SemanticAnalyzer()

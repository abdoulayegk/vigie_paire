"""Mémoire auditable de précédents validés par les analystes.

Le module est volontairement indépendant du moteur de triage. Il transforme
uniquement des décisions finales et structurées en précédents, puis recherche
des cas comparables avec un classement déterministe. Un moteur d'embeddings
peut être fourni par l'appelant; en son absence ou en cas d'échec, la recherche
revient automatiquement à une similarité lexicale locale.

Les commentaires libres ne sont jamais interprétés comme une correction. Une
décision rejetée n'est retenue que si elle contient des champs ``corrected_*``
ou ``final_*`` explicites.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    get_args,
)

from vigilance.amf_taxonomy import (
    ChangeNature,
    THEMES_AMF_PIPELINE_2,
)

PRECEDENT_SCHEMA_VERSION = "analyst_precedent_v1"
PRECEDENT_PACKET_SCHEMA_VERSION = "analyst_precedent_packet_v1"

_APPROVED_STATUSES = {
    "approved",
    "confirmed",
    "validated",
    "validated_by_analyst",
    "analyst_validated",
    "valide",
}
_CORRECTED_STATUSES = {"corrected", "corrige", "rejected"}
_MATERIALITY_DECISION_SCOPES = {
    "MATERIALITY",
    "MATERIALITE",
    "MATERIALITY_CLASSIFICATION",
}
_MATERIALITY_REVIEW_SCHEMAS = {
    "ANALYST_MATERIALITY_REVIEW_V1",
}
_TRUSTED_PRECEDENT_SCHEMAS = {
    "ANALYST_PRECEDENT_V1",
    "ANALYST_PRECEDENT_V2",
}
_ALLOWED_CHANGE_NATURES = set(get_args(ChangeNature))
_ALLOWED_THEMES = set(THEMES_AMF_PIPELINE_2)
_NON_FINAL_DECISION_STATUSES = {
    "A_CONFIRMER",
    "A_REVOIR",
    "INDETERMINE",
    "PENDING",
    "PROVISOIRE",
    "REVIEW_REQUIRED",
}
_MATERIALITY_ALIASES = {
    "MAJEUR": "MAJEUR",
    "MAJOR": "MAJEUR",
    "MODERE": "MODERE",
    "MODERATE": "MODERE",
    "MINEUR": "MINEUR",
    "MINOR": "MINEUR",
}
_MATERIALITY_KEYS = (
    "corrected_materiality_level",
    "final_materiality_level",
    "materiality_level",
    "corrected_impact_level",
    "final_impact_level",
    "impact_level",
)
_CHANGE_NATURE_KEYS = (
    "corrected_change_nature",
    "final_change_nature",
    "change_nature",
    "change_type",
    "diff_type",
)
_RATIONALE_KEYS = (
    "corrected_materiality_rationale",
    "final_materiality_rationale",
    "materiality_rationale",
    "corrected_decision_rationale",
    "final_decision_rationale",
    "decision_rationale",
    "rationale_metier",
    "signification_metier",
    "relevance_reason",
)
_CORRECTION_CONTAINERS = (
    "structured_correction",
    "correction",
    "corrected_decision",
    "final_decision",
    "analyst_decision",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "et",
    "est",
    "il",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "ou",
    "par",
    "pour",
    "que",
    "qui",
    "se",
    "son",
    "sur",
    "un",
    "une",
}


class EmbeddingEngine(Protocol):
    """Interface minimale d'un moteur d'embeddings optionnel."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Retourne un vecteur par texte, dans le même ordre."""


@dataclass(frozen=True, slots=True)
class AnalystPrecedent:
    """Décision structurée dont la validation analyste est démontrée."""

    precedent_id: str
    change_id: str
    bank_code: str
    section_key: str
    text_before: str
    text_after: str
    materiality_level: str
    change_nature: str
    business_equivalence: str = "INDETERMINE"
    materiality_confidence: str | float | None = None
    evidence_sufficiency: str = ""
    decision_status: str = "VALIDATED_ANALYSTE"
    review_required: bool = False
    is_relevant: bool | None = None
    themes_amf: tuple[str, ...] = ()
    rationale: str = ""
    supporting_evidence: tuple[str, ...] = ()
    counterarguments: tuple[str, ...] = ()
    reviewer: str = ""
    validated_at: str = ""
    decision_origin: str = "analyst_approved"
    source_kind: str = ""
    source_reference: str = ""

    def retrieval_text(self) -> str:
        """Construit le texte stable remis au moteur de recherche."""
        themes = " ".join(self.themes_amf)
        return (
            f"banque {self.bank_code}\n"
            f"section {self.section_key}\n"
            f"nature {self.change_nature}\n"
            f"themes {themes}\n"
            f"avant {self.text_before}\n"
            f"apres {self.text_after}"
        )

    def to_audit_dict(
        self,
        *,
        text_limit: int = 1_000,
        rationale_limit: int = 600,
        evidence_limit: int = 400,
    ) -> dict[str, Any]:
        """Retourne une représentation compacte destinée à un prompt ou journal."""
        return {
            "precedent_id": self.precedent_id,
            "change_id": self.change_id,
            "bank_code": self.bank_code,
            "section_key": self.section_key,
            "text_before": _clip(self.text_before, text_limit),
            "text_after": _clip(self.text_after, text_limit),
            "analyst_decision": {
                "materiality_level": self.materiality_level,
                "change_nature": self.change_nature,
                "business_equivalence": self.business_equivalence,
                "materiality_confidence": self.materiality_confidence,
                "evidence_sufficiency": self.evidence_sufficiency,
                "decision_status": self.decision_status,
                "review_required": self.review_required,
                "is_relevant": self.is_relevant,
                "themes_amf": list(self.themes_amf),
                "rationale": _clip(self.rationale, rationale_limit),
                "supporting_evidence": [
                    _clip(value, evidence_limit) for value in self.supporting_evidence
                ],
                "counterarguments": [
                    _clip(value, evidence_limit) for value in self.counterarguments
                ],
            },
            "validation": {
                "reviewer": self.reviewer,
                "validated_at": self.validated_at,
                "decision_origin": self.decision_origin,
            },
            "source": {
                "kind": self.source_kind,
                "reference": self.source_reference,
            },
        }


@dataclass(frozen=True, slots=True)
class PrecedentQuery:
    """Description structurée du changement pour lequel chercher des analogues."""

    text_before: str
    text_after: str
    bank_code: str = ""
    section_key: str = ""
    change_nature: str = ""
    themes_amf: tuple[str, ...] = ()
    candidate_materiality_level: str = ""
    business_equivalence: str = "INDETERMINE"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrecedentQuery":
        """Construit une requête depuis un changement du pipeline."""
        payload = _as_mapping(value.get("payload"))
        triage = _as_mapping(value.get("genai_triage"))
        return cls(
            text_before=_first_text(
                value,
                payload,
                keys=(
                    "text_before",
                    "semantic_text_t1",
                    "source_text_t1",
                    "old_text",
                    "from",
                ),
                evidence_key="evidence_t1",
            ),
            text_after=_first_text(
                value,
                payload,
                keys=(
                    "text_after",
                    "semantic_text_t2",
                    "source_text_t2",
                    "new_text",
                    "to",
                ),
                evidence_key="evidence_t2",
            ),
            bank_code=_clean_text(value.get("bank_code") or value.get("bank")).lower(),
            section_key=_clean_text(value.get("section_key") or value.get("section")),
            change_nature=_normalize_change_nature(
                _pick((value, payload, triage), _CHANGE_NATURE_KEYS)
            ),
            themes_amf=tuple(
                _string_list(
                    _pick(
                        (value, payload, triage),
                        ("themes_amf", "candidate_themes", "themes"),
                    )
                )
            ),
            candidate_materiality_level=_normalize_materiality(
                _pick((value, payload, triage), _MATERIALITY_KEYS)
            ),
            business_equivalence=_normalize_equivalence(
                _pick(
                    (value, payload, triage),
                    ("business_equivalence", "equivalence_metier"),
                )
            ),
        )

    def retrieval_text(self) -> str:
        """Construit le texte stable utilisé pour les embeddings."""
        return (
            f"banque {self.bank_code}\n"
            f"section {self.section_key}\n"
            f"nature {self.change_nature}\n"
            f"themes {' '.join(self.themes_amf)}\n"
            f"avant {self.text_before}\n"
            f"apres {self.text_after}"
        )

    def fingerprint(self) -> str:
        """Retourne l'empreinte stable de la requête sans exposer son contenu."""
        payload = {
            "bank_code": self.bank_code,
            "section_key": self.section_key,
            "change_nature": self.change_nature,
            "themes_amf": list(self.themes_amf),
            "candidate_materiality_level": self.candidate_materiality_level,
            "business_equivalence": self.business_equivalence,
            "text_before": self.text_before,
            "text_after": self.text_after,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievedPrecedent:
    """Précédent retenu, accompagné de son rôle et de ses scores."""

    precedent: AnalystPrecedent
    role: str
    score: float
    score_breakdown: Mapping[str, float]

    def to_audit_dict(self) -> dict[str, Any]:
        """Retourne l'entrée compacte incluse dans le paquet de précédents."""
        result = self.precedent.to_audit_dict()
        result["retrieval"] = {
            "role": self.role,
            "score": round(self.score, 6),
            "score_breakdown": {
                key: round(float(value), 6)
                for key, value in sorted(self.score_breakdown.items())
            },
        }
        return result


@dataclass(frozen=True, slots=True)
class PrecedentPacket:
    """Petit paquet auditable de cas analogues et contrastifs."""

    query_fingerprint: str
    retrieval_method: str
    positive_precedents: tuple[RetrievedPrecedent, ...] = ()
    contrastive_precedents: tuple[RetrievedPrecedent, ...] = ()
    anchor_materiality_level: str = ""
    fallback_reason: str = ""
    corpus_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Sérialise le paquet sans état interne ni vecteurs."""
        return {
            "schema_version": PRECEDENT_PACKET_SCHEMA_VERSION,
            "query_fingerprint": self.query_fingerprint,
            "retrieval_method": self.retrieval_method,
            "fallback_reason": self.fallback_reason,
            "selection_policy": {
                "anchor_materiality_level": self.anchor_materiality_level,
                "positive_definition": "Même niveau que l'ancre parmi les cas les plus similaires.",
                "contrastive_definition": (
                    "Cas proche avec un autre niveau ou une conclusion d'équivalence différente."
                ),
            },
            "corpus_size": self.corpus_size,
            "positive_precedents": [
                result.to_audit_dict() for result in self.positive_precedents
            ],
            "contrastive_precedents": [
                result.to_audit_dict() for result in self.contrastive_precedents
            ],
        }

    def to_prompt_json(self) -> str:
        """Retourne le paquet JSON prêt à être injecté dans un prompt."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class PrecedentLoadReport:
    """Diagnostic non bloquant produit pendant le chargement."""

    files_seen: int = 0
    records_seen: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    duplicate_records: int = 0
    errors: tuple[str, ...] = ()


@dataclass(slots=True)
class _MutableLoadReport:
    files_seen: int = 0
    records_seen: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    duplicate_records: int = 0
    errors: list[str] = field(default_factory=list)

    def freeze(self) -> PrecedentLoadReport:
        return PrecedentLoadReport(
            files_seen=self.files_seen,
            records_seen=self.records_seen,
            accepted_records=self.accepted_records,
            rejected_records=self.rejected_records,
            duplicate_records=self.duplicate_records,
            errors=tuple(self.errors),
        )


@dataclass(slots=True)
class PrecedentMemory:
    """Collection en mémoire et moteur de recherche de précédents."""

    precedents: tuple[AnalystPrecedent, ...] = ()
    embedding_engine: EmbeddingEngine | Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None
    load_report: PrecedentLoadReport = field(default_factory=PrecedentLoadReport)

    def __post_init__(self) -> None:
        """Stabilise l'ordre, déduplique et neutralise les conflits."""
        deduplicated: dict[str, AnalystPrecedent] = {}
        quarantined_ids: set[str] = set()
        for precedent in self.precedents:
            if precedent.precedent_id in quarantined_ids:
                continue
            current = deduplicated.get(precedent.precedent_id)
            if current is None:
                deduplicated[precedent.precedent_id] = precedent
                continue
            current_priority = _origin_priority(current)
            precedent_priority = _origin_priority(precedent)
            if (
                current_priority == precedent_priority
                and not _same_precedent_decision(current, precedent)
            ):
                deduplicated.pop(precedent.precedent_id, None)
                quarantined_ids.add(precedent.precedent_id)
                continue
            if precedent_priority > current_priority:
                deduplicated[precedent.precedent_id] = precedent
        self.precedents = tuple(
            sorted(deduplicated.values(), key=lambda value: value.precedent_id)
        )

    @classmethod
    def from_paths(
        cls,
        paths: str | Path | Iterable[str | Path],
        *,
        embedding_engine: EmbeddingEngine
        | Callable[[Sequence[str]], Sequence[Sequence[float]]]
        | None = None,
        strict: bool = False,
    ) -> "PrecedentMemory":
        """Charge les décisions validées présentes dans des JSON ou dossiers."""
        precedents, report = _load_paths(paths, strict=strict)
        return cls(
            precedents=tuple(precedents),
            embedding_engine=embedding_engine,
            load_report=report,
        )

    def build_packet(
        self,
        query: PrecedentQuery | Mapping[str, Any],
        *,
        positive_limit: int = 2,
        contrastive_limit: int = 2,
        minimum_score: float = 0.10,
    ) -> PrecedentPacket:
        """Recherche un ensemble court d'analogues et de cas contrastifs."""
        if not isinstance(query, PrecedentQuery):
            query = PrecedentQuery.from_mapping(query)
        positive_limit = max(0, int(positive_limit))
        contrastive_limit = max(0, int(contrastive_limit))

        if not self.precedents or positive_limit + contrastive_limit == 0:
            return PrecedentPacket(
                query_fingerprint=query.fingerprint(),
                retrieval_method="lexical",
                corpus_size=len(self.precedents),
            )

        lexical = [
            _lexical_score(query, precedent) for precedent in self.precedents
        ]
        method = "lexical"
        fallback_reason = ""
        scores = [value[0] for value in lexical]
        breakdowns = [dict(value[1]) for value in lexical]

        if self.embedding_engine is not None:
            try:
                embeddings = _embed(
                    self.embedding_engine,
                    [query.retrieval_text()]
                    + [precedent.retrieval_text() for precedent in self.precedents],
                )
                query_vector = embeddings[0]
                embedding_scores = [
                    max(0.0, _cosine_vectors(query_vector, vector))
                    for vector in embeddings[1:]
                ]
                scores = [
                    (0.72 * embedding_score) + (0.28 * lexical_score)
                    for embedding_score, lexical_score in zip(embedding_scores, scores)
                ]
                for breakdown, embedding_score in zip(
                    breakdowns, embedding_scores
                ):
                    breakdown["embedding"] = embedding_score
                    breakdown["lexical_composite"] = breakdown.pop("total")
                method = "embedding_hybrid"
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {_clip(str(exc), 160)}"

        ranked = sorted(
            (
                (score, precedent, breakdown)
                for score, precedent, breakdown in zip(
                    scores, self.precedents, breakdowns
                )
                if score >= minimum_score
                and (
                    max(
                        breakdown.get("combined_text", 0.0),
                        breakdown.get("text_before", 0.0),
                        breakdown.get("text_after", 0.0),
                        breakdown.get("change_delta", 0.0),
                    )
                    > 0.0
                    or (
                        method == "embedding_hybrid"
                        and breakdown.get("embedding", 0.0) >= 0.25
                    )
                )
            ),
            key=lambda value: (-value[0], value[1].precedent_id),
        )
        if not ranked:
            return PrecedentPacket(
                query_fingerprint=query.fingerprint(),
                retrieval_method=method,
                fallback_reason=fallback_reason,
                corpus_size=len(self.precedents),
            )

        anchor = (
            _normalize_materiality(query.candidate_materiality_level)
            or ranked[0][1].materiality_level
        )
        positive_candidates = [
            value for value in ranked if value[1].materiality_level == anchor
        ]
        positives = positive_candidates[:positive_limit]
        positive_ids = {value[1].precedent_id for value in positives}
        anchor_equivalences = {
            value[1].business_equivalence
            for value in positives
            if value[1].business_equivalence != "INDETERMINE"
        }

        contrasts = [
            value
            for value in ranked
            if value[1].precedent_id not in positive_ids
            and (
                value[1].materiality_level != anchor
                or (
                    anchor_equivalences
                    and value[1].business_equivalence not in anchor_equivalences
                    and value[1].business_equivalence != "INDETERMINE"
                )
            )
        ][:contrastive_limit]

        return PrecedentPacket(
            query_fingerprint=query.fingerprint(),
            retrieval_method=method,
            fallback_reason=fallback_reason,
            anchor_materiality_level=anchor,
            corpus_size=len(self.precedents),
            positive_precedents=tuple(
                RetrievedPrecedent(
                    precedent=value[1],
                    role="positive",
                    score=value[0],
                    score_breakdown=value[2],
                )
                for value in positives
            ),
            contrastive_precedents=tuple(
                RetrievedPrecedent(
                    precedent=value[1],
                    role="contrastive",
                    score=value[0],
                    score_breakdown=value[2],
                )
                for value in contrasts
            ),
        )


def load_validated_precedents(
    paths: str | Path | Iterable[str | Path],
    *,
    strict: bool = False,
) -> list[AnalystPrecedent]:
    """Charge uniquement les décisions finales suffisamment structurées."""
    precedents, _report = _load_paths(paths, strict=strict)
    return precedents


def build_precedent_packet(
    precedents: Iterable[AnalystPrecedent],
    query: PrecedentQuery | Mapping[str, Any],
    *,
    embedding_engine: EmbeddingEngine
    | Callable[[Sequence[str]], Sequence[Sequence[float]]]
    | None = None,
    positive_limit: int = 2,
    contrastive_limit: int = 2,
    minimum_score: float = 0.10,
) -> dict[str, Any]:
    """Fonction pratique retournant directement un paquet sérialisable."""
    memory = PrecedentMemory(
        precedents=tuple(precedents),
        embedding_engine=embedding_engine,
    )
    return memory.build_packet(
        query,
        positive_limit=positive_limit,
        contrastive_limit=contrastive_limit,
        minimum_score=minimum_score,
    ).to_dict()


def _load_paths(
    paths: str | Path | Iterable[str | Path],
    *,
    strict: bool,
) -> tuple[list[AnalystPrecedent], PrecedentLoadReport]:
    report = _MutableLoadReport()
    loaded: dict[str, AnalystPrecedent] = {}
    quarantined_ids: set[str] = set()
    for path in _iter_json_paths(paths):
        report.files_seen += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if strict:
                raise
            report.errors.append(f"{path}: {type(exc).__name__}")
            continue

        for candidate in _iter_payload_candidates(payload, path):
            report.records_seen += 1
            precedent = _candidate_to_precedent(candidate)
            if precedent is None:
                report.rejected_records += 1
                continue
            if precedent.precedent_id in quarantined_ids:
                report.duplicate_records += 1
                report.rejected_records += 1
                continue
            existing = loaded.get(precedent.precedent_id)
            if existing is not None:
                report.duplicate_records += 1
                same_priority = (
                    _origin_priority(precedent)
                    == _origin_priority(existing)
                )
                if (
                    same_priority
                    and not _same_precedent_decision(
                        existing,
                        precedent,
                    )
                ):
                    loaded.pop(precedent.precedent_id, None)
                    quarantined_ids.add(precedent.precedent_id)
                    report.accepted_records = max(
                        0,
                        report.accepted_records - 1,
                    )
                    report.rejected_records += 2
                    report.errors.append(
                        "Conflit de décisions analystes mis en quarantaine: "
                        f"{precedent.precedent_id}"
                    )
                    continue
                if _origin_priority(precedent) <= _origin_priority(existing):
                    continue
            else:
                report.accepted_records += 1
            loaded[precedent.precedent_id] = precedent

    return (
        sorted(loaded.values(), key=lambda value: value.precedent_id),
        report.freeze(),
    )


def _iter_json_paths(
    paths: str | Path | Iterable[str | Path],
) -> Iterable[Path]:
    if isinstance(paths, (str, Path)):
        roots = [Path(paths)]
    else:
        roots = [Path(value) for value in paths]

    discovered: set[Path] = set()
    for root in roots:
        if root.is_file():
            discovered.add(root)
            continue
        if not root.is_dir():
            continue
        patterns = (
            "text_comparison.json",
            "*.review_state.json",
            "*.review_state.*.json",
            "analyst_precedents.json",
            "analyst_precedents.*.json",
        )
        for pattern in patterns:
            discovered.update(path for path in root.rglob(pattern) if path.is_file())
    yield from sorted(discovered, key=lambda value: str(value))


@dataclass(frozen=True, slots=True)
class _Candidate:
    record: Mapping[str, Any]
    review: Mapping[str, Any]
    context: Mapping[str, Any]
    source_kind: str
    source_reference: str


def _iter_payload_candidates(payload: Any, path: Path) -> Iterable[_Candidate]:
    if isinstance(payload, list):
        for value in payload:
            if isinstance(value, Mapping):
                yield _Candidate(value, _review_mapping(value), {}, "precedent_list", str(path))
        return
    if not isinstance(payload, Mapping):
        return

    sections = payload.get("section_comparisons")
    if isinstance(sections, list):
        seen_changes: set[str] = set()
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            section_identity = _clean_text(
                section.get("section_key")
                or section.get("section")
                or section.get("section_title")
            )
            for bucket in ("all_block_comparisons", "block_comparisons"):
                changes = section.get(bucket)
                if not isinstance(changes, list):
                    continue
                for change in changes:
                    if not isinstance(change, Mapping):
                        continue
                    identity = f"{section_identity}::{_candidate_identity(change)}"
                    if identity in seen_changes:
                        continue
                    seen_changes.add(identity)
                    context = {
                        **payload,
                        **section,
                        "_source_bucket": bucket,
                    }
                    yield _Candidate(
                        change,
                        _review_mapping(change),
                        context,
                        "text_comparison",
                        str(path),
                    )
        return

    queue = payload.get("review_queue")
    if isinstance(queue, list):
        for table in queue:
            if not isinstance(table, Mapping):
                continue
            changes = table.get("changes")
            if not isinstance(changes, list):
                continue
            for change in changes:
                if isinstance(change, Mapping):
                    context = {**payload, **table}
                    yield _Candidate(
                        change,
                        _review_mapping(change),
                        context,
                        "review_state",
                        str(path),
                    )
        return

    precedents = payload.get("precedents")
    if isinstance(precedents, list):
        for record in precedents:
            if isinstance(record, Mapping):
                yield _Candidate(
                    record,
                    _review_mapping(record),
                    payload,
                    "precedent_registry",
                    str(path),
                )
        return

    if any(key in payload for key in _MATERIALITY_KEYS):
        yield _Candidate(
            payload,
            _review_mapping(payload),
            {},
            "precedent_record",
            str(path),
        )


def _candidate_to_precedent(candidate: _Candidate) -> AnalystPrecedent | None:
    record = candidate.record
    review = candidate.review
    context = candidate.context
    status = _normalize_status(
        review.get("status")
        or review.get("review_status")
        or record.get("validation_status")
        or record.get("review_status")
    )
    decision_scope = _normalize_identifier(
        review.get("decision_scope")
        or record.get("decision_scope")
    )
    review_schema = _normalize_identifier(
        review.get("schema_version")
        or record.get("schema_version")
    )
    trusted_registry = (
        candidate.source_kind == "precedent_registry"
        and review_schema in _TRUSTED_PRECEDENT_SCHEMAS
    )
    declared_materiality_review = (
        decision_scope in _MATERIALITY_DECISION_SCOPES
        and review_schema in _MATERIALITY_REVIEW_SCHEMAS
    )
    if not trusted_registry and not declared_materiality_review:
        return None
    explicit_correction_sources = _explicit_correction_sources(record, review)
    is_approved = status in _APPROVED_STATUSES
    is_corrected = status in _CORRECTED_STATUSES and bool(
        explicit_correction_sources
    )
    if not is_approved and not is_corrected:
        return None

    payload = _as_mapping(record.get("payload"))
    triage = _as_mapping(record.get("genai_triage"))
    decision = _as_mapping(record.get("decision"))
    context_analysis = _as_mapping(context.get("genai_analysis"))

    if is_corrected:
        sources = tuple(explicit_correction_sources)
        origin = "analyst_correction"
    else:
        sources = (
            *explicit_correction_sources,
            review,
            decision,
            record,
            payload,
            triage,
            context_analysis,
        )
        origin = (
            "analyst_correction"
            if explicit_correction_sources
            else "analyst_approved"
        )

    materiality = _normalize_materiality(_pick(sources, _MATERIALITY_KEYS))
    change_nature = _normalize_change_nature(
        _pick(sources, _CHANGE_NATURE_KEYS)
    )
    if not materiality or not change_nature:
        return None

    text_before = _first_text(
        record,
        payload,
        keys=(
            "text_before",
            "semantic_text_t1",
            "source_text_t1",
            "old_text",
            "from",
        ),
        evidence_key="evidence_t1",
    )
    text_after = _first_text(
        record,
        payload,
        keys=(
            "text_after",
            "semantic_text_t2",
            "source_text_t2",
            "new_text",
            "to",
        ),
        evidence_key="evidence_t2",
    )
    if not text_before and not text_after:
        return None

    bank_code = _clean_text(
        _pick(
            (record, payload, context),
            ("bank_code", "bank", "institution"),
        )
    ).lower()
    if not bank_code:
        table_key = _clean_text(context.get("table_key") or context.get("review_id"))
        if "::" in table_key:
            bank_code = table_key.split("::", 1)[0].lower()
    section_key = _clean_text(
        _pick(
            (record, payload, context),
            ("section_key", "section", "section_title"),
        )
    )
    change_id = _clean_text(
        record.get("change_id")
        or record.get("id")
        or record.get("precedent_id")
    )
    themes = tuple(
        _string_list(
            _pick(
                sources,
                (
                    "corrected_themes_amf",
                    "final_themes_amf",
                    "themes_amf",
                    "themes",
                ),
            )
        )
    )
    rationale = _clean_text(_pick(sources, _RATIONALE_KEYS))
    equivalence = _normalize_equivalence(
        _pick(
            sources,
            (
                "corrected_business_equivalence",
                "final_business_equivalence",
                "business_equivalence",
                "equivalence_metier",
            ),
        )
    )
    source_reference = _clean_text(
        record.get("source_reference") or candidate.source_reference
    )
    precedent_id = _clean_text(record.get("precedent_id")) or _stable_precedent_id(
        bank_code=bank_code,
        section_key=section_key,
        change_id=change_id,
        text_before=text_before,
        text_after=text_after,
    )
    evidence_sufficiency = _normalize_identifier(
        _pick(
            sources,
            (
                "corrected_evidence_sufficiency",
                "final_evidence_sufficiency",
                "evidence_sufficiency",
            ),
        )
    )
    decision_status = _normalize_identifier(
        _pick(
            sources,
            (
                "corrected_decision_status",
                "final_decision_status",
                "decision_status",
            ),
        )
    )
    review_required = _normalize_bool(
        _pick(
            sources,
            (
                "corrected_review_required",
                "final_review_required",
                "review_required",
            ),
        ),
        default=False,
    )
    if review_required or decision_status in _NON_FINAL_DECISION_STATUSES:
        return None
    materiality_confidence = _normalize_confidence(
        _pick(
            sources,
            (
                "corrected_materiality_confidence",
                "final_materiality_confidence",
                "materiality_confidence",
                "confidence",
            ),
        )
    )
    is_relevant = _normalize_optional_bool(
        _pick(
            sources,
            (
                "corrected_is_relevant",
                "final_is_relevant",
                "is_relevant",
            ),
        )
    )
    supporting_evidence = tuple(
        _string_list(
            _pick(
                sources,
                (
                    "corrected_supporting_evidence",
                    "final_supporting_evidence",
                    "supporting_evidence",
                ),
            )
        )
    )
    counterarguments = tuple(
        _string_list(
            _pick(
                sources,
                (
                    "corrected_counterarguments",
                    "final_counterarguments",
                    "counterarguments",
                ),
            )
        )
    )
    nature_parts = set(change_nature.split("|"))
    if (
        not nature_parts
        or not nature_parts <= _ALLOWED_CHANGE_NATURES
        or equivalence
        not in {
            "CONFIRMEE",
            "PROBABLE",
            "NON_DEMONTREE",
            "REFUTEE",
        }
        or materiality_confidence not in {"ELEVEE", "MOYENNE"}
        or evidence_sufficiency != "SUFFISANTE"
        or decision_status not in {"CONFIRME", "VALIDATED_ANALYSTE"}
        or is_relevant is None
        or not rationale
        or not supporting_evidence
    ):
        return None
    if is_relevant:
        if (
            not 1 <= len(themes) <= 2
            or not set(themes) <= _ALLOWED_THEMES
        ):
            return None
    elif themes or materiality != "MINEUR":
        return None
    if (
        materiality == "MINEUR"
        and equivalence != "CONFIRMEE"
    ):
        return None
    if (
        materiality in {"MODERE", "MAJEUR"}
        and equivalence == "CONFIRMEE"
    ):
        return None

    return AnalystPrecedent(
        precedent_id=precedent_id,
        change_id=change_id,
        bank_code=bank_code,
        section_key=section_key,
        text_before=text_before,
        text_after=text_after,
        materiality_level=materiality,
        change_nature=change_nature,
        business_equivalence=equivalence,
        materiality_confidence=materiality_confidence,
        evidence_sufficiency=evidence_sufficiency,
        decision_status=decision_status,
        review_required=review_required,
        is_relevant=is_relevant,
        themes_amf=themes,
        rationale=rationale,
        supporting_evidence=supporting_evidence,
        counterarguments=counterarguments,
        reviewer=_clean_text(
            review.get("review_user")
            or review.get("validated_by")
            or record.get("validated_by")
            or context.get("username")
        ),
        validated_at=_clean_text(
            review.get("reviewed_at")
            or review.get("validated_at")
            or record.get("validated_at")
        ),
        decision_origin=origin,
        source_kind=candidate.source_kind,
        source_reference=source_reference,
    )


def _review_mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    review = record.get("_analyst_review")
    if isinstance(review, Mapping):
        return review
    review = record.get("review")
    if isinstance(review, Mapping):
        return review
    if any(
        key in record
        for key in (
            "validation_status",
            "review_status",
            "validated_at",
            "validated_by",
        )
    ):
        return record
    return {}


def _explicit_correction_sources(
    record: Mapping[str, Any],
    review: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for container in (review, record):
        for key in _CORRECTION_CONTAINERS:
            nested = container.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)
        if any(
            str(key).startswith(("corrected_", "final_"))
            for key in container
        ):
            sources.append(container)
    return sources


def _candidate_identity(record: Mapping[str, Any]) -> str:
    change_id = _clean_text(record.get("change_id") or record.get("id"))
    if change_id:
        return change_id
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stable_precedent_id(
    *,
    bank_code: str,
    section_key: str,
    change_id: str,
    text_before: str,
    text_after: str,
) -> str:
    material = "\n".join(
        (
            bank_code,
            section_key,
            change_id,
            text_before,
            text_after,
        )
    )
    return f"ap_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20]}"


def _origin_priority(precedent: AnalystPrecedent) -> int:
    return 2 if precedent.decision_origin == "analyst_correction" else 1


def _same_precedent_decision(
    left: AnalystPrecedent,
    right: AnalystPrecedent,
) -> bool:
    """Compare les champs décisionnels qui ne peuvent diverger silencieusement."""
    return (
        left.materiality_level,
        left.change_nature,
        left.business_equivalence,
        left.evidence_sufficiency,
        left.decision_status,
        left.review_required,
        left.is_relevant,
        left.themes_amf,
    ) == (
        right.materiality_level,
        right.change_nature,
        right.business_equivalence,
        right.evidence_sufficiency,
        right.decision_status,
        right.review_required,
        right.is_relevant,
        right.themes_amf,
    )


def _lexical_score(
    query: PrecedentQuery,
    precedent: AnalystPrecedent,
) -> tuple[float, dict[str, float]]:
    query_before = _token_counter(query.text_before)
    query_after = _token_counter(query.text_after)
    precedent_before = _token_counter(precedent.text_before)
    precedent_after = _token_counter(precedent.text_after)
    query_combined = query_before + query_after
    precedent_combined = precedent_before + precedent_after
    query_delta = (query_after - query_before) + (query_before - query_after)
    precedent_delta = (precedent_after - precedent_before) + (
        precedent_before - precedent_after
    )

    combined = _cosine_counters(query_combined, precedent_combined)
    before = _side_similarity(
        query_before,
        precedent_before,
        query_side_present=bool(_clean_text(query.text_before)),
        precedent_side_present=bool(_clean_text(precedent.text_before)),
    )
    after = _side_similarity(
        query_after,
        precedent_after,
        query_side_present=bool(_clean_text(query.text_after)),
        precedent_side_present=bool(_clean_text(precedent.text_after)),
    )
    delta = _cosine_counters(query_delta, precedent_delta)
    nature = _identifier_similarity(query.change_nature, precedent.change_nature)
    themes = _set_similarity(query.themes_amf, precedent.themes_amf)
    section = _identifier_similarity(query.section_key, precedent.section_key)
    bank = (
        1.0
        if query.bank_code
        and precedent.bank_code
        and _normalize_text(query.bank_code) == _normalize_text(precedent.bank_code)
        else 0.0
    )
    breakdown = {
        "combined_text": combined,
        "text_before": before,
        "text_after": after,
        "change_delta": delta,
        "change_nature": nature,
        "themes": themes,
        "section": section,
        "bank": bank,
    }
    score = (
        (0.34 * combined)
        + (0.14 * before)
        + (0.14 * after)
        + (0.16 * delta)
        + (0.09 * nature)
        + (0.06 * themes)
        + (0.05 * section)
        + (0.02 * bank)
    )
    breakdown["total"] = score
    return score, breakdown


def _embed(
    engine: EmbeddingEngine
    | Callable[[Sequence[str]], Sequence[Sequence[float]]],
    texts: Sequence[str],
) -> list[tuple[float, ...]]:
    if hasattr(engine, "embed"):
        raw = engine.embed(texts)  # type: ignore[union-attr]
    elif callable(engine):
        raw = engine(texts)
    else:
        raise TypeError("Le moteur d'embeddings ne fournit pas embed().")

    vectors = [tuple(float(value) for value in vector) for vector in raw]
    if len(vectors) != len(texts):
        raise ValueError("Nombre de vecteurs incohérent.")
    dimensions = {len(vector) for vector in vectors}
    if not dimensions or 0 in dimensions or len(dimensions) != 1:
        raise ValueError("Dimensions d'embeddings incohérentes.")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Embedding non fini.")
    return vectors


def _cosine_vectors(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_counters(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _side_similarity(
    left: Counter[str],
    right: Counter[str],
    *,
    query_side_present: bool,
    precedent_side_present: bool,
) -> float:
    if not query_side_present and not precedent_side_present:
        return 1.0
    if query_side_present != precedent_side_present:
        return 0.0
    return _cosine_counters(left, right)


def _identifier_similarity(left: str, right: str) -> float:
    left_normalized = _normalize_identifier(left)
    right_normalized = _normalize_identifier(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return _set_similarity(
        left_normalized.split("_"),
        right_normalized.split("_"),
    )


def _set_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {_normalize_identifier(value) for value in left if value}
    right_set = {_normalize_identifier(value) for value in right if value}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _token_counter(text: str) -> Counter[str]:
    tokens = [
        token
        for token in _TOKEN_RE.findall(_normalize_text(text))
        if token not in _STOP_WORDS and len(token) > 1
    ]
    return Counter(tokens)


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _clean_text(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())


def _normalize_identifier(value: Any) -> str:
    normalized = _normalize_text(value)
    return re.sub(r"[^A-Z0-9]+", "_", normalized.upper()).strip("_")


def _normalize_change_nature(value: Any) -> str:
    """Préserve une à trois natures structurées dans une clé stable."""
    normalized_values = [
        _normalize_identifier(item)
        for item in _string_list(value)
        if _normalize_identifier(item)
    ]
    return "|".join(dict.fromkeys(normalized_values[:3]))


def _normalize_materiality(value: Any) -> str:
    return _MATERIALITY_ALIASES.get(_normalize_identifier(value), "")


def _normalize_status(value: Any) -> str:
    return _normalize_text(value).replace(" ", "_")


def _normalize_equivalence(value: Any) -> str:
    if isinstance(value, bool):
        return "CONFIRMEE" if value else "REFUTEE"
    normalized = _normalize_identifier(value)
    if normalized in {
        "CONFIRMEE",
        "OUI",
        "TRUE",
        "EQUIVALENT",
        "EQUIVALENTE",
    }:
        return "CONFIRMEE"
    if normalized in {"PROBABLE"}:
        return "PROBABLE"
    if normalized in {"NON_DEMONTREE", "NON_DEMONTRE"}:
        return "NON_DEMONTREE"
    if normalized in {
        "REFUTEE",
        "REFUTE",
        "NON",
        "FALSE",
        "NON_EQUIVALENT",
        "NON_EQUIVALENTE",
    }:
        return "REFUTEE"
    return "INDETERMINE"


def _normalize_confidence(value: Any) -> str | float | None:
    if value is None or isinstance(value, bool):
        return None
    categorical = _normalize_identifier(value)
    if categorical in {"ELEVEE", "MOYENNE", "FAIBLE", "INDETERMINE"}:
        return categorical
    try:
        result = float(str(value).strip().replace(",", ".").rstrip("%"))
    except (TypeError, ValueError):
        return None
    if result > 1.0 and result <= 100.0:
        result /= 100.0
    if result < 0.0 or result > 1.0 or not math.isfinite(result):
        return None
    return result


def _normalize_bool(value: Any, *, default: bool) -> bool:
    normalized = _normalize_optional_bool(value)
    return default if normalized is None else normalized


def _normalize_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _normalize_identifier(value)
    if normalized in {"1", "OUI", "TRUE", "VRAI"}:
        return True
    if normalized in {"0", "NON", "FALSE", "FAUX"}:
        return False
    return None


def _first_text(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    keys: Sequence[str],
    evidence_key: str,
) -> str:
    value = _clean_text(_pick((primary, secondary), keys))
    if value:
        return value
    evidence = _as_mapping(primary.get(evidence_key))
    value = _clean_text(evidence.get("snippet"))
    if value:
        return value
    segments = primary.get("change_segments")
    if not isinstance(segments, list):
        triage = _as_mapping(primary.get("genai_triage"))
        segments = triage.get("change_segments")
    if not isinstance(segments, list):
        return ""
    segment_key = "text_t1" if evidence_key.endswith("t1") else "text_t2"
    return "\n".join(
        _clean_text(segment.get(segment_key))
        for segment in segments
        if isinstance(segment, Mapping) and _clean_text(segment.get(segment_key))
    )


def _pick(
    sources: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            value = source.get(key)
            if value is not None and value != "" and value != []:
                return value
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, Mapping):
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return [serialized]
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            else:
                text = _clean_text(item)
            if text:
                result.append(text)
        return result
    text = _clean_text(value)
    return [text] if text else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _clip(value: str, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"

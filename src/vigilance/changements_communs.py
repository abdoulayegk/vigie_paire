"""RAG/LLM helpers for common cross-bank disclosure changes.

This module is intentionally downstream-only: it reads existing
``text_comparison.json`` artifacts and prepares atomic change records for a
semantic analysis of common changes between banks. It does not mutate or
rerun the extraction and comparison pipelines.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vigilance.analyst_change_presentation import build_change_presentation
from vigilance.config import resolve_openai_model
from vigilance.ui_config import RESULTATS_DIR, TEXT_COMPARISON_DIR
from vigilance.utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)

CHANGEMENTS_COMMUNS_SCHEMA_VERSION = 2
DEFAULT_CHANGEMENTS_COMMUNS_DIR = RESULTATS_DIR / "changements_communs_banques"
DEFAULT_CHANGEMENTS_COMMUNS_PATH = (
    DEFAULT_CHANGEMENTS_COMMUNS_DIR / "changements_communs_banques.json"
)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_CANDIDATES = 160
DEFAULT_CANDIDATES_PER_DISCOVERY_QUERY = 35
DEFAULT_DISCOVERY_QUERIES = (
    "tarifs douaniers, incertitude commerciale, geopolitique, recession, inflation",
    "risque de credit, pertes sur creances, provisions, correction de valeur",
    "risque de marche, VAR, volatilite, derives, expositions de negociation",
    "fonds propres reglementaires, capital, CET1, TLAC, levier, actifs ponderes",
    "liquidite, ratio de liquidite a court terme, financement stable, LCR, NSFR",
    "risque climatique, durabilite, ESG, B-15, NCID, CSRD, IFRS S2, emissions GES",
    "faits nouveaux reglementaires, BSIF, ACVM, exigences de divulgation",
    "risques emergents, gestion integree des risques, controle et gouvernance",
)


@dataclass(frozen=True, slots=True)
class ChangementCommunRecord:
    """Atomic disclosure change record used as a RAG document."""

    record_id: str
    bank_code: str
    period: str
    source_path: str
    change_id: str
    section_title: str
    subsection_heading: str
    diff_type: str
    change_summary: str
    text_before: str
    text_after: str
    pages_before: tuple[int, ...]
    pages_after: tuple[int, ...]
    themes: tuple[str, ...]
    impact_level: str
    impact_it: str
    impact_it_justification: str
    changement_posture: str
    justification_posture: str
    statut_mise_en_oeuvre: str
    confiance_posture: str

    def retrieval_text(self, max_chars: int = 1800) -> str:
        """Return compact semantic text for embedding retrieval."""
        parts = [
            f"Banque: {self.bank_code.upper()}",
            f"Periode: {self.period}",
            f"Section: {self.section_title}",
            f"Sous-section: {self.subsection_heading}",
            f"Type: {self.diff_type}",
            f"Impact: {self.impact_level}",
            f"Impact IT: {self.impact_it}",
            f"Posture: {self.changement_posture}",
            f"Justification posture: {self.justification_posture}",
            f"Mise en oeuvre: {self.statut_mise_en_oeuvre}",
            f"Confiance posture: {self.confiance_posture}",
            f"Themes: {', '.join(self.themes)}",
            f"Resume: {self.change_summary}",
            f"Avant: {self.text_before}",
            f"Apres: {self.text_after}",
        ]
        return _truncate(_clean_text(" | ".join(parts)), max_chars)

    def prompt_dict(self, max_text_chars: int = 900) -> dict[str, Any]:
        """Return the compact representation passed to the LLM judge."""
        return {
            "record_id": self.record_id,
            "bank_code": self.bank_code,
            "period": self.period,
            "section": self.section_title,
            "subsection": self.subsection_heading,
            "diff_type": self.diff_type,
            "impact_level": self.impact_level,
            "impact_it": self.impact_it,
            "impact_it_justification": self.impact_it_justification,
            "changement_posture": self.changement_posture,
            "justification_posture": self.justification_posture,
            "statut_mise_en_oeuvre": self.statut_mise_en_oeuvre,
            "confiance_posture": self.confiance_posture,
            "themes": list(self.themes),
            "change_summary": self.change_summary,
            "text_before": _truncate(self.text_before, max_text_chars),
            "text_after": _truncate(self.text_after, max_text_chars),
        }

    def evidence_dict(self) -> dict[str, Any]:
        """Return full evidence metadata for dashboard rendering."""
        return {
            "record_id": self.record_id,
            "bank_code": self.bank_code,
            "period": self.period,
            "source_path": self.source_path,
            "change_id": self.change_id,
            "section": self.section_title,
            "subsection": self.subsection_heading,
            "diff_type": self.diff_type,
            "change_summary": self.change_summary,
            "text_before": self.text_before,
            "text_after": self.text_after,
            "pages_before": list(self.pages_before),
            "pages_after": list(self.pages_after),
            "themes": list(self.themes),
            "impact_level": self.impact_level,
            "impact_it": self.impact_it,
            "impact_it_justification": self.impact_it_justification,
            "changement_posture": self.changement_posture,
            "justification_posture": self.justification_posture,
            "statut_mise_en_oeuvre": self.statut_mise_en_oeuvre,
            "confiance_posture": self.confiance_posture,
        }


def collect_changements_communs_records(
    root_dir: Path | str | None = None,
    *,
    period: str | None = None,
    include_unchanged: bool = False,
) -> list[ChangementCommunRecord]:
    """Collect atomic text changes from available result folders.

    Args:
        root_dir: Optional root directory containing ``<bank>/<period>`` result
            folders. Defaults to ``TEXT_COMPARISON_DIR``.
        period: Optional period folder to include, for example
            ``2025_t2_vs_2025_t1``. When provided, only that comparison
            period is collected across banks.
        include_unchanged: Whether unchanged blocks should be indexed.

    Returns:
        Sorted list of atomic change records.
    """
    root = Path(root_dir) if root_dir else TEXT_COMPARISON_DIR
    normalized_period = normalize_changements_communs_period(period)
    records: list[ChangementCommunRecord] = []
    for path in sorted(root.glob("*/*/text_comparison.json")):
        if normalized_period and path.parent.name != normalized_period:
            continue
        records.extend(
            _records_from_text_comparison(path, include_unchanged=include_unchanged)
        )
    return records


def normalize_changements_communs_period(period: str | None) -> str | None:
    """Normalize a period folder name supplied by CLI/UI."""
    text = str(period or "").strip().lower()
    if not text:
        return None
    return text.replace("-", "_")


def changements_communs_output_path(
    period: str | None,
    *,
    base_dir: Path | str | None = None,
) -> Path:
    """Return the default period-scoped common-changes artifact path."""
    normalized_period = normalize_changements_communs_period(period)
    if not normalized_period:
        return DEFAULT_CHANGEMENTS_COMMUNS_PATH
    root = Path(base_dir) if base_dir else DEFAULT_CHANGEMENTS_COMMUNS_DIR
    return root / normalized_period / "changements_communs_banques.json"


def list_changements_communs_periods(root_dir: Path | str | None = None) -> list[str]:
    """List result periods that contain at least one text comparison."""
    root = Path(root_dir) if root_dir else TEXT_COMPARISON_DIR
    periods = {
        path.parent.name
        for path in root.glob("*/*/text_comparison.json")
        if path.parent.name
    }
    return sorted(periods)


def latest_changements_communs_report_path(
    base_dir: Path | str | None = None,
) -> Path | None:
    """Return the most recently modified period-scoped report path."""
    root = Path(base_dir) if base_dir else DEFAULT_CHANGEMENTS_COMMUNS_DIR
    candidates = [
        path
        for path in root.glob("*/changements_communs_banques.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_changements_communs_source_stats(records: list[ChangementCommunRecord]) -> dict[str, Any]:
    """Summarize source records available for common-change generation."""
    bank_counts: dict[str, int] = {}
    period_counts: dict[str, int] = {}
    impact_counts: dict[str, int] = {}
    impact_it_counts: dict[str, int] = {}
    posture_counts: dict[str, int] = {}
    implementation_counts: dict[str, int] = {}
    posture_confidence_counts: dict[str, int] = {}
    for record in records:
        bank_counts[record.bank_code] = bank_counts.get(record.bank_code, 0) + 1
        period_counts[record.period] = period_counts.get(record.period, 0) + 1
        impact = record.impact_level or "INCONNU"
        impact_counts[impact] = impact_counts.get(impact, 0) + 1
        impact_it = record.impact_it or "INDETERMINE"
        impact_it_counts[impact_it] = impact_it_counts.get(impact_it, 0) + 1
        posture = record.changement_posture or "INDETERMINE"
        posture_counts[posture] = posture_counts.get(posture, 0) + 1
        implementation = record.statut_mise_en_oeuvre or "INDETERMINE"
        implementation_counts[implementation] = (
            implementation_counts.get(implementation, 0) + 1
        )
        posture_confidence = record.confiance_posture or "INDETERMINE"
        posture_confidence_counts[posture_confidence] = (
            posture_confidence_counts.get(posture_confidence, 0) + 1
        )

    return {
        "total_changes": len(records),
        "bank_count": len(bank_counts),
        "banks": sorted(bank_counts),
        "bank_counts": dict(sorted(bank_counts.items())),
        "period_count": len(period_counts),
        "periods": sorted(period_counts),
        "impact_counts": dict(sorted(impact_counts.items())),
        "impact_it_counts": dict(sorted(impact_it_counts.items())),
        "posture_counts": dict(sorted(posture_counts.items())),
        "implementation_counts": dict(sorted(implementation_counts.items())),
        "posture_confidence_counts": dict(
            sorted(posture_confidence_counts.items())
        ),
    }


def retrieve_changements_communs_candidates(
    query: str,
    records: list[ChangementCommunRecord],
    *,
    client: Any,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_MAX_CANDIDATES,
) -> list[ChangementCommunRecord]:
    """Retrieve semantically relevant change records using OpenAI embeddings.

    Args:
        query: Analyst topic or question.
        records: Source records to search.
        client: OpenAI client instance.
        embedding_model: Embedding model name.
        top_k: Maximum number of candidate records to return.

    Returns:
        Records ranked by embedding similarity.
    """
    if not records:
        return []

    search_query = str(query or "").strip() or (
        "changements de divulgation similaires entre banques canadiennes"
    )
    query_embedding = _embed_texts(client, [search_query], embedding_model)[0]
    record_texts = [record.retrieval_text() for record in records]
    record_embeddings = _embed_texts(client, record_texts, embedding_model)

    scored = [
        (_cosine_similarity(query_embedding, embedding), record)
        for embedding, record in zip(record_embeddings, records, strict=False)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[: max(1, top_k)]]


def build_changements_communs_discovery_queries(topic: str) -> list[str]:
    """Build the semantic search queries used for broad per-period discovery."""
    queries: list[str] = []
    analyst_topic = str(topic or "").strip()
    if analyst_topic:
        queries.append(analyst_topic)
    queries.extend(DEFAULT_DISCOVERY_QUERIES)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = _clean_text(query).lower()
        if key and key not in seen:
            deduped.append(query)
            seen.add(key)
    return deduped


def retrieve_changements_communs_discovery_candidates(
    topic: str,
    records: list[ChangementCommunRecord],
    *,
    client: Any,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    candidates_per_query: int = DEFAULT_CANDIDATES_PER_DISCOVERY_QUERY,
) -> tuple[list[ChangementCommunRecord], list[str]]:
    """Retrieve candidates across multiple business themes for one period.

    The selection is round-robin across theme queries so a broad topic like
    capital or liquidity is not crowded out by the first query's nearest
    neighbours.
    """
    queries = build_changements_communs_discovery_queries(topic)
    if not records:
        return [], queries

    record_texts = [record.retrieval_text() for record in records]
    query_embeddings = _embed_texts(client, queries, embedding_model)
    record_embeddings = _embed_texts(client, record_texts, embedding_model)

    per_query_rankings: list[list[ChangementCommunRecord]] = []
    per_query_limit = max(1, int(candidates_per_query))
    for query_embedding in query_embeddings:
        scored = [
            (_cosine_similarity(query_embedding, embedding), record)
            for embedding, record in zip(record_embeddings, records, strict=False)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        per_query_rankings.append([record for _, record in scored[:per_query_limit]])

    selected: list[ChangementCommunRecord] = []
    seen_record_ids: set[str] = set()
    max_total = max(1, int(max_candidates))
    for rank in range(per_query_limit):
        for ranking in per_query_rankings:
            if rank >= len(ranking):
                continue
            record = ranking[rank]
            if record.record_id in seen_record_ids:
                continue
            selected.append(record)
            seen_record_ids.add(record.record_id)
            if len(selected) >= max_total:
                return selected, queries
    return selected, queries


def generate_changements_communs_report(
    topic: str,
    *,
    min_banks: int = 3,
    period: str | None = None,
    root_dir: Path | str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    candidates_per_query: int = DEFAULT_CANDIDATES_PER_DISCOVERY_QUERY,
    model: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate common cross-bank changes with RAG retrieval and an LLM judge.

    Args:
        topic: Analyst topic or free-form question.
        min_banks: Minimum distinct banks required for a consensus signal.
        period: Optional comparison period. This is the main business mode:
            compare all available banks for the same period only.
        root_dir: Optional result root.
        max_candidates: Maximum retrieved records sent to the LLM judge.
        candidates_per_query: Maximum records retrieved per discovery theme.
        model: Optional chat model override.
        embedding_model: Embedding model used for RAG retrieval.
        client: Optional OpenAI client, useful for tests.

    Returns:
        JSON-serializable common-changes report.

    Raises:
        RuntimeError: If no OpenAI API key is configured and no client is
            injected.
    """
    normalized_period = normalize_changements_communs_period(period)
    records = collect_changements_communs_records(
        root_dir=root_dir,
        period=normalized_period,
    )
    source_stats = build_changements_communs_source_stats(records)
    if client is None:
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY absent: la generation LLM des changements "
                "communs entre banques ne peut pas s'executer."
            )
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    candidates, discovery_queries = retrieve_changements_communs_discovery_candidates(
        topic,
        records,
        client=client,
        embedding_model=embedding_model,
        max_candidates=max_candidates,
        candidates_per_query=candidates_per_query,
    )
    messages = build_changements_communs_judge_messages(
        topic=topic,
        discovery_queries=discovery_queries,
        candidates=candidates,
        min_banks=min_banks,
    )
    response = client.chat.completions.create(
        model=model or resolve_openai_model("default_genai"),
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    payload = _parse_chat_json_response(response)
    report = _normalize_llm_report(
        payload,
        topic=topic,
        min_banks=min_banks,
        period=normalized_period,
        source_count=len(records),
        source_stats=source_stats,
        candidates=candidates,
        discovery_queries=discovery_queries,
    )
    return report


def build_changements_communs_judge_messages(
    *,
    topic: str,
    discovery_queries: list[str] | None = None,
    candidates: list[ChangementCommunRecord],
    min_banks: int = 3,
) -> list[dict[str, str]]:
    """Build chat messages for the LLM common-changes judge."""
    candidate_payload = [record.prompt_dict() for record in candidates]
    system = (
        "Tu es un analyste senior en divulgation bancaire et gestion integree "
        "des risques. Regroupe seulement les changements qui traitent du meme "
        "theme de risque ou de la meme evolution de divulgation. Les titres de "
        "section peuvent differer. Ne conclus jamais a un consensus sans preuve "
        "textuelle."
    )
    user = {
        "objectif": (
            "Decouvrir les signaux de divulgation comparables entre banques "
            "pour une meme periode. Le rapport doit contenir deux familles "
            "quand les preuves existent: les consensus a 3 banques ou plus "
            "et les signaux mineurs presents dans exactement 2 banques."
        ),
        "topic": topic,
        "discovery_queries": discovery_queries or [],
        "min_banks": min_banks,
        "instructions": [
            "Retourne uniquement du JSON valide.",
            "Utilise seulement les record_id fournis dans candidates.",
            "Cherche plusieurs themes communs, pas seulement le topic initial.",
            "Un consensus doit contenir au moins min_banks banques distinctes.",
            "Un signal present dans exactement 2 banques doit etre retourne avec status='signal_mineur_2_banques'.",
            "Ne t'arrete pas apres les consensus: cherche aussi les meilleurs signaux mineurs a 2 banques.",
            "Un signal mineur ne doit pas etre un simple sous-ensemble ou doublon d'un consensus deja retourne.",
            "Un signal present dans une seule banque doit etre ignore.",
            "Vise une couverture large mais controlee: jusqu'a 12 signaux au total si les preuves sont suffisantes.",
            "Pour chaque signal, explique les similarites et les differences.",
            "Evalue impact_it uniquement a partir des preuves fournies; utilise Indetermine si les rapports ne permettent pas de conclure.",
            "Resume changement_posture en distinguant renforcement, allegement, nouveau dispositif et retrait de dispositif.",
            "Compare le statut de mise en oeuvre entre banques sans confondre annonce, planification, travaux en cours et dispositif mis en oeuvre.",
            "Evalue la confiance de la synthese de posture uniquement a partir des preuves rattachees.",
            "Evite les doublons: ne separe pas artificiellement deux signaux qui portent sur le meme changement de pratique.",
            "Garde les citations courtes et rattache chaque preuve a un record_id.",
        ],
        "schema_attendu": {
            "signals": [
                {
                    "theme": "string",
                    "summary": "string",
                    "status": "consensus_3_plus | signal_mineur_2_banques",
                    "banks": ["bank_code"],
                    "impact": "Majeur | Modere | Mineur",
                    "impact_it": "Eleve | Moyen | Faible | Indetermine",
                    "posture_summary": "string",
                    "mise_en_oeuvre_summary": "string",
                    "confiance_posture": "Elevee | Moyenne | Faible | Indetermine",
                    "action": "string",
                    "rationale": "string",
                    "differences": "string",
                    "evidence": [
                        {
                            "record_id": "string",
                            "quote": "short exact quote",
                            "why_relevant": "string",
                        }
                    ],
                }
            ]
        },
        "candidates": candidate_payload,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def load_changements_communs_report(
    path: Path | str | None = None,
    *,
    period: str | None = None,
) -> dict[str, Any] | None:
    """Load a previously generated common-changes artifact if it exists."""
    report_path = Path(path) if path else changements_communs_output_path(period)
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Erreur lecture changements communs entre banques %s: %s",
            report_path,
            exc,
        )
        return None


def write_changements_communs_report(
    report: dict[str, Any],
    path: Path | str | None = None,
    *,
    period: str | None = None,
) -> Path:
    """Persist a generated common-changes artifact."""
    report_path = Path(path) if path else changements_communs_output_path(
        period or report.get("period")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def _records_from_text_comparison(
    path: Path,
    *,
    include_unchanged: bool,
) -> list[ChangementCommunRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Erreur lecture text_comparison %s: %s", path, exc)
        return []

    bank_code = str(payload.get("bank_code") or path.parent.parent.name).lower()
    period = path.parent.name
    records: list[ChangementCommunRecord] = []
    for section in payload.get("section_comparisons", []) or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("section_title") or section.get("section_key") or "")
        for index, block in enumerate(section.get("block_comparisons", []) or []):
            if not isinstance(block, dict):
                continue
            diff_type = str(block.get("diff_type") or "").strip().lower()
            if diff_type == "unchanged" and not include_unchanged:
                continue
            record = _record_from_block(
                block,
                bank_code=bank_code,
                period=period,
                source_path=str(path),
                section_title=section_title,
                index=index,
            )
            if record is not None:
                records.append(record)
    return records


def _record_from_block(
    block: dict[str, Any],
    *,
    bank_code: str,
    period: str,
    source_path: str,
    section_title: str,
    index: int,
) -> ChangementCommunRecord | None:
    change_id = str(block.get("change_id") or f"change_{index:04d}").strip()
    block_t1 = block.get("block_t1") if isinstance(block.get("block_t1"), dict) else {}
    block_t2 = block.get("block_t2") if isinstance(block.get("block_t2"), dict) else {}

    text_before = str(
        block.get("source_text_t1")
        or block.get("semantic_text_t1")
        or block_t1.get("text")
        or ""
    ).strip()
    text_after = str(
        block.get("source_text_t2")
        or block.get("semantic_text_t2")
        or block_t2.get("text")
        or ""
    ).strip()
    triage = block.get("genai_triage") if isinstance(block.get("genai_triage"), dict) else {}
    raw_change_summary = str(
        triage.get("changement_constate")
        or block.get("change_summary")
        or ""
    ).strip()
    if not any([text_before, text_after, raw_change_summary]):
        return None
    change_summary = build_change_presentation(
        block,
        bank_code=bank_code,
        candidate_summary=raw_change_summary,
    ).summary

    themes = tuple(str(theme) for theme in triage.get("themes_amf", []) or [])
    impact_level = str(triage.get("impact_level") or block.get("impact_level") or "").strip()
    impact_it = str(triage.get("impact_it") or "INDETERMINE").strip()
    impact_it_justification = str(
        triage.get("impact_it_justification") or ""
    ).strip()
    changement_posture = str(
        triage.get("changement_posture") or "INDETERMINE"
    ).strip()
    justification_posture = str(
        triage.get("justification_posture") or ""
    ).strip()
    statut_mise_en_oeuvre = str(
        triage.get("statut_mise_en_oeuvre") or "INDETERMINE"
    ).strip()
    confiance_posture = str(
        triage.get("confiance_posture") or "INDETERMINE"
    ).strip()
    pages_before = _page_tuple(block.get("pages_t1") or block_t1.get("page"))
    pages_after = _page_tuple(block.get("pages_t2") or block_t2.get("page"))
    subsection = str(block.get("subsection_heading") or "").strip() or "Non classee"
    record_id = f"{bank_code}:{period}:{change_id}"

    return ChangementCommunRecord(
        record_id=record_id,
        bank_code=bank_code,
        period=period,
        source_path=source_path,
        change_id=change_id,
        section_title=section_title or "Section inconnue",
        subsection_heading=subsection,
        diff_type=str(block.get("diff_type") or "").strip().lower() or "unknown",
        change_summary=change_summary,
        text_before=text_before,
        text_after=text_after,
        pages_before=pages_before,
        pages_after=pages_after,
        themes=themes,
        impact_level=impact_level or "INCONNU",
        impact_it=impact_it or "INDETERMINE",
        impact_it_justification=impact_it_justification,
        changement_posture=changement_posture or "INDETERMINE",
        justification_posture=justification_posture,
        statut_mise_en_oeuvre=statut_mise_en_oeuvre or "INDETERMINE",
        confiance_posture=confiance_posture or "INDETERMINE",
    )


def _normalize_llm_report(
    payload: dict[str, Any],
    *,
    topic: str,
    min_banks: int,
    period: str | None,
    source_count: int,
    source_stats: dict[str, Any],
    candidates: list[ChangementCommunRecord],
    discovery_queries: list[str],
) -> dict[str, Any]:
    by_id = {record.record_id: record for record in candidates}
    normalized_signals: list[dict[str, Any]] = []
    for raw_signal in payload.get("signals", []) or []:
        if not isinstance(raw_signal, dict):
            continue
        raw_evidence = raw_signal.get("evidence", []) or []
        evidence: list[dict[str, Any]] = []
        for raw_item in raw_evidence:
            if not isinstance(raw_item, dict):
                continue
            record_id = str(raw_item.get("record_id") or "").strip()
            record = by_id.get(record_id)
            if record is None:
                continue
            item = record.evidence_dict()
            item["quote"] = str(raw_item.get("quote") or "").strip()
            item["why_relevant"] = str(raw_item.get("why_relevant") or "").strip()
            evidence.append(item)
        banks = sorted({str(item.get("bank_code") or "").lower() for item in evidence})
        if not banks:
            banks = sorted(str(bank).lower() for bank in raw_signal.get("banks", []) or [])
        bank_count = len(set(banks))
        if bank_count < 2:
            continue
        min_banks_met = bank_count >= min_banks
        status = "consensus_3_plus" if min_banks_met else "signal_mineur_2_banques"
        relevance_level = "consensus" if min_banks_met else "mineur"
        normalized_signals.append(
            {
                "theme": str(raw_signal.get("theme") or "").strip(),
                "summary": str(raw_signal.get("summary") or "").strip(),
                "status": status,
                "relevance_level": relevance_level,
                "banks": banks,
                "bank_count": bank_count,
                "min_banks_met": min_banks_met,
                "impact": str(raw_signal.get("impact") or "").strip(),
                "impact_it": str(raw_signal.get("impact_it") or "").strip(),
                "posture_summary": str(
                    raw_signal.get("posture_summary") or ""
                ).strip(),
                "mise_en_oeuvre_summary": str(
                    raw_signal.get("mise_en_oeuvre_summary") or ""
                ).strip(),
                "confiance_posture": str(
                    raw_signal.get("confiance_posture") or ""
                ).strip(),
                "action": str(raw_signal.get("action") or "").strip(),
                "rationale": str(raw_signal.get("rationale") or "").strip(),
                "differences": str(raw_signal.get("differences") or "").strip(),
                "evidence": evidence,
            }
        )

    normalized_signals.sort(
        key=lambda signal: (
            not bool(signal.get("min_banks_met")),
            -int(signal.get("bank_count", 0) or 0),
            str(signal.get("theme") or ""),
        )
    )
    consensus_count = sum(1 for signal in normalized_signals if signal["min_banks_met"])
    minor_count = len(normalized_signals) - consensus_count

    return {
        "schema_version": CHANGEMENTS_COMMUNS_SCHEMA_VERSION,
        "artifact_type": "changements_communs_banques",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "period": period,
        "analysis_scope": "single_period" if period else "all_periods",
        "min_banks": min_banks,
        "source_change_count": source_count,
        "source_stats": source_stats,
        "candidate_count": len(candidates),
        "discovery_queries": discovery_queries,
        "signal_counts": {
            "total": len(normalized_signals),
            "consensus": consensus_count,
            "minor": minor_count,
        },
        "signals": normalized_signals,
    }


def _parse_chat_json_response(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    content = choice.message.content
    if not isinstance(content, str):
        raise ValueError("OpenAI response did not contain text content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response JSON root must be an object")
    return payload


def _embed_texts(client: Any, texts: list[str], model: str) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 96
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend([list(item.embedding) for item in ordered])
    return embeddings


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _page_tuple(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, list | tuple):
        pages: list[int] = []
        for value in raw:
            try:
                pages.append(int(value))
            except (TypeError, ValueError):
                continue
        return tuple(pages)
    try:
        return (int(raw),)
    except (TypeError, ValueError):
        return ()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate(value: str, max_chars: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}..."

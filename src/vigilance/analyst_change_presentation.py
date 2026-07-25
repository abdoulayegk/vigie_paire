"""Règles communes de présentation des changements pour les analystes.

Le pipeline conserve ses identifiants techniques (T1/T2, rôles atomiques,
raisons d'exclusion) dans les artefacts. Ce module construit une vue métier
stable pour Dash et Excel sans modifier les preuves sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from vigilance.i18n.fr import sanitize_analyst_french

_SECONDARY_EXCLUSION_REASONS = {
    "deplacement_texte",
    "formatage_visuel",
    "mise_a_jour_calendrier",
    "operation_interne_banque",
    "reformulation_mineure",
    "variation_numerique_propre_banque",
}

_CHANGE_VERBS_RE = re.compile(
    r"\b("
    r"ajout(?:e|ent)|retir(?:e|ent)|supprim(?:e|ent)|précis(?:e|ent)|"
    r"élargi(?:t|ssent)|restrei(?:nt|gnent)|remplac(?:e|ent)|"
    r"reformul(?:e|ent)|modifi(?:e|ent)|renomm(?:e|ent)|"
    r"introdui(?:t|sent)|inclu(?:t|ent)|indiqu(?:e|ent)|"
    r"mentionn(?:e|ent)|omet(?:|tent)|identifi(?:e|ent)|"
    r"renforc(?:e|ent)|actualis(?:e|ent)|décri(?:t|vent)|"
    r"enrichi(?:t|ssent)|distingu(?:e|ent)|redéfini(?:t|ssent)|"
    r"attribu(?:e|ent)|simplifi(?:e|ent)|clarifi(?:e|ent)|"
    r"réorganis(?:e|ent)|regroup(?:e|ent)|sépar(?:e|ent)|"
    r"reclass(?:e|ent)|exclu(?:t|ent)"
    r")\b",
    flags=re.IGNORECASE,
)

_GENERIC_META_RE = re.compile(
    r"^(?:"
    r"les deux (?:passages|textes)|"
    r"divulgation distincte|"
    r"sous-section (?:ajoutée|retirée|supprimée)|"
    r"le texte (?:a été|est) modifié"
    r")\b",
    flags=re.IGNORECASE,
)

_LEADING_ANALYSIS_LABEL_RE = re.compile(
    r"^(?:ce qui change|changement constaté)\s*:\s*",
    flags=re.IGNORECASE,
)

_GENERIC_RELEVANCE_PREFIX_RE = re.compile(
    r"^(?:"
    r"pour la vigie(?:\s+AMF|\s+prudentielle)?|"
    r"dans le cadre de (?:cette\s+|l['’])analyse"
    r")\s*[,;:–—-]\s*",
    flags=re.IGNORECASE,
)

_GENERIC_RELEVANCE_OBSERVATION_RE = re.compile(
    r"^il convient de noter que\s+",
    flags=re.IGNORECASE,
)

_GENERIC_RELEVANCE_SENTENCE_RE = re.compile(
    r"^(?:"
    r"cette information est importante"
    r"(?: pour (?:la vigie(?:\s+AMF|\s+prudentielle)?|"
    r"l['’]analyse|la comparaison entre pairs))?|"
    r"ce changement est pertinent pour la vigie"
    r"(?:\s+AMF|\s+prudentielle)?"
    r")\s*[.!?]?$",
    flags=re.IGNORECASE,
)

_GENERIC_IMPORTANCE_CAUSE_RE = re.compile(
    r"^cette information est importante"
    r"(?: pour [^,.;]+)?\s*,?\s*"
    r"(?:car elle|puisqu['’]elle)\s+",
    flags=re.IGNORECASE,
)

_CURRENT_REPORT_SUBJECT_RE = re.compile(
    r"(^|(?<=[.!?:])\s+)"
    r"(?:le\s+)?(?:t2|rapport courant|texte courant|document courant)"
    r"(?=\s+)",
    flags=re.IGNORECASE,
)

_GENERIC_BANK_SUBJECT_RE = re.compile(
    r"(^|(?<=[.!?:])\s+)(?:la banque|l'institution)(?=\s+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AnalystChangePresentation:
    """Vue métier compacte d'un changement textuel."""

    summary: str
    nature_label: str
    scope: str
    quality_status: str
    quality_issues: tuple[str, ...]


def bank_subject(bank_code: str | None) -> str:
    """Retourne le sujet court utilisé dans les phrases analystes."""
    value = " ".join(str(bank_code or "").strip().split())
    if not value:
        return "La banque"
    if len(value) <= 8 and " " not in value:
        return value.upper()
    return value


def canonicalize_analyst_narrative(
    value: str,
    *,
    bank_code: str | None,
) -> str:
    """Nettoie un texte IA et remplace le rapport courant par la banque."""
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    subject = bank_subject(bank_code)
    normalized_lines: list[str] = []
    for raw_line in raw_value.splitlines():
        normalized = " ".join(raw_line.split()).strip()
        if not normalized:
            normalized_lines.append("")
            continue
        normalized = _CURRENT_REPORT_SUBJECT_RE.sub(
            lambda match: f"{match.group(1)}{subject}",
            normalized,
        )
        if subject != "La banque":
            normalized = _GENERIC_BANK_SUBJECT_RE.sub(
                lambda match: f"{match.group(1)}{subject}",
                normalized,
            )
        normalized_lines.append(sanitize_analyst_french(normalized))
    return "\n".join(normalized_lines).strip()


def _truncate_at_sentence(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    window = value[:limit]
    sentence_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if sentence_end >= max(60, limit // 3):
        return window[: sentence_end + 1].strip()
    word_end = window.rfind(" ")
    if word_end >= max(60, limit // 3):
        return window[:word_end].rstrip(" ,;:") + "."
    return window.rstrip(" ,;:") + "."


def _first_complete_sentence(value: str) -> str:
    """Conserve une seule idée principale lorsque le candidat est explicatif."""
    match = re.search(
        r"[.!?](?=\s+[A-ZÀ-ÖØ-Þ])",
        value,
    )
    if match:
        return value[: match.end()].strip()
    return value


def _sentence_parts(value: str) -> list[str]:
    normalized = " ".join(str(value or "").split()).strip()
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])",
            normalized,
        )
        if part.strip()
    ]


def _sentence_comparison_key(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _capitalize_sentence_start(value: str) -> str:
    match = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", value)
    if match is None:
        return value
    index = match.start()
    return f"{value[:index]}{value[index].upper()}{value[index + 1:]}"


def _clean_business_relevance_sentence(value: str) -> str:
    """Retire les introductions génériques sans altérer l'idée métier."""
    sentence = " ".join(str(value or "").split()).strip()
    if not sentence or _GENERIC_RELEVANCE_SENTENCE_RE.fullmatch(sentence):
        return ""

    previous = ""
    while sentence != previous:
        previous = sentence
        sentence = _GENERIC_RELEVANCE_PREFIX_RE.sub("", sentence).strip()
        sentence = _GENERIC_RELEVANCE_OBSERVATION_RE.sub("", sentence).strip()

    sentence = _GENERIC_IMPORTANCE_CAUSE_RE.sub("Elle ", sentence).strip()
    return _capitalize_sentence_start(sentence)


def _duplicates_summary(sentence: str, summary: str) -> bool:
    sentence_key = _sentence_comparison_key(sentence)
    summary_key = _sentence_comparison_key(summary)
    if not sentence_key or not summary_key:
        return False
    if sentence_key == summary_key:
        return True
    if min(len(sentence_key), len(summary_key)) >= 60 and (
        sentence_key in summary_key or summary_key in sentence_key
    ):
        return True
    return SequenceMatcher(None, sentence_key, summary_key, autojunk=False).ratio() >= 0.88


def business_relevance_paragraph(
    *candidates: str,
    summary: str,
    bank_code: str | None,
    limit: int = 720,
) -> str:
    """Retourne jusqu'à trois phrases métier sans répéter le résumé factuel."""
    for candidate in candidates:
        narrative = canonicalize_analyst_narrative(
            candidate,
            bank_code=bank_code,
        )
        relevant_sentences: list[str] = []
        for sentence in _sentence_parts(narrative):
            cleaned_sentence = _clean_business_relevance_sentence(sentence)
            if not cleaned_sentence or _duplicates_summary(cleaned_sentence, summary):
                continue
            relevant_sentences.append(cleaned_sentence)
        if not relevant_sentences:
            continue
        paragraph = " ".join(relevant_sentences[:3])
        return _truncate_at_sentence(paragraph, limit)
    return ""


def change_scope(change: dict[str, Any]) -> str:
    """Classe un changement comme qualitatif, secondaire ou masqué."""
    triage = change.get("genai_triage") or {}
    if (
        str(change.get("diff_type") or "").lower() == "unchanged"
        or str(triage.get("source") or "").lower() == "skip"
    ):
        return "hidden"

    exclusion_reason = str(triage.get("exclusion_reason") or "").strip().lower()
    alignment_decision = str(change.get("alignment_decision") or "").strip().lower()
    if exclusion_reason in _SECONDARY_EXCLUSION_REASONS or alignment_decision == "moved_text":
        return "secondary"

    if not triage:
        return "qualitative"
    if bool(triage.get("is_relevant")) or bool(triage.get("nouvelle_idee")):
        return "qualitative"
    if "is_relevant" not in triage and "nouvelle_idee" not in triage:
        return "qualitative"
    return "secondary"


def _fallback_summary(change: dict[str, Any], *, subject: str) -> str:
    diff_type = str(change.get("diff_type") or "").strip().lower()
    if diff_type == "added":
        verb = "ajoute"
        source = change.get("source_text_t2") or change.get("semantic_text_t2")
    elif diff_type == "removed":
        verb = "retire"
        source = change.get("source_text_t1") or change.get("semantic_text_t1")
    elif diff_type == "renamed":
        verb = "renomme"
        source = change.get("source_text_t2") or change.get("semantic_text_t2")
    else:
        verb = "modifie"
        source = (
            change.get("source_text_t2")
            or change.get("semantic_text_t2")
            or change.get("source_text_t1")
            or change.get("semantic_text_t1")
        )
    detail = " ".join(str(source or "").split()).strip()
    if detail:
        return f"{subject} {verb} la divulgation suivante : {detail}"
    return f"{subject} {verb} une divulgation de ce bloc."


def _ensure_bank_subject(
    value: str,
    *,
    change: dict[str, Any],
    subject: str,
) -> str:
    if not value:
        return _fallback_summary(change, subject=subject)
    if value.casefold().startswith(subject.casefold()):
        return value

    diff_type = str(change.get("diff_type") or "").strip().lower()
    verb = {
        "added": "ajoute",
        "removed": "retire",
        "renamed": "renomme",
        "modified": "modifie",
    }.get(diff_type, "modifie")
    lowered = value[0].lower() + value[1:] if value else value
    return f"{subject} {verb} la divulgation suivante : {lowered}"


def _normalize_subject_verb(value: str, *, subject: str) -> str:
    """Uniformise au présent les formulations actives déjà factuelles."""
    escaped_subject = re.escape(subject)
    replacements = (
        (rf"^{escaped_subject}\s+a supprimé\b", f"{subject} retire"),
        (rf"^{escaped_subject}\s+a retiré\b", f"{subject} retire"),
        (rf"^{escaped_subject}\s+ne (?:mentionne|présente|décrit) plus\b", f"{subject} retire"),
        (rf"^{escaped_subject}\s+ne (?:liste|reprend) (?:plus|pas)\b", f"{subject} retire"),
        (rf"^{escaped_subject}\s+a ajouté\b", f"{subject} ajoute"),
        (rf"^{escaped_subject}\s+a introduit\b", f"{subject} introduit"),
        (rf"^{escaped_subject}\s+a précisé\b", f"{subject} précise"),
        (rf"^{escaped_subject}\s+a modifié\b", f"{subject} modifie"),
        (rf"^{escaped_subject}\s+a remplacé\b", f"{subject} remplace"),
        (rf"^{escaped_subject}\s+a élargi\b", f"{subject} élargit"),
    )
    normalized = value
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, count=1, flags=re.IGNORECASE)
    return normalized


def _nature_label(change: dict[str, Any], summary: str) -> str:
    normalized = summary.casefold()
    for needle, label in (
        (" ajoute ", "Ajout qualitatif"),
        (" introduit ", "Ajout qualitatif"),
        (" retire ", "Retrait qualitatif"),
        (" supprime ", "Retrait qualitatif"),
        (" précise ", "Précision"),
        (" élargit ", "Élargissement"),
        (" restreint ", "Restriction"),
        (" remplace ", "Remplacement"),
        (" renomme ", "Renommage"),
        (" reformule ", "Reformulation"),
        (" omet ", "Retrait qualitatif"),
        (" mentionne ", "Précision"),
        (" indique ", "Précision"),
        (" identifie ", "Précision"),
        (" renforce ", "Renforcement"),
        (" actualise ", "Mise à jour qualitative"),
        (" décrit ", "Précision"),
        (" enrichit ", "Précision"),
        (" distingue ", "Précision"),
        (" redéfinit ", "Modification"),
        (" attribue ", "Modification"),
        (" simplifie ", "Simplification"),
        (" clarifie ", "Précision"),
        (" réorganise ", "Réorganisation"),
        (" regroupe ", "Réorganisation"),
        (" sépare ", "Réorganisation"),
        (" reclasse ", "Réorganisation"),
        (" exclut ", "Retrait qualitatif"),
    ):
        if needle in f" {normalized} ":
            return label
    return {
        "added": "Ajout qualitatif",
        "removed": "Retrait qualitatif",
        "renamed": "Renommage",
        "modified": "Modification",
    }.get(str(change.get("diff_type") or "").lower(), "Changement")


def build_change_presentation(
    change: dict[str, Any],
    *,
    bank_code: str | None,
    candidate_summary: str,
    limit: int = 320,
) -> AnalystChangePresentation:
    """Construit le résumé canonique et les indicateurs de qualité."""
    subject = bank_subject(bank_code)
    raw_candidate = " ".join(str(candidate_summary or "").split()).strip()
    raw_candidate = _LEADING_ANALYSIS_LABEL_RE.sub("", raw_candidate, count=1)
    issues: list[str] = []

    if _GENERIC_META_RE.match(raw_candidate):
        issues.append("resume_generique")
        raw_candidate = ""

    summary = canonicalize_analyst_narrative(
        raw_candidate,
        bank_code=bank_code,
    )
    summary = _normalize_subject_verb(summary, subject=subject)
    summary = _ensure_bank_subject(summary, change=change, subject=subject)
    summary = _first_complete_sentence(summary)
    summary = _truncate_at_sentence(summary, limit)
    if summary and summary[-1] not in ".!?":
        summary += "."

    if not summary.casefold().startswith(subject.casefold()):
        issues.append("sujet_banque_absent")
    if not _CHANGE_VERBS_RE.search(summary):
        issues.append("verbe_changement_absent")
    if re.search(r"\bT[12]\b", summary, flags=re.IGNORECASE):
        issues.append("alias_periode_interne")

    unique_issues = tuple(dict.fromkeys(issues))
    return AnalystChangePresentation(
        summary=summary,
        nature_label=_nature_label(change, summary),
        scope=change_scope(change),
        quality_status="review" if unique_issues else "ready",
        quality_issues=unique_issues,
    )

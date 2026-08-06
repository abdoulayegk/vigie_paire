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

from vigie.support.i18n.fr import sanitize_analyst_french

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


@dataclass(frozen=True, slots=True)
class AnalystNarrative:
    """Unités sémantiques prêtes à être affichées à l'analyste.

    ``pertinence_metier`` est réservée aux changements qualitatifs retenus.
    ``motif_non_pertinence`` porte la justification des changements
    secondaires. ``business_relevance`` offre une vue commune aux exports qui
    affichent une seule colonne de justification.
    """

    changement_constate: str
    pertinence_metier: str
    motif_non_pertinence: str
    source: str

    @property
    def business_relevance(self) -> str:
        """Retourne l'analyse métier applicable à la décision de triage."""
        return self.pertinence_metier or self.motif_non_pertinence


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
    return f"{value[:index]}{value[index].upper()}{value[index + 1 :]}"


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
    if min(len(sentence_key), len(summary_key)) >= 60 and (sentence_key in summary_key or summary_key in sentence_key):
        return True
    return SequenceMatcher(None, sentence_key, summary_key, autojunk=False).ratio() >= 0.88


def business_relevance_paragraph(
    *candidates: str,
    summary: str,
    bank_code: str | None,
    limit: int = 720,
) -> str:
    """Retourne jusqu'à quatre phrases métier sans répéter le résumé factuel."""
    relevant_sentences: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        narrative = canonicalize_analyst_narrative(
            candidate,
            bank_code=bank_code,
        )
        for sentence in _sentence_parts(narrative):
            cleaned_sentence = _clean_business_relevance_sentence(sentence)
            key = _sentence_comparison_key(cleaned_sentence)
            if not cleaned_sentence or not key or key in seen or _duplicates_summary(cleaned_sentence, summary):
                continue
            seen.add(key)
            relevant_sentences.append(cleaned_sentence)
            if len(relevant_sentences) >= 4:
                break
        if len(relevant_sentences) >= 4:
            break

    if not relevant_sentences:
        return ""
    paragraph = " ".join(relevant_sentences)
    return _truncate_at_sentence(paragraph, limit)


def change_scope(change: dict[str, Any]) -> str:
    """Classe un changement comme qualitatif, secondaire ou masqué."""
    triage = change.get("genai_triage") or {}
    if str(change.get("diff_type") or "").lower() == "unchanged" or str(triage.get("source") or "").lower() == "skip":
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


_STRUCTURED_TRIAGE_FIELDS = frozenset(
    {
        "changement_constate",
        "signification_metier",
        "comparaison_interbanques",
        "comparaison_interbancaire",
        "limite_interpretation",
        "motif_non_pertinence",
    }
)

_LEGACY_CHANGE_SECTION_RE = re.compile(
    r"(?:Ce qui change|Changement constaté)\s*:\s*"
    r"(.*?)(?=\n\s*\n(?:Pertinence métier|Point de surveillance|Lecture de vigie)\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

_LEGACY_RELEVANCE_SECTION_RE = re.compile(
    r"Pertinence métier\s*:\s*"
    r"(.*?)(?=\n\s*\n(?:Point de surveillance|Lecture de vigie)\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

_LEGACY_SURVEILLANCE_SECTION_RE = re.compile(
    r"(?:Point de surveillance|Lecture de vigie)\s*:\s*"
    r"(.*?)(?=\n\s*\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _clean_narrative_unit(value: Any, *, bank_code: str | None) -> str:
    """Normalise une unité structurée sans tenter d'en déduire la structure."""
    if bank_code:
        normalized = canonicalize_analyst_narrative(
            str(value or ""),
            bank_code=bank_code,
        )
    else:
        normalized = "\n".join(
            sanitize_analyst_french(" ".join(line.split()).strip()) for line in str(value or "").strip().splitlines()
        ).strip()
    normalized = _LEADING_ANALYSIS_LABEL_RE.sub("", normalized, count=1).strip()
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def _structured_business_paragraph(
    values: tuple[Any, ...],
    *,
    summary: str,
    bank_code: str | None,
    limit: int,
) -> str:
    """Assemble les unités métier déjà structurées, sans lire le champ legacy."""
    sentences: list[str] = []
    seen: set[str] = set()
    for value in values:
        sentence = _clean_narrative_unit(value, bank_code=bank_code)
        sentence = _clean_business_relevance_sentence(sentence)
        key = _sentence_comparison_key(sentence)
        if not sentence or not key or key in seen or _duplicates_summary(sentence, summary):
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        seen.add(key)
        sentences.append(sentence)
    return _truncate_at_sentence(" ".join(sentences), limit) if sentences else ""


def _legacy_labeled_section(pattern: re.Pattern[str], value: Any) -> str:
    """Extrait une rubrique uniquement pour la compatibilité des anciens JSON."""
    match = pattern.search(str(value or ""))
    return " ".join(match.group(1).split()).strip() if match else ""


def _legacy_change_candidate(change: dict[str, Any], triage: dict[str, Any]) -> str:
    justification = triage.get("nouvelle_idee_justification")
    labeled_change = _legacy_labeled_section(
        _LEGACY_CHANGE_SECTION_RE,
        justification,
    )
    if labeled_change:
        return labeled_change

    compact_reason = " ".join(str(triage.get("relevance_reason") or "").split())
    if compact_reason:
        return _first_complete_sentence(compact_reason)
    return str(change.get("what_changed") or change.get("change_summary") or "").strip()


def build_analyst_narrative(
    change: dict[str, Any],
    *,
    bank_code: str | None = None,
    candidate_summary: str = "",
    summary_limit: int = 320,
    relevance_limit: int = 720,
) -> AnalystNarrative:
    """Construit les unités canoniques « constat » et « pertinence ».

    Les nouveaux triages sont lus champ par champ. Dès qu'un de leurs champs
    structurés est présent, ``relevance_reason`` n'est jamais découpé ni
    utilisé pour compléter les unités manquantes. Le découpage d'une raison
    compacte demeure uniquement un repli pour les artefacts historiques.
    """
    triage_value = change.get("genai_triage") or {}
    triage = triage_value if isinstance(triage_value, dict) else {}
    has_structured_fields = any(str(triage.get(field) or "").strip() for field in _STRUCTURED_TRIAGE_FIELDS)

    if has_structured_fields:
        factual_candidate = str(
            triage.get("changement_constate")
            or candidate_summary
            or change.get("what_changed")
            or change.get("change_summary")
            or ""
        ).strip()
        source = "structured"
    else:
        factual_candidate = str(candidate_summary or "").strip() or _legacy_change_candidate(change, triage)
        source = "legacy"

    if not factual_candidate:
        factual_candidate = _fallback_summary(
            change,
            subject=bank_subject(bank_code),
        )

    if bank_code:
        changement_constate = build_change_presentation(
            change,
            bank_code=bank_code,
            candidate_summary=factual_candidate,
            limit=summary_limit,
        ).summary
    else:
        changement_constate = _clean_narrative_unit(
            factual_candidate,
            bank_code=None,
        )
        changement_constate = _truncate_at_sentence(
            changement_constate,
            summary_limit,
        )

    if has_structured_fields:
        if bool(triage.get("is_relevant", False)):
            pertinence_metier = _structured_business_paragraph(
                (
                    triage.get("signification_metier"),
                    triage.get("comparaison_interbanques") or triage.get("comparaison_interbancaire"),
                    triage.get("limite_interpretation"),
                ),
                summary=changement_constate,
                bank_code=bank_code,
                limit=relevance_limit,
            )
            motif_non_pertinence = ""
        else:
            pertinence_metier = ""
            motif_non_pertinence = _structured_business_paragraph(
                (triage.get("motif_non_pertinence"),),
                summary=changement_constate,
                bank_code=bank_code,
                limit=relevance_limit,
            )
    else:
        legacy_justification = triage.get("nouvelle_idee_justification")
        labeled_relevance = _legacy_labeled_section(
            _LEGACY_RELEVANCE_SECTION_RE,
            legacy_justification,
        )
        labeled_surveillance = _legacy_labeled_section(
            _LEGACY_SURVEILLANCE_SECTION_RE,
            legacy_justification,
        )
        unstructured_legacy_justification = ""
        if legacy_justification and not labeled_relevance:
            unstructured_legacy_justification = re.sub(
                r"^(?:OUI|NON)\s*[-—:]\s*",
                "",
                " ".join(str(legacy_justification).split()),
                flags=re.IGNORECASE,
            ).strip()
        legacy_relevance = business_relevance_paragraph(
            labeled_relevance,
            labeled_surveillance,
            str(triage.get("relevance_reason") or ""),
            str(triage.get("explanation") or ""),
            str(triage.get("impact_description") or ""),
            summary=changement_constate,
            bank_code=bank_code,
            limit=relevance_limit,
        )
        if not legacy_relevance and unstructured_legacy_justification:
            legacy_relevance = _truncate_at_sentence(
                sanitize_analyst_french(unstructured_legacy_justification),
                relevance_limit,
            )
        if not legacy_relevance:
            exclusion_code = str(triage.get("exclusion_reason") or "").strip()
            if exclusion_code:
                from vigie.comparaison.triage.amf_taxonomy import EXCLUSION_REASONS_DESCRIPTIONS

                legacy_relevance = sanitize_analyst_french(
                    EXCLUSION_REASONS_DESCRIPTIONS.get(
                        exclusion_code,
                        exclusion_code.replace("_", " "),
                    )
                )
        if bool(triage.get("is_relevant", False)):
            pertinence_metier = legacy_relevance
            motif_non_pertinence = ""
        else:
            pertinence_metier = ""
            motif_non_pertinence = legacy_relevance

    return AnalystNarrative(
        changement_constate=changement_constate,
        pertinence_metier=pertinence_metier,
        motif_non_pertinence=motif_non_pertinence,
        source=source,
    )

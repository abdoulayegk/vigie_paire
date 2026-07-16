"""Champs de presentation communs pour la vigie par paires.

Le triage IA conserve ses codes AMF techniques dans les artefacts JSON. Ce
module les traduit en champs stables et lisibles dans les exports analystes :
une categorie principale unique et des etiquettes secondaires multi-label.
"""

from __future__ import annotations

import re
from typing import Any

from vigilance.amf_taxonomy import (
    EXCLUSION_REASONS_DESCRIPTIONS,
    THEMES_AMF_ANALYST_SUBJECTS,
)


_CATEGORY_LABELS: dict[str, str] = {
    "reglementation": "1 — Changements réglementaires",
    "cadre_risques": "2 — Cadre de gestion des risques",
    "gouvernance": "3 — Gouvernance",
    "macro": "4 — Contexte économique",
    "geopolitique": "5 — Géopolitique",
    "ia": "6 — Intelligence artificielle",
    "cyber": "7 — Cyberrisque",
    "donnees_technologie": "8 — Données et technologie",
    "esg": "9 — Risque climatique / ESG",
    "credit": "10 — Risque de crédit",
    "marche": "11 — Risque de marché",
    "liquidite": "12 — Risque de liquidité",
    "operationnel": "13 — Risque opérationnel",
    "modele": "14 — Risque de modèle",
    "conformite": "15 — Conformité juridique et réglementaire",
    "fonds_propres": "16 — Fonds propres",
    "apr": "17 — APR (RWA)",
    "terminologie": "18 — Terminologie",
    "tableaux": "19 — Tableaux",
    "structure": "20 — Sections ajoutées ou supprimées",
    "hors_grille": "À qualifier — sujet émergent hors grille",
}

DIFF_TYPE_LABELS_FR: dict[str, str] = {
    "added": "Ajout",
    "removed": "Suppression",
    "modified": "Modification",
    "renamed": "Renommage",
    "unchanged": "Inchangé",
}


def _normalize(value: Any) -> str:
    """Normalise legerement un texte pour les regles de classement."""
    text = str(value or "").lower()
    replacements = str.maketrans({"é": "e", "è": "e", "ê": "e", "à": "a", "ç": "c", "ô": "o", "î": "i", "ï": "i", "û": "u", "ù": "u"})
    return re.sub(r"\s+", " ", text.translate(replacements)).strip()


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def derive_vigie_category(
    triage: dict[str, Any] | None,
    *,
    text: str = "",
    section: str = "",
    source_kind: str = "text",
) -> str:
    """Retourne une categorie de vigie unique pour un changement.

    La regle privilegie les signaux explicites du texte, puis les themes AMF.
    Cette approche donne une categorie utilisable meme pour les anciens
    artefacts qui ne possedent pas encore de champ de categorie de vigie.
    """
    triage = triage or {}
    themes = {str(theme).upper() for theme in (triage.get("themes_amf") or [])}
    corpus = _normalize(" ".join((text, section)))

    if "SUJET_EMERGENT_HORS_GRILLE" in themes:
        return _CATEGORY_LABELS["hors_grille"]

    # Themes et termes suffisamment specifiques pour etre prioritaires.
    if _has_any(corpus, "actifs ponderes", "actifs pondérés", " apr", "rwa", "risk-weighted"):
        return _CATEGORY_LABELS["apr"]
    if _has_any(corpus, "risque de modele", "risque de modèle", "validation des modeles", "validation des modèles", "e-23"):
        return _CATEGORY_LABELS["modele"]
    if _has_any(corpus, "intelligence artificielle", "ia generative", "ia générative", " ai "):
        return _CATEGORY_LABELS["ia"]
    if _has_any(corpus, "cyber", "ransomware", "hameconnage", "phishing", "attaque par deni"):
        return _CATEGORY_LABELS["cyber"]
    if _has_any(corpus, "donnees", "données", "technologie", "infonuag", "cloud", "fournisseur critique", "tiers critique"):
        return _CATEGORY_LABELS["donnees_technologie"]
    if "ESG_CLIMATIQUE" in themes or _has_any(corpus, "climat", "esg", "b-15", "durabilite", "durabilité", "nzba"):
        return _CATEGORY_LABELS["esg"]
    if "RISQUE_MACRO_GEOPOLITIQUE" in themes or _has_any(corpus, "tarif douanier", "geopolit", "géopolit", "ukraine", "moyen-orient", "sanction", "guerre commerciale"):
        return _CATEGORY_LABELS["geopolitique"]
    if _has_any(corpus, "inflation", "recession", "récession", "chomage", "chômage", "croissance economique", "croissance économique"):
        return _CATEGORY_LABELS["macro"]
    if _has_any(corpus, "risque de credit", "risque de crédit", "hypothec", "hypothéc", "endettement des menages", "endettement des ménages", "garantie"):
        return _CATEGORY_LABELS["credit"]
    if _has_any(corpus, "risque de marche", "risque de marché", "var", "rtipb", "risque de change", "risque actions", "sensibilite aux taux", "sensibilité aux taux"):
        return _CATEGORY_LABELS["marche"]
    if "LIQUIDITE" in themes or _has_any(corpus, "liquidite", "liquidité", " lcr", " nsfr", "financement"):
        return _CATEGORY_LABELS["liquidite"]
    if _has_any(corpus, "risque operationnel", "risque opérationnel", "continuite des affaires", "continuité des affaires", "paiement", "fraude"):
        return _CATEGORY_LABELS["operationnel"]
    if _has_any(corpus, "aml", "fat", "blanchiment", "protection des consommateurs", "vie privee", "vie privée", "confidentialite", "confidentialité", "loi fiscale") or "CONTROLE_CONFORMITE" in themes:
        return _CATEGORY_LABELS["conformite"]
    if _has_any(corpus, "comite", "comité", "conseil d'administration", "audit interne", "trois lignes de defense", "trois lignes de défense") or "GOUVERNANCE_RISQUES" in themes:
        return _CATEGORY_LABELS["gouvernance"]
    if _has_any(corpus, "appetit pour le risque", "appétit pour le risque", "cadre de gestion des risques", "taxonomie des risques", "simulation de crise", "controle interne", "contrôle interne"):
        return _CATEGORY_LABELS["cadre_risques"]
    if _has_any(corpus, "cet1", "tier 1", "tier 2", "tlac", "fonds propres", "capital economique", "capital économique") or themes & {"CAPITAL_REGLEMENTAIRE", "FONDS_PROPRES_REGLEMENTAIRES", "RATIOS_REGLEMENTAIRES"}:
        return _CATEGORY_LABELS["fonds_propres"]
    if source_kind == "table" or _has_any(corpus, "tableau", "colonnes ajoutees", "colonnes ajoutées"):
        return _CATEGORY_LABELS["tableaux"]
    if _has_any(corpus, "renommage", "terminologie", "vocabulaire"):
        return _CATEGORY_LABELS["terminologie"]
    if "STRUCTURE_RAPPORT" in themes or _has_any(corpus, "sous-section", "section ajoutee", "section ajoutée", "section retiree", "section retirée"):
        return _CATEGORY_LABELS["structure"]
    if themes & {"NOUVELLE_MENTION_REGLEMENTAIRE", "EXIGENCES_REGLEMENTAIRES", "MONTANT_REGLEMENTAIRE"} or _has_any(corpus, "bsif", "osfi", "bale", "bâle", "ligne directrice", "reglement", "réglement"):
        return _CATEGORY_LABELS["reglementation"]
    return _CATEGORY_LABELS["cadre_risques"]


def derive_secondary_labels(triage: dict[str, Any] | None) -> str:
    """Traduit les themes AMF multi-label en etiquettes analystes."""
    triage = triage or {}
    labels: list[str] = []
    for theme in triage.get("themes_amf") or []:
        label = THEMES_AMF_ANALYST_SUBJECTS.get(str(theme).upper())
        if label and label not in labels:
            labels.append(label)
    return " · ".join(labels)


def _truncate_at_sentence(value: str, limit: int) -> str:
    """Coupe à la dernière phrase complète avant ``limit``, sans couper au milieu."""
    if len(value) <= limit:
        return value
    window = value[:limit]
    # Prefer the last sentence-ending punctuation inside the window.
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = window.rfind(sep)
        if idx >= max(40, limit // 4):
            return window[: idx + 1].rstrip()
    # Fallback: cut at last space then ensure we do not leave a hanging ellipsis mid-word.
    space_idx = window.rfind(" ")
    if space_idx >= max(40, limit // 4):
        return window[:space_idx].rstrip(" ,;:") + "."
    return window.rstrip(" ,;:") + "."


def summarize_change(
    change: dict[str, Any],
    *,
    previous_text: str = "",
    current_text: str = "",
    limit: int = 300,
) -> str:
    """Produit une phrase factuelle courte, indépendante du triage AMF."""
    from vigilance.i18n.fr import sanitize_analyst_french

    summary = str(
        change.get("what_changed") or change.get("change_summary") or ""
    ).strip()
    if summary:
        value = sanitize_analyst_french(re.sub(r"\s+", " ", summary))
        if len(value) > limit:
            value = _truncate_at_sentence(value, limit)
        return value

    diff_type = str(change.get("diff_type") or change.get("change_type") or "").lower()
    if diff_type in {"added", "table_added"}:
        value = current_text
        prefix = "Ajout : "
    elif diff_type in {"removed", "table_removed"}:
        value = previous_text
        prefix = "Suppression : "
    elif diff_type == "renamed":
        value = f"{previous_text} → {current_text}".strip(" →")
        prefix = "Renommage : "
    else:
        value = current_text or previous_text
        prefix = "Modification : "

    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(value) > limit:
        value = _truncate_at_sentence(value, limit)
    return sanitize_analyst_french(prefix + value if value else prefix.rstrip(" :"))


def subsection_label(change: dict[str, Any]) -> str:
    """Retourne le libellé de sous-section utilisable par Dash et Excel."""
    heading = str(change.get("subsection_heading") or "").strip()
    if heading and heading not in {"__intro__", "full"}:
        return heading

    change_id = str(change.get("change_id") or "")
    section_key = str(change.get("section_key") or "")
    prefix = f"{section_key}_" if section_key else ""
    if prefix and change_id.startswith(prefix):
        slug = re.sub(r"_change_\d+$", "", change_id[len(prefix) :])
        if slug and slug != "full":
            return slug.replace("_", " ").strip()
    return ""


def relevance_reason_for_display(change: dict[str, Any]) -> str:
    """Lit la raison compacte, avec repli compatible sur les anciens artefacts."""
    from vigilance.i18n.fr import sanitize_analyst_french

    triage = change.get("genai_triage") or {}
    compact_reason = " ".join(str(triage.get("relevance_reason") or "").split())
    if compact_reason:
        return sanitize_analyst_french(compact_reason)

    legacy_justification = str(
        triage.get("nouvelle_idee_justification") or ""
    ).strip()
    match = re.search(
        r"Pertinence métier\s*:\s*(.*?)(?=\n\s*\n(?:Point de surveillance|Lecture de vigie)\s*:|$)",
        legacy_justification,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return sanitize_analyst_french(" ".join(match.group(1).split()))

    explanation = " ".join(str(triage.get("explanation") or "").split())
    if explanation:
        return sanitize_analyst_french(explanation)
    if legacy_justification:
        cleaned_legacy = re.sub(
            r"^(OUI|NON)\s*[-—:]\s*",
            "",
            " ".join(legacy_justification.split()),
            flags=re.IGNORECASE,
        ).strip()
        if cleaned_legacy:
            return sanitize_analyst_french(cleaned_legacy)

    exclusion_code = str(triage.get("exclusion_reason") or "").strip()
    if exclusion_code:
        return EXCLUSION_REASONS_DESCRIPTIONS.get(exclusion_code, exclusion_code)
    return "La pertinence n’a pas encore été qualifiée par l’analyse automatisée."


_META_SUMMARY_RE = re.compile(
    r"^Les deux (?:fragments|passages)\b",
    flags=re.IGNORECASE,
)
_CE_QUI_CHANGE_RE = re.compile(
    r"Ce qui change\s*:\s*(.*?)(?=\n\s*\n(?:Pertinence métier|Point de surveillance|Lecture de vigie)\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def what_changed_for_display(change: dict[str, Any], *, limit: int = 300) -> str:
    """Texte « Ce qui change » pour analyste : phrase complète, sans meta GPT."""
    from vigilance.amf_taxonomy import _compact_complete_sentence_parts
    from vigilance.i18n.fr import sanitize_analyst_french

    triage = change.get("genai_triage") or {}
    # Prefer compact relevance_reason only (not Pertinence métier fallback).
    compact_reason = " ".join(str(triage.get("relevance_reason") or "").split())
    if compact_reason:
        parts = _compact_complete_sentence_parts(compact_reason)
        candidate = parts[0] if parts else compact_reason
        value = sanitize_analyst_french(candidate)
        if value:
            return value if len(value) <= limit else _truncate_at_sentence(value, limit)

    justification = str(triage.get("nouvelle_idee_justification") or "")
    match = _CE_QUI_CHANGE_RE.search(justification)
    if match:
        value = sanitize_analyst_french(" ".join(match.group(1).split()))
        if value:
            return value if len(value) <= limit else _truncate_at_sentence(value, limit)

    previous_text = str(change.get("source_text_t1") or "")
    current_text = str(change.get("source_text_t2") or "")
    summary = str(change.get("change_summary") or "").strip()
    if summary and not _META_SUMMARY_RE.match(summary):
        return summarize_change(
            change,
            previous_text=previous_text,
            current_text=current_text,
            limit=limit,
        )
    return summarize_change(
        change,
        previous_text=previous_text,
        current_text=current_text,
        limit=limit,
    )


def build_text_vigie_display_row(
    change: dict[str, Any],
    *,
    section_title: str,
) -> dict[str, Any]:
    """Construit les huit champs analyste communs à Dash et Excel."""
    triage = change.get("genai_triage") or {}
    previous_text = str(change.get("source_text_t1") or "")
    current_text = str(change.get("source_text_t2") or "")
    what_changed = what_changed_for_display(change)
    category = derive_vigie_category(
        triage,
        text=" ".join((what_changed, previous_text, current_text)),
        section=section_title,
    )
    diff_type = str(change.get("diff_type") or "")
    nouvelle_idee = bool(triage.get("nouvelle_idee", False))
    return {
        "category": category,
        "secondary_labels": derive_secondary_labels(triage),
        "section": section_title,
        "subsection": subsection_label(change),
        "change_type": DIFF_TYPE_LABELS_FR.get(
            diff_type.lower(),
            diff_type.capitalize(),
        ),
        "what_changed": what_changed,
        "nouvelle_idee": nouvelle_idee,
        "nouvelle_idee_label": "Oui" if nouvelle_idee else "Non",
        "relevance_reason": relevance_reason_for_display(change),
    }

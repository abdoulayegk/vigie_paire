"""Helpers de justification pour l'analyse textuelle."""

from __future__ import annotations

import re
from typing import Any

from vigilance.analyst_change_presentation import build_analyst_narrative

_REQUIRED_JUSTIFICATION_MARKERS = (
    "Nouvel élément à surveiller :",
    "Sujet détecté :",
    "Ce qui change :",
    "Pertinence métier :",
    "Point de surveillance :",
)


def _clean(value: Any) -> str:
    """Retourne une chaine nettoyee sans retours parasites."""
    return " ".join(str(value or "").strip().split())


def is_structured_text_triage_justification(value: Any) -> bool:
    """Vrai si la justification contient toutes les rubriques attendues."""
    text = str(value or "")
    if "Lecture de vigie :" in text and "Point de surveillance :" not in text:
        text = text.replace("Lecture de vigie :", "Point de surveillance :")
    return all(marker in text for marker in _REQUIRED_JUSTIFICATION_MARKERS)


def synthesize_triage_justification_from_payload(triage: dict[str, Any]) -> str:
    """Reconstruit une justification AMF structurée à partir des champs de triage."""
    from vigilance.amf_taxonomy import EXCLUSION_REASONS_DESCRIPTIONS, THEMES_AMF_ANALYST_SUBJECTS

    nouvelle_idee = bool(triage.get("nouvelle_idee", False))
    is_relevant = bool(triage.get("is_relevant", False))
    prefix = "OUI" if nouvelle_idee else "NON"
    yes_no = "Oui" if nouvelle_idee else "Non"

    subject_parts: list[str] = []
    for theme in triage.get("themes_amf") or []:
        label = THEMES_AMF_ANALYST_SUBJECTS.get(str(theme))
        if label and label not in subject_parts:
            subject_parts.append(label)
    subject = ", ".join(subject_parts) or "Changement textuel détecté dans la divulgation"

    explanation = _clean(triage.get("explanation"))
    exclusion_code = str(triage.get("exclusion_reason") or "").strip()
    exclusion_label = EXCLUSION_REASONS_DESCRIPTIONS.get(exclusion_code, exclusion_code)

    if is_relevant:
        ce_qui_change = (
            explanation
            or "Le rapport courant présente une information nouvelle ou reformulée par rapport "
            "au rapport précédent sur ce sujet."
        )
        pertinence = (
            explanation
            or (
                "Ce changement met l'accent sur un sujet pertinent pour la vigie, car il peut "
                "modifier la lecture réglementaire, la comparabilité des divulgations entre pairs "
                "ou le niveau de détail exposé par la banque sur ce thème."
            )
        )
    else:
        ce_qui_change = (
            explanation
            or "Le texte diffère entre T1 et T2, mais le changement ne constitue pas un signal "
            "métier prioritaire pour la vigie."
        )
        pertinence = (
            explanation
            or (
                "Ce changement n'est pas retenu comme nouvelle idée, car il relève principalement de "
                f"{exclusion_label or 'une variation ou reformulation sans impact réglementaire direct'}."
            )
        )

    surveillance_subject = subject.split(",")[0].strip() or "Ce point"
    surveillance = (
        f"{surveillance_subject} — Ce point résume l'élément à conserver dans la lecture de vigie "
        "et son importance relative pour la surveillance des divulgations de la banque."
    )

    return (
        f"{prefix} — Nouvel élément à surveiller : {yes_no}.\n\n"
        f"Sujet détecté : {subject}.\n\n"
        f"Ce qui change : {_sentence(ce_qui_change)}\n\n"
        f"Pertinence métier : {_sentence(pertinence)}\n\n"
        f"Point de surveillance : {_sentence(surveillance)}"
    )


def _sentence(value: str) -> str:
    """Termine une phrase sans doubler la ponctuation."""
    value = value.strip()
    if not value:
        return ""
    return value if value[-1] in ".!?" else f"{value}."


def _strip_decision_prefix(value: str) -> str:
    """Retire le préfixe OUI/NON d'une justification legacy."""
    value = value.strip()
    return re.sub(r"^(OUI|NON)\s*[-—:]\s*", "", value, flags=re.IGNORECASE).strip()


def _combined_text(change: dict[str, Any]) -> str:
    """Concatène les champs textuels du changement (résumé + extraits T1/T2) pour analyse."""
    parts = [
        change.get("change_summary"),
        change.get("source_text_t1"),
        change.get("source_text_t2"),
        change.get("semantic_text_t1"),
        change.get("semantic_text_t2"),
    ]
    return " ".join(_clean(part).lower() for part in parts if part)


def _is_climate_b15_change(change: dict[str, Any]) -> bool:
    """Détecte si le changement concerne la ligne directrice B-15 BSIF / risque climatique."""
    text = _combined_text(change)
    return any(marker in text for marker in ("b-15", "climat", "ges", "durabilit"))


def _infer_subject(change: dict[str, Any], triage: dict[str, Any]) -> str:
    """Infere une ligne Sujet detecte pour les anciens triages incomplets."""
    text = _combined_text(change)
    diff_type = str(change.get("diff_type") or "").lower()
    category = str(triage.get("category") or "").upper()
    risk_type = _clean(triage.get("risk_type"))
    reference = _clean(triage.get("reference_reglementaire"))

    topics: list[str] = []
    if any(marker in text for marker in ("b-15", "climat", "ges", "durabilit")):
        topics.extend(["Risque climatique", "ESG"])
    if reference:
        topics.append(reference)
    if category == "REGLEMENTAIRE" or "bsif" in text:
        topics.append("nouvelle mention réglementaire")
    if risk_type and risk_type.lower() not in {"conformite", "conformité"}:
        topics.append(risk_type)
    elif risk_type:
        topics.append("conformité")

    if diff_type == "added":
        topics.append("information ajoutée")
    elif diff_type == "removed":
        topics.append("information retirée")
    elif diff_type == "modified":
        topics.append("texte modifié")

    if not topics and category:
        topics.append(category.lower().replace("_", " "))

    deduped = list(dict.fromkeys(topic for topic in topics if topic))
    return ", ".join(deduped) or "Changement textuel detecte"


def _first_complete_sentence(text: str) -> str:
    """Retourne la première phrase complète ponctuée, ou le texte nettoyé."""
    from vigilance.amf_taxonomy import _compact_complete_sentence_parts

    normalized = _clean(text)
    if not normalized:
        return ""
    parts = _compact_complete_sentence_parts(normalized)
    if parts:
        return parts[0]
    return normalized


def _change_sentence(change: dict[str, Any]) -> str:
    """Formule une phrase « Ce qui change » à partir des diff_type et extraits sources."""
    from vigilance.i18n.fr import sanitize_analyst_french

    diff_type = str(change.get("diff_type") or "").lower()
    summary = _clean(change.get("change_summary"))
    source_t1 = _clean(change.get("source_text_t1") or change.get("semantic_text_t1"))
    source_t2 = _clean(change.get("source_text_t2") or change.get("semantic_text_t2"))

    if diff_type == "added" and source_t2:
        return _sentence(
            sanitize_analyst_french(
                "Le rapport courant ajoute l'information suivante, absente "
                f"du rapport précédent : {source_t2}"
            )
        )
    if diff_type == "removed" and source_t1:
        return _sentence(
            sanitize_analyst_french(
                "Le rapport courant retire l'information suivante, présente "
                f"au rapport précédent : {source_t1}"
            )
        )
    if diff_type == "modified" and source_t1 and source_t2:
        return _sentence(
            sanitize_analyst_french(
                "Le passage est modifié entre les deux versions. "
                f"Le rapport précédent indiquait : {source_t1} "
                f"Le rapport courant indique : {source_t2}"
            )
        )
    if summary:
        return _sentence(sanitize_analyst_french(summary))
    return (
        "Le changement textuel modifie l'information communiquée entre "
        "le rapport précédent et le rapport courant."
    )


def _fallback_pertinence(change: dict[str, Any], triage: dict[str, Any]) -> str:
    """Construit une pertinence metier exploitable pour les triages legacy."""
    if _is_climate_b15_change(change):
        return _climate_b15_pertinence()

    legacy_justification = _clean(
        _strip_decision_prefix(str(triage.get("nouvelle_idee_justification") or ""))
    )
    explanation = _clean(triage.get("explanation"))
    impact_description = _clean(triage.get("impact_description"))
    return (
        legacy_justification
        or explanation
        or impact_description
        or "La pertinence métier doit être appréciée à partir du changement détecté."
    )


def _climate_b15_pertinence() -> str:
    """Retourne la pertinence métier standard pour les changements climatiques B-15."""
    return (
        "Ce changement met l'accent sur l'évolution du cadre réglementaire "
        "applicable aux divulgations climatiques des institutions financières. "
        "La mise à jour de la ligne directrice B-15 par le BSIF, notamment "
        "le report des exigences liées aux émissions de GES de portée 3, "
        "peut influencer la manière dont les banques planifient leur conformité, "
        "structurent leurs contrôles ESG et communiquent leurs risques climatiques. "
        "Ce point est important à suivre, car il permet d'évaluer l'évolution "
        "des attentes prudentielles, la comparabilité des pratiques de divulgation "
        "entre pairs et le niveau de préparation des banques face aux exigences "
        "climatiques."
    )


def _replace_pertinence(justification: str, pertinence: str) -> str:
    """Remplace la rubrique « Pertinence métier » dans une justification structurée."""
    marker = "Pertinence métier :"
    start = justification.find(marker)
    end = _surveillance_marker_index(justification, start)
    if start == -1 or end == -1:
        return justification
    return (
        justification[:start]
        + f"{marker} {_sentence(pertinence)}"
        + justification[end:]
    )


def _replace_surveillance_point(justification: str, point: str) -> str:
    """Remplace la rubrique « Point de surveillance » dans une justification structurée."""
    start = _surveillance_marker_index(justification, 0)
    if start == -1:
        return justification
    marker = "\n\nPoint de surveillance :"
    next_section = justification.find("\n\n", start + len(marker))
    end = len(justification) if next_section == -1 else next_section
    return justification[:start] + f"{marker} {_sentence(point)}" + justification[end:]


def _surveillance_marker_index(justification: str, start: int = 0) -> int:
    """Retourne l'index du premier marqueur de surveillance (canonique ou legacy)."""
    indexes = [
        idx
        for marker in ("\n\nPoint de surveillance :", "\n\nLecture de vigie :")
        if (idx := justification.find(marker, start)) != -1
    ]
    return min(indexes) if indexes else -1


def _normalize_surveillance_marker(justification: str) -> str:
    """Remplace le marqueur legacy « Lecture de vigie : » par le marqueur canonique."""
    return justification.replace("Lecture de vigie :", "Point de surveillance :")


def _surveillance_point(
    change: dict[str, Any],
    triage: dict[str, Any],
    subject: str,
) -> str:
    """Construit la phrase « Point de surveillance » à partir du sujet et de l'impact."""
    impact_description = _clean(triage.get("impact_description"))
    if _is_climate_b15_change(change):
        return (
            "Risque climatique / ESG — Le changement indique que la banque tient "
            "compte de la mise à jour de la ligne directrice B-15 du BSIF et du "
            "report des exigences liées aux émissions de GES de portée 3. Ce point "
            "permet de suivre l'adaptation des banques aux attentes prudentielles "
            "climatiques, leur niveau de préparation ESG et la comparabilité de "
            "leurs pratiques de divulgation entre pairs."
        )
    if impact_description:
        label = subject.split(",")[0].strip() or "Point de surveillance"
        return _sentence(f"{label} — {impact_description}")
    return (
        f"{subject} — Ce point résume le changement à suivre, son lien avec "
        "les attentes prudentielles et son effet possible sur la comparabilité "
        "des pratiques entre institutions."
    )


def build_compact_triage_justification(
    change: dict[str, Any],
    triage: dict[str, Any],
) -> str:
    """Construit localement le format historique depuis le triage compact."""
    from vigilance.amf_taxonomy import THEMES_AMF_ANALYST_SUBJECTS
    from vigilance.i18n.fr import sanitize_analyst_french

    nouvelle_idee = bool(triage.get("nouvelle_idee", False))
    prefix = "OUI" if nouvelle_idee else "NON"
    yes_no = "Oui" if nouvelle_idee else "Non"
    subjects = [
        THEMES_AMF_ANALYST_SUBJECTS.get(str(theme), str(theme))
        for theme in triage.get("themes_amf") or []
    ]
    subject = ", ".join(dict.fromkeys(subjects)) or "Changement non retenu"
    change_with_triage = {**change, "genai_triage": triage}
    bank_code = str(
        change.get("bank_code")
        or triage.get("bank_code")
        or ""
    ).strip()
    narrative = build_analyst_narrative(
        change_with_triage,
        bank_code=bank_code or None,
    )
    what_changed = narrative.changement_constate or _change_sentence(change)
    relevance_reason = narrative.business_relevance
    if not relevance_reason:
        relevance_reason = (
            "La justification métier structurée n’a pas été renseignée."
            if narrative.source == "structured"
            else _fallback_pertinence(change, triage)
        )
    what_changed = sanitize_analyst_french(what_changed)
    relevance_reason = sanitize_analyst_french(relevance_reason)
    surveillance = (
        f"{subject.split(',')[0]} — La décision reste disponible pour validation "
        "avec les textes sources et les différences exactes."
    )
    return (
        f"{prefix} — Nouvel élément à surveiller : {yes_no}.\n\n"
        f"Sujet détecté : {_sentence(subject)}\n\n"
        f"Ce qui change : {_sentence(what_changed)}\n\n"
        f"Pertinence métier : {_sentence(relevance_reason)}\n\n"
        f"Point de surveillance : {_sentence(surveillance)}"
    )


def build_text_triage_justification(change: dict[str, Any]) -> str:
    """Retourne la justification AMF v2 ou un fallback structure pour Dash/Excel.

    Les nouveaux artefacts doivent contenir ``genai_triage.nouvelle_idee_justification``.
    Les anciens artefacts texte produits par ``gpt4o_triage`` peuvent avoir
    ``nouvelle_idee=True`` sans ce champ. Dans ce cas, on reformate les champs
    disponibles en note d'analyste structuree pour eviter un affichage vide.
    """
    triage = change.get("genai_triage") or {}
    if not isinstance(triage, dict) or not triage:
        return ""

    if str(triage.get("relevance_reason") or "").strip() or any(
        str(triage.get(field) or "").strip()
        for field in (
            "changement_constate",
            "signification_metier",
            "comparaison_interbanques",
            "comparaison_interbancaire",
            "limite_interpretation",
            "motif_non_pertinence",
        )
    ):
        return build_compact_triage_justification(change, triage)

    explicit = str(triage.get("nouvelle_idee_justification") or "").strip()
    if explicit:
        explicit = _normalize_surveillance_marker(explicit)
        if is_structured_text_triage_justification(explicit):
            if _is_climate_b15_change(change):
                subject = _infer_subject(change, triage)
                explicit = _replace_pertinence(explicit, _climate_b15_pertinence())
                return _replace_surveillance_point(
                    explicit,
                    _surveillance_point(change, triage, subject),
                )
            return explicit

    explanation = _clean(triage.get("explanation"))
    impact_description = _clean(triage.get("impact_description"))
    if not explicit and not explanation and not impact_description:
        return ""

    nouvelle_idee = bool(triage.get("nouvelle_idee", False))
    prefix = "OUI" if nouvelle_idee else "NON"
    yes_no = "Oui" if nouvelle_idee else "Non"
    subject = _infer_subject(change, triage)
    change_sentence = _change_sentence(change)

    pertinence = _fallback_pertinence(change, triage)
    lecture = _surveillance_point(change, triage, subject)

    return (
        f"{prefix} — Nouvel élément à surveiller : {yes_no}.\n\n"
        f"Sujet détecté : {subject}.\n\n"
        f"Ce qui change : {change_sentence}\n\n"
        f"Pertinence métier : {_sentence(pertinence)}\n\n"
        f"Point de surveillance : {lecture}"
    )

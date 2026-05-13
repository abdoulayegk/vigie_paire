"""Helpers de justification pour l'analyse textuelle."""

from __future__ import annotations

from typing import Any


def _clean(value: Any) -> str:
    """Retourne une chaine nettoyee sans retours parasites."""
    return " ".join(str(value or "").strip().split())


def _sentence(value: str) -> str:
    """Termine une phrase sans doubler la ponctuation."""
    value = value.strip()
    if not value:
        return ""
    return value if value[-1] in ".!?" else f"{value}."


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


def _change_sentence(change: dict[str, Any]) -> str:
    """Formule une phrase « Ce qui change » à partir des diff_type et extraits T1/T2."""
    diff_type = str(change.get("diff_type") or "").lower()
    summary = _clean(change.get("change_summary"))
    source_t1 = _clean(change.get("source_text_t1") or change.get("semantic_text_t1"))
    source_t2 = _clean(change.get("source_text_t2") or change.get("semantic_text_t2"))

    if diff_type == "added" and source_t2:
        return _sentence(
            f"Le T2 ajoute l'information suivante, absente du T1 : {source_t2}"
        )
    if diff_type == "removed" and source_t1:
        return _sentence(
            f"Le T2 retire l'information suivante, présente au T1 : {source_t1}"
        )
    if diff_type == "modified" and source_t1 and source_t2:
        return _sentence(
            "Le passage est modifie entre T1 et T2. "
            f"T1 indiquait : {source_t1} T2 indique : {source_t2}"
        )
    if summary:
        return _sentence(summary)
    return "Le changement textuel modifie l'information communiquee entre T1 et T2."


def _fallback_pertinence(change: dict[str, Any], triage: dict[str, Any]) -> str:
    """Construit une pertinence metier exploitable pour les triages legacy."""
    if _is_climate_b15_change(change):
        return _climate_b15_pertinence()

    explanation = _clean(triage.get("explanation"))
    impact_description = _clean(triage.get("impact_description"))
    return (
        explanation
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

    explicit = str(triage.get("nouvelle_idee_justification") or "").strip()
    if explicit:
        explicit = _normalize_surveillance_marker(explicit)
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
    if not explanation and not impact_description:
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

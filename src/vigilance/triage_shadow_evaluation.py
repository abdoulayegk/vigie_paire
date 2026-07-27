"""Evaluation parallele des classifications de triage historique et candidate.

Ce module reste independant du moteur de triage. Il evalue deux verdicts sur
le meme corpus de reference et rend visibles les erreurs de sous-classification
les plus couteuses pour la vigie.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMPACT_LEVELS: tuple[str, ...] = ("MINEUR", "MODERE", "MAJEUR")
REVIEW_LEVEL = "A_CONFIRMER"
PREDICTION_LEVELS: tuple[str, ...] = (*IMPACT_LEVELS, REVIEW_LEVEL)
SIX_BANK_COVERAGE: tuple[str, ...] = ("BMO", "BNC", "BNS", "CIBC", "RBC", "TD")
DEFAULT_MIN_CASES_PER_BANK = 20
_IMPACT_SCORE = {level: index for index, level in enumerate(IMPACT_LEVELS)}
_MISSING_DIMENSION = "NON_RENSEIGNE"
_MISSING_THEME = "SANS_THEME"
_ROUND_DIGITS = 6


class ShadowTriageEvaluationError(ValueError):
    """Signale un corpus de validation parallele invalide."""


@dataclass(frozen=True, slots=True)
class ShadowAcceptanceThresholds:
    """Seuils de qualite du triage candidat.

    Les taux de sous-classification sont plafonnes, tandis que le rappel, la
    precision et l'accord pondere doivent atteindre un minimum.
    """

    min_non_minor_recall: float = 0.95
    min_major_recall: float = 0.90
    max_major_to_minor_rate: float = 0.02
    max_moderate_to_minor_rate: float = 0.05
    min_major_precision: float = 0.75
    min_automatic_coverage: float = 0.80
    min_weighted_agreement: float = 0.70

    def __post_init__(self) -> None:
        """Valide les domaines numeriques des seuils."""
        unit_interval = {
            "min_non_minor_recall": self.min_non_minor_recall,
            "min_major_recall": self.min_major_recall,
            "max_major_to_minor_rate": self.max_major_to_minor_rate,
            "max_moderate_to_minor_rate": self.max_moderate_to_minor_rate,
            "min_major_precision": self.min_major_precision,
            "min_automatic_coverage": self.min_automatic_coverage,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ShadowTriageEvaluationError(
                    f"Le seuil {name} doit etre compris entre 0 et 1."
                )
        if not -1.0 <= self.min_weighted_agreement <= 1.0:
            raise ShadowTriageEvaluationError(
                "Le seuil min_weighted_agreement doit etre compris entre -1 et 1."
            )

    def as_dict(self) -> dict[str, float]:
        """Retourne une representation JSON stable des seuils."""
        return {
            "min_non_minor_recall": self.min_non_minor_recall,
            "min_major_recall": self.min_major_recall,
            "max_major_to_minor_rate": self.max_major_to_minor_rate,
            "max_moderate_to_minor_rate": self.max_moderate_to_minor_rate,
            "min_major_precision": self.min_major_precision,
            "min_automatic_coverage": self.min_automatic_coverage,
            "min_weighted_agreement": self.min_weighted_agreement,
        }


DEFAULT_SHADOW_THRESHOLDS = ShadowAcceptanceThresholds()


@dataclass(frozen=True, slots=True)
class ShadowTriageCase:
    """Cas de reference avec verdict historique et verdict candidat."""

    change_id: str
    reference_impact: str
    legacy_impact: str
    candidate_impact: str
    bank: str = _MISSING_DIMENSION
    themes_amf: tuple[str, ...] = ()
    change_nature: str = _MISSING_DIMENSION
    reference_record_fingerprint: str = ""

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> "ShadowTriageCase":
        """Construit un cas valide depuis un enregistrement JSON."""
        change_id = _first_non_empty(
            record,
            ("change_id", "id", "uid", "change_index"),
            default=f"case-{index}",
        )
        reference = _read_impact(
            record,
            direct_keys=("reference_impact", "expected_impact", "gold_impact"),
            nested_keys=("reference", "expected", "gold"),
            field_name="reference_impact",
            change_id=change_id,
        )
        legacy = _read_impact(
            record,
            direct_keys=("legacy_impact", "old_impact"),
            nested_keys=("legacy_triage", "old_triage", "legacy", "old"),
            field_name="legacy_impact",
            change_id=change_id,
        )
        candidate = _read_candidate_impact(
            record,
            direct_keys=("candidate_impact", "new_impact"),
            nested_keys=("candidate_triage", "new_triage", "candidate", "new"),
            field_name="candidate_impact",
            change_id=change_id,
        )
        bank = _normalize_dimension(
            _first_non_empty(
                record,
                ("bank", "bank_code", "banque"),
                default=_MISSING_DIMENSION,
            )
        )
        nature = _normalize_dimension(
            _first_non_empty(
                record,
                ("change_nature", "nature", "nature_changement"),
                default=_MISSING_DIMENSION,
            )
        )
        themes = _normalize_themes(record.get("themes_amf", record.get("themes")))
        return cls(
            change_id=change_id,
            reference_impact=reference,
            legacy_impact=legacy,
            candidate_impact=candidate,
            bank=bank,
            themes_amf=themes,
            change_nature=nature,
            reference_record_fingerprint=_reference_record_fingerprint(record),
        )


def _first_non_empty(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: str,
) -> str:
    """Retourne la premiere valeur textuelle non vide parmi plusieurs cles."""
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _read_impact(
    record: Mapping[str, Any],
    *,
    direct_keys: Sequence[str],
    nested_keys: Sequence[str],
    field_name: str,
    change_id: str,
) -> str:
    """Extrait puis normalise un niveau d'impact direct ou imbrique."""
    raw: Any = None
    for key in direct_keys:
        if record.get(key) is not None:
            raw = record.get(key)
            break
    if raw is None:
        for key in nested_keys:
            nested = record.get(key)
            if isinstance(nested, Mapping) and nested.get("impact_level") is not None:
                raw = nested.get("impact_level")
                break
    if raw is None:
        raise ShadowTriageEvaluationError(
            f"Le cas {change_id!r} ne contient pas le champ {field_name!r}."
        )
    try:
        return normalize_impact_level(raw)
    except ShadowTriageEvaluationError as exc:
        raise ShadowTriageEvaluationError(
            f"Le cas {change_id!r} contient un {field_name} invalide: {raw!r}."
        ) from exc


def _read_candidate_impact(
    record: Mapping[str, Any],
    *,
    direct_keys: Sequence[str],
    nested_keys: Sequence[str],
    field_name: str,
    change_id: str,
) -> str:
    """Lit le verdict candidat sans convertir une abstention en MINEUR."""
    nested_materialities: list[str] = []
    nested_fallbacks: list[str] = []
    for key in nested_keys:
        nested = record.get(key)
        if not isinstance(nested, Mapping):
            continue
        materiality = nested.get("materiality_level")
        normalized_materiality: str | None = None
        if materiality is not None and str(materiality).strip():
            try:
                normalized_materiality = normalize_impact_level(materiality)
            except ShadowTriageEvaluationError as exc:
                raise ShadowTriageEvaluationError(
                    f"Le cas {change_id!r} contient un {field_name} invalide: "
                    f"{materiality!r}."
                ) from exc
            nested_materialities.append(normalized_materiality)

        decision_status = _normalize_dimension(nested.get("decision_status"))
        review_required = _normalize_optional_bool(
            nested.get("review_required"),
            field_name="review_required",
            change_id=change_id,
        )
        if (
            review_required
            or decision_status in {"A_CONFIRMER", "PROVISOIRE"}
        ):
            return REVIEW_LEVEL

        if nested.get("impact_level") is not None:
            raw = nested.get("impact_level")
            try:
                nested_fallbacks.append(normalize_candidate_level(raw))
            except ShadowTriageEvaluationError as exc:
                raise ShadowTriageEvaluationError(
                    f"Le cas {change_id!r} contient un {field_name} invalide: "
                    f"{raw!r}."
                ) from exc

    for key in direct_keys:
        if record.get(key) is None:
            continue
        raw = record.get(key)
        try:
            return normalize_candidate_level(raw)
        except ShadowTriageEvaluationError as exc:
            raise ShadowTriageEvaluationError(
                f"Le cas {change_id!r} contient un {field_name} invalide: "
                f"{raw!r}."
            ) from exc

    if nested_materialities:
        return nested_materialities[0]
    if nested_fallbacks:
        return nested_fallbacks[0]

    raise ShadowTriageEvaluationError(
        f"Le cas {change_id!r} ne contient pas le champ {field_name!r}."
    )


def _normalize_optional_bool(
    value: Any,
    *,
    field_name: str,
    change_id: str,
) -> bool:
    """Normalise un booleen JSON sans traiter la chaine ``false`` comme vraie."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "vrai", "yes", "oui", "1"}:
        return True
    if normalized in {"false", "faux", "no", "non", "0", ""}:
        return False
    raise ShadowTriageEvaluationError(
        f"Le cas {change_id!r} contient un {field_name} invalide: {value!r}."
    )


def _normalize_dimension(value: Any) -> str:
    """Normalise une valeur de ventilation sans supprimer ses accents."""
    normalized = re.sub(r"\s+", " ", str(value or "").strip()).upper()
    return normalized or _MISSING_DIMENSION


def _normalize_themes(value: Any) -> tuple[str, ...]:
    """Normalise une liste de themes et supprime ses doublons."""
    if value is None:
        return ()
    raw_values: Iterable[Any]
    if isinstance(value, str):
        raw_values = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        raw_values = value
    else:
        raise ShadowTriageEvaluationError(
            "themes_amf doit etre une chaine ou une collection de chaines."
        )
    themes: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        theme = _normalize_dimension(raw)
        if theme == _MISSING_DIMENSION or theme in seen:
            continue
        seen.add(theme)
        themes.append(theme)
    return tuple(themes)


def normalize_impact_level(value: Any) -> str:
    """Normalise un niveau d'impact francais vers MINEUR, MODERE ou MAJEUR."""
    text = unicodedata.normalize("NFKD", str(value or "").strip().upper())
    normalized = "".join(char for char in text if not unicodedata.combining(char))
    normalized = re.sub(r"[\s_-]+", "", normalized)
    aliases = {
        "MINEUR": "MINEUR",
        "MINEURE": "MINEUR",
        "MINOR": "MINEUR",
        "MODERE": "MODERE",
        "MODEREE": "MODERE",
        "MODERATE": "MODERE",
        "MAJEUR": "MAJEUR",
        "MAJEURE": "MAJEUR",
        "MAJOR": "MAJEUR",
    }
    if normalized not in aliases:
        raise ShadowTriageEvaluationError(
            f"Niveau d'impact inconnu {value!r}; valeurs permises: {', '.join(IMPACT_LEVELS)}."
        )
    return aliases[normalized]


def normalize_candidate_level(value: Any) -> str:
    """Normalise un verdict candidat, y compris l'abstention de revue."""
    text = unicodedata.normalize("NFKD", str(value or "").strip().upper())
    normalized = "".join(
        char for char in text if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[\s_-]+", "", normalized)
    if normalized in {
        "ACONFIRMER",
        "NIVEAUACONFIRMER",
        "REVIEW",
        "REVIEWREQUIRED",
        "ABSTENTION",
    }:
        return REVIEW_LEVEL
    return normalize_impact_level(value)


def coerce_shadow_cases(
    records: Sequence[ShadowTriageCase | Mapping[str, Any]],
) -> list[ShadowTriageCase]:
    """Valide et convertit une sequence d'enregistrements de reference."""
    cases: list[ShadowTriageCase] = []
    for index, record in enumerate(records):
        if isinstance(record, ShadowTriageCase):
            cases.append(record)
        elif isinstance(record, Mapping):
            cases.append(ShadowTriageCase.from_mapping(record, index=index))
        else:
            raise ShadowTriageEvaluationError(
                f"L'entree {index} doit etre un objet JSON, pas {type(record).__name__}."
            )
    if not cases:
        raise ShadowTriageEvaluationError(
            "Le corpus de validation parallele ne contient aucun cas."
        )
    seen_change_ids: set[str] = set()
    duplicate_change_ids: set[str] = set()
    for case in cases:
        if case.change_id in seen_change_ids:
            duplicate_change_ids.add(case.change_id)
        seen_change_ids.add(case.change_id)
    if duplicate_change_ids:
        duplicates = ", ".join(sorted(duplicate_change_ids))
        raise ShadowTriageEvaluationError(
            f"Le corpus contient des change_id dupliques: {duplicates}."
        )
    return cases


def load_shadow_cases(path: Path) -> list[ShadowTriageCase]:
    """Charge un corpus JSON sans modifier le fichier source."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records: Any
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = payload.get("cases", payload.get("records"))
    else:
        records = None
    if not isinstance(records, list):
        raise ShadowTriageEvaluationError(
            "Le JSON doit etre une liste ou un objet contenant une liste `cases`."
        )
    return coerce_shadow_cases(records)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """Calcule un ratio arrondi ou None lorsque le denominateur est nul."""
    if denominator == 0:
        return None
    return round(numerator / denominator, _ROUND_DIGITS)


def _confusion_matrix(
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Construit une matrice reference par prediction avec ordre stable."""
    matrix = {
        reference: {prediction: 0 for prediction in PREDICTION_LEVELS}
        for reference in IMPACT_LEVELS
    }
    for reference, prediction in zip(references, predictions, strict=True):
        matrix[reference][prediction] += 1
    return matrix


def _linear_weighted_kappa(
    references: Sequence[str],
    predictions: Sequence[str],
) -> float | None:
    """Calcule le kappa de Cohen lineairement pondere pour trois classes ordinales."""
    automated_pairs = [
        (reference, prediction)
        for reference, prediction in zip(
            references,
            predictions,
            strict=True,
        )
        if prediction in IMPACT_LEVELS
    ]
    total = len(automated_pairs)
    if total == 0:
        return None
    automated_references = [value[0] for value in automated_pairs]
    automated_predictions = [value[1] for value in automated_pairs]
    matrix = _confusion_matrix(
        automated_references,
        automated_predictions,
    )
    row_totals = {
        level: sum(matrix[level].values())
        for level in IMPACT_LEVELS
    }
    column_totals = {
        level: sum(matrix[reference][level] for reference in IMPACT_LEVELS)
        for level in IMPACT_LEVELS
    }
    max_distance = len(IMPACT_LEVELS) - 1
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for reference in IMPACT_LEVELS:
        for prediction in IMPACT_LEVELS:
            weight = abs(_IMPACT_SCORE[reference] - _IMPACT_SCORE[prediction]) / max_distance
            observed_disagreement += weight * matrix[reference][prediction] / total
            expected_disagreement += (
                weight
                * row_totals[reference]
                * column_totals[prediction]
                / (total * total)
            )
    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else 0.0
    return round(
        1.0 - (observed_disagreement / expected_disagreement),
        _ROUND_DIGITS,
    )


def compute_triage_metrics(
    references: Sequence[Any],
    predictions: Sequence[Any],
) -> dict[str, Any]:
    """Calcule les metriques asymetriques d'une serie de verdicts.

    ``weighted_agreement`` correspond au kappa de Cohen lineairement pondere.
    Une confusion MAJEUR-MINEUR pese donc davantage qu'une confusion entre
    deux niveaux adjacents.
    """
    if len(references) != len(predictions):
        raise ShadowTriageEvaluationError(
            "Les series de reference et de prediction doivent avoir la meme taille."
        )
    if not references:
        raise ShadowTriageEvaluationError(
            "Le calcul des metriques exige au moins une observation."
        )
    normalized_references = [normalize_impact_level(value) for value in references]
    normalized_predictions = [
        normalize_candidate_level(value) for value in predictions
    ]
    total = len(normalized_references)

    gold_major = sum(level == "MAJEUR" for level in normalized_references)
    gold_moderate = sum(level == "MODERE" for level in normalized_references)
    gold_non_minor = gold_major + gold_moderate
    predicted_major = sum(level == "MAJEUR" for level in normalized_predictions)
    correct_major = sum(
        reference == prediction == "MAJEUR"
        for reference, prediction in zip(
            normalized_references,
            normalized_predictions,
            strict=True,
        )
    )
    automatic_count = sum(
        level in IMPACT_LEVELS for level in normalized_predictions
    )
    review_count = total - automatic_count
    recognized_non_minor = sum(
        reference in {"MAJEUR", "MODERE"} and prediction in {"MAJEUR", "MODERE"}
        for reference, prediction in zip(
            normalized_references,
            normalized_predictions,
            strict=True,
        )
    )
    major_to_minor = sum(
        reference == "MAJEUR" and prediction == "MINEUR"
        for reference, prediction in zip(
            normalized_references,
            normalized_predictions,
            strict=True,
        )
    )
    major_to_moderate = sum(
        reference == "MAJEUR" and prediction == "MODERE"
        for reference, prediction in zip(
            normalized_references,
            normalized_predictions,
            strict=True,
        )
    )
    moderate_to_minor = sum(
        reference == "MODERE" and prediction == "MINEUR"
        for reference, prediction in zip(
            normalized_references,
            normalized_predictions,
            strict=True,
        )
    )
    exact = sum(
        reference == prediction
        for reference, prediction in zip(
            normalized_references,
            normalized_predictions,
            strict=True,
        )
    )
    return {
        "case_count": total,
        "reference_major_count": gold_major,
        "reference_moderate_count": gold_moderate,
        "reference_non_minor_count": gold_non_minor,
        "predicted_major_count": predicted_major,
        "automatic_decision_count": automatic_count,
        "review_required_count": review_count,
        "automatic_coverage": _safe_ratio(automatic_count, total),
        "review_required_rate": _safe_ratio(review_count, total),
        "non_minor_recall": _safe_ratio(recognized_non_minor, gold_non_minor),
        "major_recall": _safe_ratio(correct_major, gold_major),
        "major_to_minor_count": major_to_minor,
        "major_to_minor_rate": _safe_ratio(major_to_minor, gold_major),
        "major_to_moderate_count": major_to_moderate,
        "major_to_moderate_rate": _safe_ratio(
            major_to_moderate,
            gold_major,
        ),
        "moderate_to_minor_count": moderate_to_minor,
        "moderate_to_minor_rate": _safe_ratio(moderate_to_minor, gold_moderate),
        "major_precision": _safe_ratio(correct_major, predicted_major),
        "exact_agreement": _safe_ratio(exact, total),
        "weighted_agreement": _linear_weighted_kappa(
            normalized_references,
            normalized_predictions,
        ),
        "weighted_agreement_method": "linear_weighted_kappa_on_automatic_decisions",
        "weighted_agreement_case_count": automatic_count,
        "confusion_matrix": _confusion_matrix(
            normalized_references,
            normalized_predictions,
        ),
    }


_IMPROVEMENT_DIRECTIONS = {
    "non_minor_recall": 1,
    "major_recall": 1,
    "major_to_minor_count": -1,
    "major_to_minor_rate": -1,
    "major_to_moderate_count": -1,
    "major_to_moderate_rate": -1,
    "moderate_to_minor_count": -1,
    "moderate_to_minor_rate": -1,
    "major_precision": 1,
    "automatic_coverage": 1,
    "review_required_rate": -1,
    "exact_agreement": 1,
    "weighted_agreement": 1,
}


def _metric_difference(
    candidate_value: Any,
    legacy_value: Any,
) -> float | int | None:
    """Soustrait deux valeurs numeriques comparables."""
    if candidate_value is None or legacy_value is None:
        return None
    if not isinstance(candidate_value, (int, float)) or not isinstance(
        legacy_value,
        (int, float),
    ):
        return None
    difference = candidate_value - legacy_value
    if isinstance(candidate_value, int) and isinstance(legacy_value, int):
        return int(difference)
    return round(float(difference), _ROUND_DIGITS)


def _paired_report(cases: Sequence[ShadowTriageCase]) -> dict[str, Any]:
    """Compare les metriques historiques et candidates sur un sous-corpus."""
    references = [case.reference_impact for case in cases]
    legacy = compute_triage_metrics(
        references,
        [case.legacy_impact for case in cases],
    )
    candidate = compute_triage_metrics(
        references,
        [case.candidate_impact for case in cases],
    )
    delta: dict[str, float | int | None] = {}
    improvement: dict[str, float | int | None] = {}
    for metric, direction in _IMPROVEMENT_DIRECTIONS.items():
        difference = _metric_difference(candidate.get(metric), legacy.get(metric))
        delta[metric] = difference
        improvement[metric] = (
            None
            if difference is None
            else round(direction * difference, _ROUND_DIGITS)
        )
    return {
        "case_count": len(cases),
        "legacy": legacy,
        "candidate": candidate,
        "delta_candidate_minus_legacy": delta,
        "improvement": improvement,
    }


def _group_by_dimension(
    cases: Sequence[ShadowTriageCase],
    dimension: str,
) -> dict[str, list[ShadowTriageCase]]:
    """Regroupe les cas par banque, theme ou nature."""
    grouped: dict[str, list[ShadowTriageCase]] = defaultdict(list)
    for case in cases:
        if dimension == "bank":
            values = (case.bank,)
        elif dimension == "theme":
            values = case.themes_amf or (_MISSING_THEME,)
        elif dimension == "nature":
            values = (case.change_nature,)
        else:
            raise ShadowTriageEvaluationError(
                f"Dimension de ventilation inconnue: {dimension!r}."
            )
        for value in values:
            grouped[value].append(case)
    return dict(sorted(grouped.items()))


def evaluate_acceptance(
    candidate_metrics: Mapping[str, Any],
    *,
    thresholds: ShadowAcceptanceThresholds = DEFAULT_SHADOW_THRESHOLDS,
) -> dict[str, Any]:
    """Evalue les metriques candidates contre les seuils d'acceptation.

    Un contrôle dont le dénominateur est absent produit une valeur ``None`` et
    est marqué ``SKIPPED``. La porte est alors ``INCOMPLETE`` afin qu'un corpus
    dépourvu d'une classe importante ne soit jamais déclaré conforme.
    """
    specifications = (
        ("non_minor_recall", ">=", thresholds.min_non_minor_recall),
        ("major_recall", ">=", thresholds.min_major_recall),
        ("major_to_minor_rate", "<=", thresholds.max_major_to_minor_rate),
        (
            "moderate_to_minor_rate",
            "<=",
            thresholds.max_moderate_to_minor_rate,
        ),
        ("major_precision", ">=", thresholds.min_major_precision),
        (
            "automatic_coverage",
            ">=",
            thresholds.min_automatic_coverage,
        ),
        ("weighted_agreement", ">=", thresholds.min_weighted_agreement),
    )
    checks: dict[str, dict[str, Any]] = {}
    failed_checks: list[str] = []
    skipped_checks: list[str] = []
    for metric, operator, threshold in specifications:
        value = candidate_metrics.get(metric)
        if value is None:
            passed: bool | None = None
            status = "SKIPPED"
            skipped_checks.append(metric)
        else:
            numeric_value = float(value)
            passed = (
                numeric_value >= threshold
                if operator == ">="
                else numeric_value <= threshold
            )
            status = "PASS" if passed else "FAIL"
            if not passed:
                failed_checks.append(metric)
        checks[metric] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "status": status,
            "passed": passed,
        }
    overall_status = (
        "FAIL"
        if failed_checks
        else ("INCOMPLETE" if skipped_checks else "PASS")
    )
    return {
        "status": overall_status,
        "thresholds": thresholds.as_dict(),
        "checks": checks,
        "failed_checks": failed_checks,
        "skipped_checks": skipped_checks,
    }


def evaluate_bank_coverage(
    records: Sequence[ShadowTriageCase | Mapping[str, Any]],
    *,
    required_banks: Sequence[str] = SIX_BANK_COVERAGE,
    min_cases_per_bank: int = DEFAULT_MIN_CASES_PER_BANK,
) -> dict[str, Any]:
    """Verifie la presence et un volume minimal pour chaque banque exigee."""
    cases = coerce_shadow_cases(records)
    if min_cases_per_bank < 1:
        raise ShadowTriageEvaluationError(
            "min_cases_per_bank doit etre superieur ou egal a 1."
        )
    normalized_required = tuple(
        dict.fromkeys(_normalize_dimension(bank) for bank in required_banks)
    )
    if not normalized_required:
        raise ShadowTriageEvaluationError(
            "La couverture bancaire exige au moins une banque."
        )
    bank_case_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        if case.bank != _MISSING_DIMENSION:
            bank_case_counts[case.bank] += 1
    present = set(bank_case_counts)
    required_set = set(normalized_required)
    covered = sorted(required_set & present)
    missing = sorted(required_set - present)
    insufficient = sorted(
        bank
        for bank in required_set & present
        if bank_case_counts[bank] < min_cases_per_bank
    )
    qualified = sorted(
        bank
        for bank in required_set & present
        if bank_case_counts[bank] >= min_cases_per_bank
    )
    return {
        "status": "PASS" if not missing and not insufficient else "FAIL",
        "required_banks": list(normalized_required),
        "present_required_banks": covered,
        "qualified_required_banks": qualified,
        "missing_banks": missing,
        "insufficient_banks": insufficient,
        "unexpected_banks": sorted(present - required_set),
        "minimum_cases_per_bank": min_cases_per_bank,
        "case_counts_by_bank": dict(sorted(bank_case_counts.items())),
        "required_bank_count": len(normalized_required),
        "covered_bank_count": len(qualified),
        "coverage_rate": _safe_ratio(
            len(qualified),
            len(normalized_required),
        ),
    }


_EVALUATION_ONLY_FIELDS = {
    "candidate_impact",
    "new_impact",
    "candidate_triage",
    "new_triage",
    "candidate",
    "new",
    "legacy_impact",
    "old_impact",
    "legacy_triage",
    "old_triage",
    "legacy",
    "old",
}


def _sha256_json(payload: Any) -> str:
    """Calcule une empreinte stable d'une valeur serialisable."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _reference_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Empreinte les donnees de reference sans les sorties a comparer."""
    reference_payload = {
        str(key): value
        for key, value in record.items()
        if str(key) not in _EVALUATION_ONLY_FIELDS
    }
    return _sha256_json(reference_payload)


def _reference_corpus_fingerprint(
    cases: Sequence[ShadowTriageCase],
) -> str:
    """Produit une empreinte stable et independante du candidat evalue."""
    payload = [
        {
            "change_id": case.change_id,
            "bank": case.bank,
            "reference_impact": case.reference_impact,
            "themes_amf": list(case.themes_amf),
            "change_nature": case.change_nature,
            "reference_record_fingerprint": (
                case.reference_record_fingerprint
                or _sha256_json(
                    {
                        "change_id": case.change_id,
                        "bank": case.bank,
                        "reference_impact": case.reference_impact,
                        "themes_amf": list(case.themes_amf),
                        "change_nature": case.change_nature,
                    }
                )
            ),
        }
        for case in sorted(cases, key=lambda value: value.change_id)
    ]
    return _sha256_json(payload)


def _evaluation_fingerprint(cases: Sequence[ShadowTriageCase]) -> str:
    """Empreinte le corpus et les deux series de verdicts mesurees."""
    payload = [
        {
            "change_id": case.change_id,
            "legacy_impact": case.legacy_impact,
            "candidate_impact": case.candidate_impact,
        }
        for case in sorted(cases, key=lambda value: value.change_id)
    ]
    return _sha256_json(
        {
            "reference_corpus_fingerprint": _reference_corpus_fingerprint(cases),
            "evaluated_predictions": payload,
        }
    )


def _bank_quality_report(
    grouped_banks: Mapping[str, Sequence[ShadowTriageCase]],
    *,
    thresholds: ShadowAcceptanceThresholds,
    min_cases_per_bank: int,
) -> dict[str, Any]:
    """Applique les seuils a chaque banque sans masquer les petits volumes."""
    banks: dict[str, dict[str, Any]] = {}
    failed_banks: list[str] = []
    incomplete_banks: list[str] = []
    for bank, bank_cases in sorted(grouped_banks.items()):
        paired = _paired_report(bank_cases)
        acceptance = evaluate_acceptance(
            paired["candidate"],
            thresholds=thresholds,
        )
        enough_cases = len(bank_cases) >= min_cases_per_bank
        if acceptance["status"] == "FAIL":
            status = "FAIL"
            failed_banks.append(bank)
        elif not enough_cases or acceptance["status"] == "INCOMPLETE":
            status = "INCOMPLETE"
            incomplete_banks.append(bank)
        else:
            status = "PASS"
        banks[bank] = {
            "status": status,
            "case_count": len(bank_cases),
            "minimum_cases_required": min_cases_per_bank,
            "minimum_cases_status": (
                "PASS" if enough_cases else "FAIL"
            ),
            "acceptance": acceptance,
        }
    overall_status = (
        "FAIL"
        if failed_banks
        else ("INCOMPLETE" if incomplete_banks else "PASS")
    )
    return {
        "status": overall_status,
        "banks": banks,
        "failed_banks": failed_banks,
        "incomplete_banks": incomplete_banks,
    }


def evaluate_shadow_triage(
    records: Sequence[ShadowTriageCase | Mapping[str, Any]],
    *,
    thresholds: ShadowAcceptanceThresholds = DEFAULT_SHADOW_THRESHOLDS,
    required_banks: Sequence[str] = SIX_BANK_COVERAGE,
    min_cases_per_bank: int = DEFAULT_MIN_CASES_PER_BANK,
) -> dict[str, Any]:
    """Evalue ancien et nouveau triages globalement et par dimensions metier."""
    cases = coerce_shadow_cases(records)
    report = _paired_report(cases)
    grouped = {
        dimension: _group_by_dimension(cases, dimension)
        for dimension in ("bank", "theme", "nature")
    }
    acceptance = evaluate_acceptance(
        report["candidate"],
        thresholds=thresholds,
    )
    coverage = evaluate_bank_coverage(
        cases,
        required_banks=required_banks,
        min_cases_per_bank=min_cases_per_bank,
    )
    bank_quality = _bank_quality_report(
        grouped["bank"],
        thresholds=thresholds,
        min_cases_per_bank=min_cases_per_bank,
    )
    readiness_components = {
        "global_acceptance": acceptance["status"],
        "bank_quality": bank_quality["status"],
        "bank_coverage": coverage["status"],
    }
    if "FAIL" in readiness_components.values():
        release_status = "FAIL"
    elif "INCOMPLETE" in readiness_components.values():
        release_status = "INCOMPLETE"
    else:
        release_status = "PASS"
    reference_fingerprint = _reference_corpus_fingerprint(cases)
    report.update(
        {
            "schema_version": "triage_shadow_evaluation.v2",
            "impact_levels": list(IMPACT_LEVELS),
            "prediction_levels": list(PREDICTION_LEVELS),
            "corpus": {
                "case_count": len(cases),
                "unique_change_id_count": len(
                    {case.change_id for case in cases}
                ),
                "fingerprint_sha256": reference_fingerprint,
                "reference_fingerprint_sha256": reference_fingerprint,
                "evaluation_fingerprint_sha256": (
                    _evaluation_fingerprint(cases)
                ),
            },
            "acceptance": acceptance,
            "bank_coverage": coverage,
            "bank_quality": bank_quality,
            "release_readiness": {
                "status": release_status,
                "components": readiness_components,
                "blocking_reasons": [
                    component
                    for component, status in readiness_components.items()
                    if status != "PASS"
                ],
            },
            "breakdowns": {
                dimension: {
                    value: _paired_report(group_cases)
                    for value, group_cases in grouped[dimension].items()
                }
                for dimension in ("bank", "theme", "nature")
            },
        }
    )
    return report


def write_shadow_report(report: Mapping[str, Any], path: Path) -> Path:
    """Ecrit un nouveau rapport JSON et refuse tout ecrasement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return target

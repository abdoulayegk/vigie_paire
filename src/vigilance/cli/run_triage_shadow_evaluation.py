"""CLI de validation parallele des verdicts de triage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vigilance.triage_shadow_evaluation import (
    DEFAULT_MIN_CASES_PER_BANK,
    DEFAULT_SHADOW_THRESHOLDS,
    ShadowAcceptanceThresholds,
    ShadowTriageEvaluationError,
    evaluate_shadow_triage,
    load_shadow_cases,
    write_shadow_report,
)


def _unit_ratio(value: str) -> float:
    """Convertit une option CLI en ratio compris entre 0 et 1."""
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("la valeur doit etre comprise entre 0 et 1")
    return ratio


def _weighted_agreement(value: str) -> float:
    """Convertit un seuil de kappa compris entre -1 et 1."""
    threshold = float(value)
    if not -1.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("la valeur doit etre comprise entre -1 et 1")
    return threshold


def _positive_integer(value: str) -> int:
    """Convertit une option CLI en entier strictement positif."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "la valeur doit etre superieure ou egale a 1"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur de la commande de validation parallele."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare les classifications historique et candidate a un corpus "
            "de reference, sans modifier les resultats de production."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Corpus JSON contenant les cas de reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Nouveau fichier JSON a creer. Sans cette option, le rapport est "
            "ecrit sur la sortie standard."
        ),
    )
    parser.add_argument(
        "--require-six-banks",
        action="store_true",
        help=(
            "Retourne un echec si BMO, BNC, BNS, CIBC, RBC et TD ne sont "
            "pas toutes presentes avec le nombre minimal de cas exige."
        ),
    )
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Retourne un echec si le triage candidat ne respecte pas les seuils.",
    )
    parser.add_argument(
        "--min-non-minor-recall",
        type=_unit_ratio,
        default=DEFAULT_SHADOW_THRESHOLDS.min_non_minor_recall,
        help="Rappel minimal des changements MAJEUR et MODERE.",
    )
    parser.add_argument(
        "--max-major-to-minor-rate",
        type=_unit_ratio,
        default=DEFAULT_SHADOW_THRESHOLDS.max_major_to_minor_rate,
        help="Taux maximal de changements MAJEUR classes MINEUR.",
    )
    parser.add_argument(
        "--min-major-recall",
        type=_unit_ratio,
        default=DEFAULT_SHADOW_THRESHOLDS.min_major_recall,
        help="Rappel minimal des changements MAJEUR.",
    )
    parser.add_argument(
        "--max-moderate-to-minor-rate",
        type=_unit_ratio,
        default=DEFAULT_SHADOW_THRESHOLDS.max_moderate_to_minor_rate,
        help="Taux maximal de changements MODERE classes MINEUR.",
    )
    parser.add_argument(
        "--min-major-precision",
        type=_unit_ratio,
        default=DEFAULT_SHADOW_THRESHOLDS.min_major_precision,
        help="Precision minimale des predictions MAJEUR.",
    )
    parser.add_argument(
        "--min-automatic-coverage",
        type=_unit_ratio,
        default=DEFAULT_SHADOW_THRESHOLDS.min_automatic_coverage,
        help="Part minimale des cas recevant un niveau automatique final.",
    )
    parser.add_argument(
        "--min-weighted-agreement",
        type=_weighted_agreement,
        default=DEFAULT_SHADOW_THRESHOLDS.min_weighted_agreement,
        help="Kappa ordinal pondere minimal.",
    )
    parser.add_argument(
        "--min-cases-per-bank",
        type=_positive_integer,
        default=DEFAULT_MIN_CASES_PER_BANK,
        help=(
            "Nombre minimal de cas valides exige pour chaque banque "
            f"(defaut: {DEFAULT_MIN_CASES_PER_BANK})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute l'evaluation parallele et retourne un code de sortie."""
    args = build_parser().parse_args(argv)
    try:
        cases = load_shadow_cases(args.input)
        thresholds = ShadowAcceptanceThresholds(
            min_non_minor_recall=args.min_non_minor_recall,
            min_major_recall=args.min_major_recall,
            max_major_to_minor_rate=args.max_major_to_minor_rate,
            max_moderate_to_minor_rate=args.max_moderate_to_minor_rate,
            min_major_precision=args.min_major_precision,
            min_automatic_coverage=args.min_automatic_coverage,
            min_weighted_agreement=args.min_weighted_agreement,
        )
        report = evaluate_shadow_triage(
            cases,
            thresholds=thresholds,
            min_cases_per_bank=args.min_cases_per_bank,
        )
        quality_status = (
            "PASS"
            if report["acceptance"]["status"] == "PASS"
            and report["bank_quality"]["status"] == "PASS"
            else (
                "FAIL"
                if "FAIL"
                in {
                    report["acceptance"]["status"],
                    report["bank_quality"]["status"],
                }
                else "INCOMPLETE"
            )
        )
        gate_failed = args.fail_on_thresholds and quality_status != "PASS"
        coverage_failed = (
            args.require_six_banks
            and report["bank_coverage"]["status"] != "PASS"
        )
        report["enforcement"] = {
            "fail_on_thresholds": args.fail_on_thresholds,
            "require_six_banks": args.require_six_banks,
            "minimum_cases_per_bank": args.min_cases_per_bank,
            "quality_status": quality_status,
            "quality_blocked": gate_failed,
            "coverage_blocked": coverage_failed,
            "exit_status": (
                "BLOCKED" if gate_failed or coverage_failed else "ALLOWED"
            ),
        }
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            output = write_shadow_report(report, args.output)
            print(f"Rapport de validation parallele cree: {output}")
    except (OSError, json.JSONDecodeError, ShadowTriageEvaluationError) as exc:
        print(f"Erreur de validation parallele: {exc}", file=sys.stderr)
        return 2
    if gate_failed:
        global_issues = [
            *report["acceptance"]["failed_checks"],
            *report["acceptance"]["skipped_checks"],
        ]
        bank_issues = [
            *report["bank_quality"]["failed_banks"],
            *report["bank_quality"]["incomplete_banks"],
        ]
        details = ", ".join(
            [
                *(f"metrique:{value}" for value in global_issues),
                *(f"banque:{value}" for value in bank_issues),
            ]
        )
        print(
            "Seuils d'acceptation non respectes ou incomplets: "
            f"{details}.",
            file=sys.stderr,
        )
    if coverage_failed:
        coverage_issues = [
            *report["bank_coverage"]["missing_banks"],
            *report["bank_coverage"]["insufficient_banks"],
        ]
        print(
            "Couverture des six banques incomplete: "
            f"{', '.join(coverage_issues)}.",
            file=sys.stderr,
        )
    if gate_failed or coverage_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

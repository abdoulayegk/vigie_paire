from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigilance.cli.run_triage_shadow_evaluation import main
from vigilance.triage_shadow_evaluation import (
    SIX_BANK_COVERAGE,
    ShadowAcceptanceThresholds,
    ShadowTriageEvaluationError,
    compute_triage_metrics,
    evaluate_acceptance,
    evaluate_bank_coverage,
    evaluate_shadow_triage,
    load_shadow_cases,
    normalize_impact_level,
    write_shadow_report,
)


def _cases() -> list[dict[str, object]]:
    return [
        {
            "change_id": "bmo-1",
            "bank": "BMO",
            "themes_amf": ["GOUVERNANCE_RISQUES"],
            "change_nature": "PERIMETRE",
            "reference_impact": "MAJEUR",
            "legacy_impact": "MINEUR",
            "candidate_impact": "MAJEUR",
        },
        {
            "change_id": "bmo-2",
            "bank": "BMO",
            "themes_amf": ["FONDS_PROPRES", "GOUVERNANCE_RISQUES"],
            "change_nature": "TERMINOLOGIE_AMBIGUE",
            "reference_impact": "MODERE",
            "legacy_impact": "MINEUR",
            "candidate_impact": "MODERE",
        },
        {
            "change_id": "rbc-1",
            "bank": "RBC",
            "themes_amf": ["FONDS_PROPRES"],
            "change_nature": "METHODE",
            "reference_impact": "MODERE",
            "legacy_impact": "MODERE",
            "candidate_impact": "MAJEUR",
        },
        {
            "change_id": "rbc-2",
            "bank": "RBC",
            "themes_amf": [],
            "change_nature": "REFORMULATION",
            "reference_impact": "MINEUR",
            "legacy_impact": "MAJEUR",
            "candidate_impact": "MINEUR",
        },
        {
            "change_id": "td-1",
            "bank": "TD",
            "themes_amf": ["GOUVERNANCE_RISQUES"],
            "change_nature": "GOUVERNANCE",
            "reference_impact": "MAJEUR",
            "legacy_impact": "MAJEUR",
            "candidate_impact": "MAJEUR",
        },
    ]


def _balanced_six_bank_cases(
    *,
    repetitions_per_level: int = 7,
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for bank in SIX_BANK_COVERAGE:
        for level in ("MINEUR", "MODERE", "MAJEUR"):
            for index in range(repetitions_per_level):
                cases.append(
                    {
                        "change_id": f"{bank}-{level}-{index}",
                        "bank": bank,
                        "reference_impact": level,
                        "legacy_impact": "MINEUR",
                        "candidate_impact": level,
                    }
                )
    return cases


def test_compute_metrics_exposes_costly_underclassifications() -> None:
    metrics = compute_triage_metrics(
        ["MAJEUR", "MODERE", "MODERE", "MINEUR", "MAJEUR"],
        ["MINEUR", "MINEUR", "MODERE", "MAJEUR", "MAJEUR"],
    )

    assert metrics["non_minor_recall"] == 0.5
    assert metrics["major_to_minor_count"] == 1
    assert metrics["major_to_minor_rate"] == 0.5
    assert metrics["moderate_to_minor_count"] == 1
    assert metrics["moderate_to_minor_rate"] == 0.5
    assert metrics["major_precision"] == 0.5
    assert metrics["confusion_matrix"]["MAJEUR"]["MINEUR"] == 1


def test_shadow_report_compares_legacy_candidate_and_breakdowns() -> None:
    report = evaluate_shadow_triage(_cases())

    assert report["legacy"]["non_minor_recall"] == 0.5
    assert report["candidate"]["non_minor_recall"] == 1.0
    assert report["candidate"]["major_to_minor_count"] == 0
    assert report["candidate"]["moderate_to_minor_count"] == 0
    assert report["candidate"]["major_precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert report["improvement"]["non_minor_recall"] == 0.5
    assert report["improvement"]["major_to_minor_count"] == 1
    assert (
        report["candidate"]["weighted_agreement"]
        > report["legacy"]["weighted_agreement"]
    )

    breakdowns = report["breakdowns"]
    assert set(breakdowns["bank"]) == {"BMO", "RBC", "TD"}
    assert breakdowns["bank"]["BMO"]["candidate"]["non_minor_recall"] == 1.0
    assert breakdowns["theme"]["GOUVERNANCE_RISQUES"]["case_count"] == 3
    assert breakdowns["theme"]["SANS_THEME"]["case_count"] == 1
    assert breakdowns["nature"]["METHODE"]["case_count"] == 1


def test_perfect_predictions_have_perfect_weighted_agreement() -> None:
    metrics = compute_triage_metrics(
        ["MINEUR", "MODERE", "MAJEUR"],
        ["MINEUR", "MODERE", "MAJEUR"],
    )

    assert metrics["exact_agreement"] == 1.0
    assert metrics["weighted_agreement"] == 1.0


def test_normalization_accepts_accents_and_nested_triages(tmp_path: Path) -> None:
    corpus = {
        "cases": [
            {
                "uid": "c-1",
                "banque": "bns",
                "themes": "risque_donnees",
                "nature": "portée",
                "gold": {"impact_level": "Modérée"},
                "old_triage": {"impact_level": "minor"},
                "new_triage": {"impact_level": "MODÉRÉ"},
            }
        ]
    }
    source = tmp_path / "corpus.json"
    source.write_text(json.dumps(corpus), encoding="utf-8")

    cases = load_shadow_cases(source)

    assert normalize_impact_level("Modérée") == "MODERE"
    assert cases[0].bank == "BNS"
    assert cases[0].change_nature == "PORTÉE"
    assert cases[0].themes_amf == ("RISQUE_DONNEES",)
    assert cases[0].reference_impact == "MODERE"
    assert cases[0].legacy_impact == "MINEUR"


def test_nested_unresolved_minor_is_preserved_as_review_abstention(
    tmp_path: Path,
) -> None:
    corpus = {
        "cases": [
            {
                "change_id": "review-1",
                "bank": "BMO",
                "reference_impact": "MAJEUR",
                "legacy_impact": "MINEUR",
                "candidate_triage": {
                    "impact_level": "MINEUR",
                    "materiality_level": "MINEUR",
                    "decision_status": "A_CONFIRMER",
                    "review_required": True,
                },
            }
        ]
    }
    source = tmp_path / "review-corpus.json"
    source.write_text(json.dumps(corpus), encoding="utf-8")

    case = load_shadow_cases(source)[0]
    metrics = compute_triage_metrics(
        [case.reference_impact],
        [case.candidate_impact],
    )

    assert case.candidate_impact == "A_CONFIRMER"
    assert metrics["review_required_count"] == 1
    assert metrics["automatic_coverage"] == 0.0
    assert metrics["major_to_minor_count"] == 0
    assert metrics["major_recall"] == 0.0
    assert metrics["confusion_matrix"]["MAJEUR"]["A_CONFIRMER"] == 1
    assert metrics["weighted_agreement"] is None


@pytest.mark.parametrize("provisional_level", ["MODERE", "MAJEUR"])
def test_nested_unresolved_non_minor_is_preserved_as_review_abstention(
    tmp_path: Path,
    provisional_level: str,
) -> None:
    corpus = {
        "cases": [
            {
                "change_id": f"review-{provisional_level.lower()}",
                "bank": "BMO",
                "reference_impact": provisional_level,
                "legacy_impact": "MINEUR",
                "candidate_triage": {
                    "impact_level": provisional_level,
                    "materiality_level": provisional_level,
                    "decision_status": "A_CONFIRMER",
                    "review_required": True,
                },
            }
        ]
    }
    source = tmp_path / f"review-{provisional_level.lower()}.json"
    source.write_text(json.dumps(corpus), encoding="utf-8")

    case = load_shadow_cases(source)[0]

    assert case.candidate_impact == "A_CONFIRMER"


def test_nested_review_status_takes_priority_over_direct_candidate_level(
    tmp_path: Path,
) -> None:
    corpus = {
        "cases": [
            {
                "change_id": "review-conflict",
                "bank": "BMO",
                "reference_impact": "MAJEUR",
                "legacy_impact": "MINEUR",
                "candidate_impact": "MAJEUR",
                "candidate_triage": {
                    "materiality_level": "MAJEUR",
                    "decision_status": "A_CONFIRMER",
                    "review_required": "true",
                },
            }
        ]
    }
    source = tmp_path / "review-conflict.json"
    source.write_text(json.dumps(corpus), encoding="utf-8")

    case = load_shadow_cases(source)[0]

    assert case.candidate_impact == "A_CONFIRMER"


def test_string_false_does_not_create_a_review_abstention(tmp_path: Path) -> None:
    corpus = {
        "cases": [
            {
                "change_id": "final-major",
                "bank": "BMO",
                "reference_impact": "MAJEUR",
                "legacy_impact": "MINEUR",
                "candidate_triage": {
                    "materiality_level": "MAJEUR",
                    "decision_status": "CONFIRME",
                    "review_required": "false",
                },
            }
        ]
    }
    source = tmp_path / "final-major.json"
    source.write_text(json.dumps(corpus), encoding="utf-8")

    assert load_shadow_cases(source)[0].candidate_impact == "MAJEUR"


def test_unknown_impact_is_rejected_with_case_context() -> None:
    case = _cases()[0] | {"candidate_impact": "CRITIQUE"}

    with pytest.raises(
        ShadowTriageEvaluationError,
        match="bmo-1.*candidate_impact",
    ):
        evaluate_shadow_triage([case])


def test_duplicate_change_ids_are_rejected() -> None:
    duplicate = dict(_cases()[0])

    with pytest.raises(
        ShadowTriageEvaluationError,
        match="change_id dupliques.*bmo-1",
    ):
        evaluate_shadow_triage([_cases()[0], duplicate])


def test_writer_and_cli_refuse_to_overwrite_existing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "corpus.json"
    source.write_text(json.dumps({"cases": _cases()}), encoding="utf-8")
    existing = tmp_path / "shadow-report.json"
    existing.write_text("contenu-utilisateur", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_shadow_report(evaluate_shadow_triage(_cases()), existing)

    assert main(["--input", str(source), "--output", str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "contenu-utilisateur"
    assert "Erreur de validation parallele" in capsys.readouterr().err


def test_cli_writes_report_to_stdout_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "corpus.json"
    source.write_text(json.dumps({"cases": _cases()}), encoding="utf-8")

    assert main(["--input", str(source)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "triage_shadow_evaluation.v2"
    assert output["case_count"] == 5
    assert output["enforcement"]["exit_status"] == "ALLOWED"
    assert len(output["corpus"]["fingerprint_sha256"]) == 64


def test_reference_fingerprint_is_independent_from_evaluated_predictions() -> None:
    first_cases = _cases()
    second_cases = [
        {
            **case,
            "legacy_impact": "MAJEUR",
            "candidate_impact": "MINEUR",
        }
        for case in first_cases
    ]

    first_report = evaluate_shadow_triage(first_cases)
    second_report = evaluate_shadow_triage(second_cases)

    assert (
        first_report["corpus"]["reference_fingerprint_sha256"]
        == second_report["corpus"]["reference_fingerprint_sha256"]
    )
    assert (
        first_report["corpus"]["evaluation_fingerprint_sha256"]
        != second_report["corpus"]["evaluation_fingerprint_sha256"]
    )


def test_reference_fingerprint_captures_source_evidence_changes() -> None:
    first_cases = [
        {
            **_cases()[0],
            "reference_evidence": "Le mandat du comité est élargi.",
        }
    ]
    second_cases = [
        {
            **_cases()[0],
            "reference_evidence": "Le mandat du comité demeure inchangé.",
        }
    ]

    first_report = evaluate_shadow_triage(first_cases)
    second_report = evaluate_shadow_triage(second_cases)

    assert (
        first_report["corpus"]["reference_fingerprint_sha256"]
        != second_report["corpus"]["reference_fingerprint_sha256"]
    )


def test_acceptance_gate_exposes_thresholds_and_failed_checks() -> None:
    report = evaluate_shadow_triage(_cases())

    assert report["acceptance"]["status"] == "FAIL"
    assert report["acceptance"]["thresholds"]["min_non_minor_recall"] == 0.95
    assert "major_precision" in report["acceptance"]["failed_checks"]
    assert report["acceptance"]["checks"]["major_precision"]["operator"] == ">="

    permissive = ShadowAcceptanceThresholds(
        min_non_minor_recall=0.0,
        max_major_to_minor_rate=1.0,
        max_moderate_to_minor_rate=1.0,
        min_major_precision=0.0,
        min_weighted_agreement=-1.0,
    )
    acceptance = evaluate_acceptance(
        report["candidate"],
        thresholds=permissive,
    )
    assert acceptance["status"] == "PASS"
    assert acceptance["failed_checks"] == []


def test_acceptance_gate_blocks_major_to_moderate_collapse() -> None:
    references = (
        ["MINEUR"] * 1000
        + ["MODERE"] * 100
        + ["MAJEUR"] * 100
    )
    predictions = (
        ["MINEUR"] * 1000
        + ["MODERE"] * 100
        + ["MAJEUR"] * 20
        + ["MODERE"] * 80
    )

    metrics = compute_triage_metrics(references, predictions)
    acceptance = evaluate_acceptance(metrics)

    assert metrics["major_recall"] == 0.2
    assert metrics["major_to_minor_rate"] == 0.0
    assert acceptance["status"] == "FAIL"
    assert "major_recall" in acceptance["failed_checks"]


def test_bank_coverage_reports_missing_six_bank_members() -> None:
    coverage = evaluate_bank_coverage(
        _cases(),
        min_cases_per_bank=1,
    )

    assert coverage["status"] == "FAIL"
    assert coverage["required_banks"] == list(SIX_BANK_COVERAGE)
    assert coverage["present_required_banks"] == ["BMO", "RBC", "TD"]
    assert coverage["missing_banks"] == ["BNC", "BNS", "CIBC"]
    assert coverage["coverage_rate"] == 0.5


def test_bank_coverage_rejects_present_but_undersized_banks() -> None:
    cases = [
        {
            "change_id": f"{bank}-1",
            "bank": bank,
            "reference_impact": "MAJEUR",
            "legacy_impact": "MINEUR",
            "candidate_impact": "MAJEUR",
        }
        for bank in SIX_BANK_COVERAGE
    ]

    coverage = evaluate_bank_coverage(cases)

    assert coverage["status"] == "FAIL"
    assert coverage["missing_banks"] == []
    assert coverage["insufficient_banks"] == list(SIX_BANK_COVERAGE)
    assert coverage["coverage_rate"] == 0.0


def test_acceptance_is_incomplete_when_a_reference_class_is_absent() -> None:
    metrics = compute_triage_metrics(
        ["MINEUR", "MODERE"],
        ["MINEUR", "MODERE"],
    )

    acceptance = evaluate_acceptance(metrics)

    assert acceptance["status"] == "INCOMPLETE"
    assert "major_to_minor_rate" in acceptance["skipped_checks"]
    assert "major_precision" in acceptance["skipped_checks"]


def test_bank_coverage_passes_for_all_six_banks() -> None:
    cases = _balanced_six_bank_cases()

    assert evaluate_bank_coverage(cases)["status"] == "PASS"


def test_bank_gate_detects_failure_hidden_by_global_volume() -> None:
    cases = _balanced_six_bank_cases()
    for index, case in enumerate(cases):
        if case["bank"] == "BMO" and case["reference_impact"] == "MAJEUR":
            cases[index] = {**case, "candidate_impact": "MODERE"}
    for bank in ("BNC", "BNS", "CIBC", "RBC", "TD"):
        for extra_index in range(100):
            level = ("MINEUR", "MODERE", "MAJEUR")[extra_index % 3]
            cases.append(
                {
                    "change_id": f"{bank}-extra-{extra_index}",
                    "bank": bank,
                    "reference_impact": level,
                    "legacy_impact": "MINEUR",
                    "candidate_impact": level,
                }
            )

    report = evaluate_shadow_triage(cases)

    assert report["acceptance"]["status"] == "PASS"
    assert report["bank_quality"]["status"] == "FAIL"
    assert "BMO" in report["bank_quality"]["failed_banks"]
    assert report["release_readiness"]["status"] == "FAIL"


def test_cli_can_fail_on_thresholds_or_bank_coverage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "corpus.json"
    source.write_text(json.dumps({"cases": _cases()}), encoding="utf-8")

    assert main(["--input", str(source), "--fail-on-thresholds"]) == 1
    first_capture = capsys.readouterr()
    assert "Seuils d'acceptation non respectes" in first_capture.err
    assert json.loads(first_capture.out)["acceptance"]["status"] == "FAIL"

    assert main(["--input", str(source), "--require-six-banks"]) == 1
    second_capture = capsys.readouterr()
    assert "Couverture des six banques incomplete" in second_capture.err
    assert json.loads(second_capture.out)["bank_coverage"]["status"] == "FAIL"


def test_cli_reports_incomplete_checks_and_still_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {
            "change_id": "bmo-minor",
            "bank": "BMO",
            "reference_impact": "MINEUR",
            "legacy_impact": "MINEUR",
            "candidate_impact": "MINEUR",
        },
        {
            "change_id": "bmo-moderate",
            "bank": "BMO",
            "reference_impact": "MODERE",
            "legacy_impact": "MINEUR",
            "candidate_impact": "MODERE",
        },
    ]
    source = tmp_path / "incomplete.json"
    output = tmp_path / "incomplete-report.json"
    source.write_text(json.dumps({"cases": cases}), encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--fail-on-thresholds",
            "--min-cases-per-bank",
            "1",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert output.exists()
    assert "metrique:major_recall" in captured.err
    assert report["acceptance"]["status"] == "INCOMPLETE"
    assert report["enforcement"]["exit_status"] == "BLOCKED"


def test_cli_passes_quality_and_coverage_with_complete_corpus(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = _balanced_six_bank_cases()
    source = tmp_path / "six-banks.json"
    source.write_text(json.dumps({"cases": cases}), encoding="utf-8")

    assert (
        main(
            [
                "--input",
                str(source),
                "--require-six-banks",
                "--fail-on-thresholds",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["acceptance"]["status"] == "PASS"
    assert output["bank_quality"]["status"] == "PASS"
    assert output["bank_coverage"]["status"] == "PASS"
    assert output["enforcement"]["exit_status"] == "ALLOWED"

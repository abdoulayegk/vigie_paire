"""
Metriques de matching pour la reduction des faux positifs (Phase 5).

Schema jeu de verite:
{
  "bank_code": "bnc",
  "pairs": [
    {"t1_table_id": "...", "t2_table_id": "...", "section": "gestion_capital"},
  ],
  "version": 1
}

Metriques: FPR, precision, recall par banque.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BankMatchingMetrics:
    """Metriques de matching pour une banque."""

    bank_code: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    pairs_evaluated: int = 0
    pairs_total: int = 0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate = FP / (FP + TN). Avec TN = total - TP - FP - FN non applicable;
        on utilise FPR = FP / (TP + FP) comme taux de faux positifs parmi les predictions positives."""
        denom = self.true_positives + self.false_positives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "bank_code": self.bank_code,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "pairs_evaluated": self.pairs_evaluated,
            "pairs_total": self.pairs_total,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "fpr": round(self.fpr, 4),
            "f1": round(self.f1, 4),
        }


def load_ground_truth(path: Path | str, bank_code: Optional[str] = None) -> list[dict]:
    """
    Charger le jeu de verite (paires t1->t2 attendues).

    Args:
        path: Fichier JSON ou repertoire contenant bank_<code>_ground_truth.json
        bank_code: Filtrer par banque si fourni

    Returns:
        Liste de dicts {"t1_table_id", "t2_table_id", "section"?, ...}
    """
    path = Path(path)
    pairs: list[dict] = []

    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if bank_code and data.get("bank_code") != bank_code:
            return []
        pairs = data.get("pairs", [])
    elif path.is_dir():
        pattern = f"*{bank_code}*ground_truth*.json" if bank_code else "*ground_truth*.json"
        for fp in path.glob(pattern):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            pairs.extend(data.get("pairs", []))

    return pairs


def evaluate_predictions(
    ground_truth_pairs: list[dict],
    predicted_matches: list[dict],
    bank_code: str = "",
) -> BankMatchingMetrics:
    """
    Evaluer les predictions vs le jeu de verite.

    Args:
        ground_truth_pairs: [{"t1_table_id": "...", "t2_table_id": "..."}, ...]
        predicted_matches: [{"t1_table_id": "...", "t2_table_id": "..."}, ...]
        bank_code: Code banque pour le rapport

    Returns:
        BankMatchingMetrics
    """
    gt_set = {(p["t1_table_id"], p["t2_table_id"]) for p in ground_truth_pairs}
    pred_set = {(p["t1_table_id"], p["t2_table_id"]) for p in predicted_matches}

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    return BankMatchingMetrics(
        bank_code=bank_code,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        pairs_evaluated=len(pred_set),
        pairs_total=len(gt_set),
    )

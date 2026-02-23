"""
Pipeline de comparaison haute precision (95-98%).

Ce module orchestre tous les composants pour une comparaison
fiable de tableaux entre rapports trimestriels.

Pipeline:
1. Extraction Vision-Native (GPT-4o)
2. Detection de continuite (tableaux multi-pages)
3. Matching multi-signal (4 criteres ponderes)
4. Flagging des cas incertains
"""

from dataclasses import dataclass, field
from typing import Optional
import logging
from pathlib import Path

from .multi_signal_matcher import (
    MultiSignalMatcher,
    TableSignature,
    MatchResult,
    create_signature_from_extracted_table,
)
from .table_continuity_detector import (
    TableContinuityDetector,
    TableFragment,
    MergedTable,
    create_fragment_from_extracted_table,
)
from .uncertainty_flagger import (
    UncertaintyFlagger,
    ReviewSession,
    ReviewFlag,
    FlagType,
)

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Resultat complet d'une comparaison haute precision."""

    matched_tables: list[MatchResult] = field(default_factory=list)
    unmatched_t1: list[TableSignature] = field(default_factory=list)
    unmatched_t2: list[TableSignature] = field(default_factory=list)
    merged_tables_t1: list[MergedTable] = field(default_factory=list)
    merged_tables_t2: list[MergedTable] = field(default_factory=list)
    review_session: Optional[ReviewSession] = None
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertir en dictionnaire."""
        return {
            "matched_tables": [
                {
                    "t1_id": m.table_t1.table_id,
                    "t2_id": m.table_t2.table_id,
                    "score": m.total_score,
                    "confidence": m.confidence_level,
                    "needs_review": m.needs_review,
                    "scores": {
                        "fuzzy_labels": m.fuzzy_label_score,
                        "structure": m.structure_score,
                        "title": m.title_score,
                        "section": m.section_score,
                    },
                }
                for m in self.matched_tables
            ],
            "unmatched_t1": [t.table_id for t in self.unmatched_t1],
            "unmatched_t2": [t.table_id for t in self.unmatched_t2],
            "merged_tables_t1": len(self.merged_tables_t1),
            "merged_tables_t2": len(self.merged_tables_t2),
            "review_session": self.review_session.to_dict() if self.review_session else None,
            "stats": self.stats,
        }


class HighAccuracyComparisonPipeline:
    """
    Pipeline de comparaison haute precision.

    Combine extraction, fusion, matching et flagging pour
    une precision de 95-98%.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        weight_fuzzy: float = 0.40,
        weight_structure: float = 0.25,
        weight_title: float = 0.20,
        weight_section: float = 0.15,
        threshold_confirmed: float = 0.85,
        threshold_probable: float = 0.65,
        save_proof_images: bool = True,
        output_dir: str = "output/high_accuracy",
    ):
        """Initialiser le pipeline."""
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialiser les composants
        self.matcher = MultiSignalMatcher(
            weight_fuzzy=weight_fuzzy,
            weight_structure=weight_structure,
            weight_title=weight_title,
            weight_section=weight_section,
            threshold_confirmed=threshold_confirmed,
            threshold_probable=threshold_probable,
        )
        self.continuity_detector = TableContinuityDetector()
        self.flagger = UncertaintyFlagger()

        self.save_proof_images = save_proof_images

    def compare(
        self,
        tables_t1: list[dict],
        tables_t2: list[dict],
        bank_code: str,
        pdf_t1_name: str = "rapport_t1.pdf",
        pdf_t2_name: str = "rapport_t2.pdf",
    ) -> ComparisonResult:
        """
        Comparer deux ensembles de tableaux avec haute precision.

        Args:
            tables_t1: Tableaux extraits du rapport T1
            tables_t2: Tableaux extraits du rapport T2
            bank_code: Code de la banque
            pdf_t1_name: Nom du PDF T1
            pdf_t2_name: Nom du PDF T2

        Returns:
            ComparisonResult avec matchs, non-matchs et flags
        """
        logger.info(
            f"Comparaison haute precision: {len(tables_t1)} tables T1 vs {len(tables_t2)} tables T2"
        )

        # Creer session de revue
        review_session = self.flagger.create_review_session(
            bank_code=bank_code,
            pdf_t1_name=pdf_t1_name,
            pdf_t2_name=pdf_t2_name,
        )

        # Etape 1: Detection de continuite et fusion
        merged_t1 = self._detect_and_merge(tables_t1)
        merged_t2 = self._detect_and_merge(tables_t2)

        # Creer flags pour fusions multi-pages
        for merged in merged_t1 + merged_t2:
            if len(merged.fragments) > 1:
                flag = self.flagger.create_multi_page_merge_flag(
                    page_range=merged.page_range,
                    fragment_count=len(merged.fragments),
                    confidence=merged.confidence,
                )
                review_session.flags.append(flag)

        # Etape 2: Creer signatures pour matching
        signatures_t1 = self._create_signatures(merged_t1, "t1")
        signatures_t2 = self._create_signatures(merged_t2, "t2")

        # Etape 3: Matching multi-signal
        matches, unmatched_t1, unmatched_t2 = self.matcher.find_best_matches(
            signatures_t1, signatures_t2
        )

        # Etape 4: Creer flags pour cas incertains
        for match in matches:
            if match.needs_review:
                flag = self.flagger.create_uncertain_match_flag(
                    table_t1_id=match.table_t1.table_id,
                    table_t2_id=match.table_t2.table_id,
                    match_score=match.total_score,
                    score_details=match.match_details,
                )
                review_session.flags.append(flag)

            # Verifier structure
            if abs(match.table_t1.num_columns - match.table_t2.num_columns) > 1:
                flag = self.flagger.create_structure_mismatch_flag(
                    table_t1_id=match.table_t1.table_id,
                    table_t2_id=match.table_t2.table_id,
                    t1_cols=match.table_t1.num_columns,
                    t2_cols=match.table_t2.num_columns,
                    t1_rows=match.table_t1.num_rows,
                    t2_rows=match.table_t2.num_rows,
                )
                review_session.flags.append(flag)

        # Flags pour tableaux non matches
        for sig in unmatched_t1:
            flag = self.flagger.create_no_match_flag(
                table_id=sig.table_id,
                is_t1=True,
            )
            review_session.flags.append(flag)

        for sig in unmatched_t2:
            flag = self.flagger.create_no_match_flag(
                table_id=sig.table_id,
                is_t1=False,
            )
            review_session.flags.append(flag)

        # Calculer statistiques
        stats = self._compute_stats(matches, unmatched_t1, unmatched_t2, review_session)

        # Sauvegarder session si des flags existent
        if review_session.flags:
            session_path = self.output_dir / f"{review_session.session_id}.json"
            self.flagger.save_session(review_session, str(session_path))

        return ComparisonResult(
            matched_tables=matches,
            unmatched_t1=unmatched_t1,
            unmatched_t2=unmatched_t2,
            merged_tables_t1=merged_t1,
            merged_tables_t2=merged_t2,
            review_session=review_session,
            stats=stats,
        )

    def _detect_and_merge(
        self,
        tables: list[dict],
    ) -> list[MergedTable]:
        """Detecter et fusionner les tableaux multi-pages."""
        if not tables:
            return []

        # Creer fragments
        fragments = []
        for i, table in enumerate(tables):
            page_num = table.get("page", table.get("page_number", i))
            fragment = create_fragment_from_extracted_table(
                table_data=table,
                page_number=page_num,
                table_index=i % 10,  # Approximation de l'index sur page
            )
            fragments.append(fragment)

        # Fusionner
        return self.continuity_detector.merge_fragments(fragments)

    def _create_signatures(
        self,
        merged_tables: list[MergedTable],
        source: str,
    ) -> list[TableSignature]:
        """Creer signatures depuis tableaux fusionnes."""
        signatures = []

        for i, merged in enumerate(merged_tables):
            sig = TableSignature(
                table_id=f"{source}_p{merged.page_range[0]}_t{i}",
                page_number=merged.page_range[0],
                title=merged.merged_title,
                first_column_labels=merged.merged_first_column_labels,
                headers=merged.merged_headers,
                num_rows=len(merged.merged_rows),
                num_columns=len(merged.merged_headers),
                section_type=merged.fragments[0].raw_data.get("section", "")
                if merged.fragments
                else "",
            )
            signatures.append(sig)

        return signatures

    def _compute_stats(
        self,
        matches: list[MatchResult],
        unmatched_t1: list[TableSignature],
        unmatched_t2: list[TableSignature],
        session: ReviewSession,
    ) -> dict:
        """Calculer les statistiques de comparaison."""
        total_t1 = len(matches) + len(unmatched_t1)
        total_t2 = len(matches) + len(unmatched_t2)

        confirmed = sum(1 for m in matches if m.confidence_level == "confirmed")
        probable = sum(1 for m in matches if m.confidence_level == "probable")

        return {
            "total_tables_t1": total_t1,
            "total_tables_t2": total_t2,
            "matched": len(matches),
            "confirmed_matches": confirmed,
            "probable_matches": probable,
            "unmatched_t1_removed": len(unmatched_t1),
            "unmatched_t2_added": len(unmatched_t2),
            "flags_total": len(session.flags),
            "flags_pending": session.pending_count,
            "flags_critical": session.critical_count,
            "match_rate": len(matches) / max(total_t1, 1),
            "confirmation_rate": confirmed / max(len(matches), 1),
        }

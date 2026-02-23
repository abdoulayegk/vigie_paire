"""
Detecteur de changements principal qui orchestre tous les composants de comparaison.
Fournit une interface unifiee pour detecter les changements entre les rapports trimestriels.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path

from .table_comparator import TableComparator, TableChange
from .text_comparator import TextComparator, TextChange
from .footnote_comparator import FootnoteComparator, FootnoteChange
from .noise_filter import NoiseFilter, ChangeQualifier
from vigilance.utils.footnotes_utils import footnotes_list_to_dict

logger = logging.getLogger(__name__)
UNKNOWN_SECTIONS = {"", "unknown", "unknown_section"}

try:
    from vigilance.extraction.section_taxonomy import canonicalize_section
except Exception:
    canonicalize_section = None


@dataclass
class Change:
    """Unified change representation."""

    change_id: str
    bank_code: str
    from_quarter: str
    to_quarter: str
    year: int
    change_type: str  # "table", "text", "footnote"
    category: str  # "REGULATORY", "RISK_EMERGING", "ESG", "INDICATOR", "OTHER"
    significance: str  # "MAJOR", "MODERATE", "MINOR"
    description: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    section: Optional[str] = None
    page_number: int = 0
    desjardins_relevance: str = "MEDIUM"
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Type metier EDTF/AMF: IFC, RG, PB
    type_metier: Optional[str] = None

    def to_dict(self) -> dict:
        from vigilance.utils.type_metier import compute_type_metier

        d = asdict(self)
        if d.get("type_metier") is None:
            d["type_metier"] = compute_type_metier(
                self.section, self.metadata.get("table_change_type", self.change_type)
            )
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @property
    def text_for_embedding(self) -> str:
        """Generate text suitable for vector embedding."""
        parts = [
            self.description,
            self.category,
            self.section or "",
            " ".join(self.keywords),
            self.new_content[:500] if self.new_content else "",
        ]
        return " ".join(filter(None, parts))


@dataclass
class ComparisonResult:
    """Result of comparing two quarterly reports."""

    bank_code: str
    bank_name: str
    from_quarter: str
    to_quarter: str
    year: int
    total_changes: int
    major_changes: int
    changes: list[Change]
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bank_code": self.bank_code,
            "bank_name": self.bank_name,
            "from_quarter": self.from_quarter,
            "to_quarter": self.to_quarter,
            "year": self.year,
            "total_changes": self.total_changes,
            "major_changes": self.major_changes,
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def get_changes_by_category(self, category: str) -> list[Change]:
        """Get changes filtered by category."""
        return [c for c in self.changes if c.category == category]

    def get_changes_by_significance(self, significance: str) -> list[Change]:
        """Get changes filtered by significance."""
        return [c for c in self.changes if c.significance == significance]


class ChangeDetector:
    """
    Main change detection engine.
    Orchestrates table, text, and footnote comparison with noise filtering.
    """

    def __init__(self, filter_noise: bool = True, min_significance: str = "MINOR"):
        """
        Initialize change detector.

        Args:
            filter_noise: Enable noise filtering
            min_significance: Minimum significance level to include
        """
        self.filter_noise = filter_noise
        self.min_significance = min_significance

        self.table_comparator = TableComparator()
        self.text_comparator = TextComparator()
        self.footnote_comparator = FootnoteComparator()
        self.noise_filter = NoiseFilter()
        self.qualifier = ChangeQualifier()

        self._change_counter = 0

    def compare_documents(
        self, doc1: dict, doc2: dict, bank_code: str, bank_name: str = ""
    ) -> ComparisonResult:
        """
        Compare two extracted documents and detect all changes.

        Args:
            doc1: Extracted document from older quarter
            doc2: Extracted document from newer quarter
            bank_code: Bank identifier
            bank_name: Full bank name

        Returns:
            ComparisonResult with all detected changes
        """
        from_quarter = doc1.get("quarter", "t1")
        to_quarter = doc2.get("quarter", "t2")
        year = doc2.get("year", 2025)

        logger.info(f"Comparing {bank_code} {from_quarter} -> {to_quarter} {year}")

        all_changes = []

        # Compare tables
        table_changes = self._compare_all_tables(
            doc1, doc2, bank_code, from_quarter, to_quarter, year
        )
        all_changes.extend(table_changes)

        # Compare text content
        text_changes = self._compare_text_content(
            doc1, doc2, bank_code, from_quarter, to_quarter, year
        )
        all_changes.extend(text_changes)

        # Filter noise if enabled
        if self.filter_noise:
            relevant, _ = self.noise_filter.filter_changes(all_changes)
            all_changes = relevant

        # Filter by minimum significance
        all_changes = self._filter_by_significance(all_changes)

        # Qualify changes for Desjardins relevance
        for change in all_changes:
            qualification = self.qualifier.qualify_change(change)
            change.desjardins_relevance = qualification.get("desjardins_relevance", "MEDIUM")

        # Calculate summary
        summary = self._generate_summary(all_changes)

        return ComparisonResult(
            bank_code=bank_code,
            bank_name=bank_name or bank_code.upper(),
            from_quarter=from_quarter,
            to_quarter=to_quarter,
            year=year,
            total_changes=len(all_changes),
            major_changes=len([c for c in all_changes if c.significance == "MAJOR"]),
            changes=all_changes,
            summary=summary,
        )

    def _compare_all_tables(
        self, doc1: dict, doc2: dict, bank_code: str, from_quarter: str, to_quarter: str, year: int
    ) -> list[Change]:
        """Compare all tables between documents."""
        changes = []

        tables1 = doc1.get("all_tables", [])
        tables2 = doc2.get("all_tables", [])

        # Create table lookup by strict section + approximate key.
        tables1_dict: dict[str, dict] = {}
        tables2_dict: dict[str, dict] = {}
        for table in tables1:
            section = self._normalize_section(table.get("section"))
            key = f"{section}::{self._get_table_key(table)}"
            tables1_dict[key] = table
        for table in tables2:
            section = self._normalize_section(table.get("section"))
            key = f"{section}::{self._get_table_key(table)}"
            tables2_dict[key] = table

        # Compare matching tables
        for key, table2 in tables2_dict.items():
            table1 = tables1_dict.get(key)

            if table1:
                section_t1 = self._normalize_section(table1.get("section"))
                section_t2 = self._normalize_section(table2.get("section"))
                if (
                    section_t1 in UNKNOWN_SECTIONS
                    or section_t2 in UNKNOWN_SECTIONS
                    or section_t1 != section_t2
                ):
                    changes.append(
                        Change(
                            change_id=self._generate_id(),
                            bank_code=bank_code,
                            from_quarter=from_quarter,
                            to_quarter=to_quarter,
                            year=year,
                            change_type="table",
                            category="INDICATOR",
                            significance="MODERATE",
                            description=f"Nouveau tableau: {table2.get('title', 'Sans titre')}",
                            new_content=str(table2.get("headers", [])),
                            section=section_t2,
                            page_number=table2.get("page_number", 0),
                            metadata={
                                "match_reason": (
                                    "unknown_section"
                                    if section_t1 in UNKNOWN_SECTIONS
                                    or section_t2 in UNKNOWN_SECTIONS
                                    else "cross_section_forbidden"
                                )
                            },
                        )
                    )
                    changes.append(
                        Change(
                            change_id=self._generate_id(),
                            bank_code=bank_code,
                            from_quarter=from_quarter,
                            to_quarter=to_quarter,
                            year=year,
                            change_type="table",
                            category="INDICATOR",
                            significance="MODERATE",
                            description=f"Tableau supprimé: {table1.get('title', 'Sans titre')}",
                            old_content=str(table1.get("headers", [])),
                            section=section_t1,
                            page_number=table1.get("page_number", 0),
                            metadata={
                                "match_reason": (
                                    "unknown_section"
                                    if section_t1 in UNKNOWN_SECTIONS
                                    or section_t2 in UNKNOWN_SECTIONS
                                    else "cross_section_forbidden"
                                )
                            },
                        )
                    )
                    continue
                # Compare the tables
                table_changes = self.table_comparator.compare_tables(
                    self._table_to_dict(table1),
                    self._table_to_dict(table2),
                    table1.get("table_id", ""),
                    table2.get("table_id", ""),
                )

                for tc in table_changes:
                    change = self._convert_table_change(
                        tc, bank_code, from_quarter, to_quarter, year
                    )
                    changes.append(change)

                # Compare footnotes for matched tables
                footnotes1 = footnotes_list_to_dict(table1.get("footnotes", []) or [])
                footnotes2 = footnotes_list_to_dict(table2.get("footnotes", []) or [])
                if footnotes1 or footnotes2:
                    fn_changes = self.footnote_comparator.compare_footnotes(
                        footnotes1, footnotes2, table1.get("table_id")
                    )
                    for fc in fn_changes:
                        changes.append(
                            self._convert_footnote_change(
                                fc,
                                bank_code,
                                from_quarter,
                                to_quarter,
                                year,
                                page_number=table1.get("page_number", 0),
                                section=table1.get("section"),
                            )
                        )
            else:
                # New table
                section = self._normalize_section(table2.get("section"))
                change = Change(
                    change_id=self._generate_id(),
                    bank_code=bank_code,
                    from_quarter=from_quarter,
                    to_quarter=to_quarter,
                    year=year,
                    change_type="table",
                    category="INDICATOR",
                    significance="MODERATE",
                    description=f"Nouveau tableau: {table2.get('title', 'Sans titre')}",
                    new_content=str(table2.get("headers", [])),
                    section=section,
                    page_number=table2.get("page_number", 0),
                    metadata={
                        "match_reason": (
                            "unknown_section" if section in UNKNOWN_SECTIONS else "no_candidate_same_section"
                        )
                    },
                )
                changes.append(change)

        # Check for removed tables
        for key, table1 in tables1_dict.items():
            if key not in tables2_dict:
                section = self._normalize_section(table1.get("section"))
                change = Change(
                    change_id=self._generate_id(),
                    bank_code=bank_code,
                    from_quarter=from_quarter,
                    to_quarter=to_quarter,
                    year=year,
                    change_type="table",
                    category="INDICATOR",
                    significance="MODERATE",
                    description=f"Tableau supprimé: {table1.get('title', 'Sans titre')}",
                    old_content=str(table1.get("headers", [])),
                    section=section,
                    page_number=table1.get("page_number", 0),
                    metadata={
                        "match_reason": (
                            "unknown_section" if section in UNKNOWN_SECTIONS else "no_candidate_same_section"
                        )
                    },
                )
                changes.append(change)

        return changes

    def _compare_text_content(
        self, doc1: dict, doc2: dict, bank_code: str, from_quarter: str, to_quarter: str, year: int
    ) -> list[Change]:
        """Compare text content between documents."""
        changes = []

        text1 = doc1.get("metadata", {}).get("text_content", "")
        text2 = doc2.get("metadata", {}).get("text_content", "")

        if text1 and text2:
            text_changes = self.text_comparator.compare_text(text1, text2, "general")

            for tc in text_changes:
                change = self._convert_text_change(tc, bank_code, from_quarter, to_quarter, year)
                changes.append(change)

        return changes

    def _get_table_key(self, table: dict) -> str:
        """Generate a key for matching tables across documents."""
        title = table.get("title", "")
        headers = table.get("headers", [])

        # Use title if available
        if title:
            return title.lower().strip()[:100]

        # Use first few headers
        if headers:
            if isinstance(headers[0], list):
                header_str = " ".join(str(h) for h in headers[0][:3])
            else:
                header_str = " ".join(str(h) for h in headers[:3])
            return header_str.lower().strip()[:100]

        return f"table_{table.get('page_number', 0)}"

    def _table_to_dict(self, table: dict) -> dict:
        """Convert table to format expected by comparator."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        # Handle nested structure
        if rows and isinstance(rows[0], dict):
            rows = [[cell.get("value", "") for cell in row] for row in rows]

        return {
            "headers": headers,
            "rows": rows,
            "page_number": table.get("page_number", 0),
            "section": self._normalize_section(table.get("section")),
            "first_column_indicators": table.get("first_column_indicators", []),
        }

    def _normalize_section(self, section: Any) -> str:
        raw = str(section or "").strip()
        if not raw:
            return "unknown_section"
        if canonicalize_section is not None:
            try:
                normalized = canonicalize_section(raw)
                if normalized:
                    return normalized
            except Exception:
                pass
        fallback = raw.lower().strip().replace(" ", "_")
        return fallback or "unknown_section"

    def _convert_table_change(
        self, tc: TableChange, bank_code: str, from_quarter: str, to_quarter: str, year: int
    ) -> Change:
        """Convert TableChange to unified Change."""
        from vigilance.utils.type_metier import compute_type_metier

        # Derive type_metier: REGULATORY->IFC, RISK_EMERGING/ESG->RG, structural->PB
        category = tc.category or "OTHER"
        if category == "REGULATORY":
            type_metier = "IFC"
        elif category in ("RISK_EMERGING", "ESG"):
            type_metier = "RG"
        else:
            type_metier = compute_type_metier(None, tc.change_type)

        return Change(
            change_id=self._generate_id(),
            bank_code=bank_code,
            from_quarter=from_quarter,
            to_quarter=to_quarter,
            year=year,
            change_type="table",
            category=category,
            significance=tc.significance,
            description=tc.description,
            old_content=tc.old_value,
            new_content=tc.new_value,
            page_number=tc.page_number,
            type_metier=type_metier,
            metadata={
                "table_id": tc.table_id,
                "row_identifier": tc.row_identifier,
                "column_identifier": tc.column_identifier,
                "table_change_type": tc.change_type,
            },
        )

    def _convert_text_change(
        self, tc: TextChange, bank_code: str, from_quarter: str, to_quarter: str, year: int
    ) -> Change:
        """Convert TextChange to unified Change."""
        from vigilance.utils.type_metier import compute_type_metier

        type_metier = compute_type_metier(tc.section, "modifie")
        return Change(
            change_id=self._generate_id(),
            bank_code=bank_code,
            from_quarter=from_quarter,
            to_quarter=to_quarter,
            year=year,
            change_type="text",
            category=tc.category,
            significance=tc.significance,
            description=tc.description,
            old_content=tc.old_text,
            new_content=tc.new_text,
            section=tc.section,
            page_number=tc.page_number,
            type_metier=type_metier,
            keywords=tc.keywords_matched or [],
        )

    def _convert_footnote_change(
        self,
        fc: FootnoteChange,
        bank_code: str,
        from_quarter: str,
        to_quarter: str,
        year: int,
        page_number: int = 0,
        section: Optional[str] = None,
    ) -> Change:
        """Convert FootnoteChange to unified Change."""
        from vigilance.utils.type_metier import compute_type_metier

        type_metier = compute_type_metier(section, fc.change_type)
        return Change(
            change_id=self._generate_id(),
            bank_code=bank_code,
            from_quarter=from_quarter,
            to_quarter=to_quarter,
            year=year,
            change_type="footnote",
            category=fc.category,
            significance=fc.significance,
            description=fc.description,
            old_content=fc.old_text,
            new_content=fc.new_text,
            section=section,
            page_number=page_number,
            type_metier=type_metier,
            metadata={
                "footnote_ref": fc.footnote_ref,
                "table_id": fc.table_id,
                "footnote_change_type": fc.change_type,
            },
        )

    def _generate_id(self) -> str:
        """Generate unique change ID."""
        self._change_counter += 1
        return f"CHG_{self._change_counter:06d}"

    def _filter_by_significance(self, changes: list[Change]) -> list[Change]:
        """Filter changes by minimum significance."""
        significance_order = {"MAJOR": 3, "MODERATE": 2, "MINOR": 1}
        min_level = significance_order.get(self.min_significance, 1)

        return [c for c in changes if significance_order.get(c.significance, 1) >= min_level]

    def _generate_summary(self, changes: list[Change]) -> dict:
        """Generate summary statistics for changes."""
        summary = {"by_category": {}, "by_significance": {}, "by_type": {}, "top_keywords": []}

        # Count by category
        for change in changes:
            cat = change.category
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

        # Count by significance
        for change in changes:
            sig = change.significance
            summary["by_significance"][sig] = summary["by_significance"].get(sig, 0) + 1

        # Count by type
        for change in changes:
            t = change.change_type
            summary["by_type"][t] = summary["by_type"].get(t, 0) + 1

        # Collect keywords
        all_keywords = []
        for change in changes:
            all_keywords.extend(change.keywords)

        # Top keywords by frequency
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        summary["top_keywords"] = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return summary


def detect_changes(doc1: dict, doc2: dict, bank_code: str) -> ComparisonResult:
    """
    Convenience function to detect changes between documents.

    Args:
        doc1: Older document
        doc2: Newer document
        bank_code: Bank identifier

    Returns:
        ComparisonResult with all changes
    """
    detector = ChangeDetector()
    return detector.compare_documents(doc1, doc2, bank_code)

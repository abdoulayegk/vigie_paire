"""
Extracteur de tableaux avec capacites avancees pour les rapports bancaires.
Gere les structures de tableaux complexes, les cellules fusionnees et les notes de bas de page.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    """Represente une cellule dans un tableau."""

    value: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    is_numeric: bool = False
    footnote_refs: list[str] = field(default_factory=list)


@dataclass
class EnhancedTable:
    """Representation amelioree d'un tableau avec metadonnees completes."""

    table_id: str
    page_number: int
    title: str | None
    headers: list[list[TableCell]]  # Peut avoir des en-tetes multi-lignes
    data_rows: list[list[TableCell]]
    footnotes: dict[str, str]  # reference -> texte
    section_name: str | None = None
    phase: int | None = None
    table_type: str | None = None  # "indicateur", "ratio", "risque", etc.

    @property
    def row_count(self) -> int:
        return len(self.data_rows)

    @property
    def col_count(self) -> int:
        if self.headers:
            return len(self.headers[0])
        if self.data_rows:
            return len(self.data_rows[0])
        return 0

    def get_column_values(self, col_index: int) -> list[str]:
        """Obtenir toutes les valeurs d'une colonne."""
        return [row[col_index].value for row in self.data_rows if col_index < len(row)]

    def get_row_values(self, row_index: int) -> list[str]:
        """Obtenir toutes les valeurs d'une ligne."""
        if row_index < len(self.data_rows):
            return [cell.value for cell in self.data_rows[row_index]]
        return []

    def to_dict(self) -> dict:
        """Convertir en dictionnaire pour la serialisation JSON."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "title": self.title,
            "headers": [[cell.value for cell in row] for row in self.headers],
            "data_rows": [[cell.value for cell in row] for row in self.data_rows],
            "footnotes": self.footnotes,
            "section_name": self.section_name,
            "phase": self.phase,
            "table_type": self.table_type,
            "row_count": self.row_count,
            "col_count": self.col_count,
        }


class TableExtractor:
    """
    Extracteur de tableaux avance pour les rapports bancaires.
    Gere les mises en page complexes courantes dans les divulgations financieres.
    """

    # Patterns pour identifier les types de tableaux
    TABLE_TYPE_PATTERNS = {
        "ratio": [
            r"ratio\s+(de\s+)?fonds\s+propres",
            r"ratio\s+CET1",
            r"ratio\s+de\s+levier",
            r"ratio\s+de\s+liquidite",
            r"LCR|NSFR",
        ],
        "indicateur": [
            r"indicateur",
            r"mesure\s+cle",
            r"KPI",
            r"points\s+saillants",
        ],
        "exposition_risque": [
            r"exposition\s+au\s+risque",
            r"provision\s+pour\s+pertes",
            r"creances\s+douteuses",
            r"actifs\s+ponderes",
        ],
        "capital": [
            r"fonds\s+propres",
            r"capital\s+reglementaire",
            r"actions\s+ordinaires",
            r"capitaux\s+propres",
        ],
    }

    # Indicateurs importants connus dans les rapports bancaires
    KEY_INDICATORS = [
        "Ratio CET1",
        "Ratio de fonds propres",
        "Ratio de levier",
        "LCR",
        "NSFR",
        "RCP",
        "ROE",
        "ROA",
        "Produits",
        "Resultat net",
        "Resultat dilue par action",
        "Provision pour pertes",
        "Marge d'interets nette",
        "Actifs ponderes",
        "Total des actifs",
        "Depots",
    ]

    def __init__(self):
        """Initialiser l'extracteur de tableaux."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compiler les patterns regex."""
        self.type_patterns = {}
        for table_type, patterns in self.TABLE_TYPE_PATTERNS.items():
            self.type_patterns[table_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def extract_from_docling_table(self, docling_table: Any, page_num: int = 0) -> EnhancedTable:
        """
        Convertir un objet tableau Docling en EnhancedTable.

        Args:
            docling_table: Objet tableau de Docling
            page_num: Numero de page

        Returns:
            EnhancedTable avec structure complete
        """
        title = None
        if hasattr(docling_table, "caption"):
            title = str(docling_table.caption) if docling_table.caption else None

        headers = []
        data_rows = []

        try:
            if hasattr(docling_table, "data") and docling_table.data:
                grid = docling_table.data.grid

                if grid and len(grid) > 0:
                    # Premiere(s) ligne(s) comme en-tetes
                    header_row = [
                        TableCell(
                            value=str(cell.text) if hasattr(cell, "text") else str(cell),
                            is_header=True,
                            is_numeric=self._is_numeric(
                                str(cell.text) if hasattr(cell, "text") else str(cell)
                            ),
                        )
                        for cell in grid[0]
                    ]
                    headers.append(header_row)

                    # Lignes de donnees
                    for row in grid[1:]:
                        data_row = [
                            TableCell(
                                value=str(cell.text) if hasattr(cell, "text") else str(cell),
                                is_numeric=self._is_numeric(
                                    str(cell.text) if hasattr(cell, "text") else str(cell)
                                ),
                                footnote_refs=self._extract_footnote_refs(
                                    str(cell.text) if hasattr(cell, "text") else str(cell)
                                ),
                            )
                            for cell in row
                        ]
                        data_rows.append(data_row)
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse du tableau Docling: {e}")

        # Extraire les notes de bas de page
        footnotes = {}
        if hasattr(docling_table, "footnotes"):
            for i, fn in enumerate(docling_table.footnotes):
                footnotes[str(i + 1)] = str(fn)

        table_id = f"tableau_p{page_num}_{id(docling_table) % 1000}"

        enhanced = EnhancedTable(
            table_id=table_id,
            page_number=page_num,
            title=title,
            headers=headers,
            data_rows=data_rows,
            footnotes=footnotes,
        )

        # Classifier le type de tableau
        enhanced.table_type = self._classify_table_type(enhanced)

        return enhanced

    def extract_from_raw_data(
        self,
        headers: list[list[str]],
        rows: list[list[str]],
        page_num: int = 0,
        title: str | None = None,
        footnotes: dict[str, str] | None = None,
    ) -> EnhancedTable:
        """
        Creer un EnhancedTable a partir de donnees brutes.

        Args:
            headers: Liste des lignes d'en-tete
            rows: Liste des lignes de donnees
            page_num: Numero de page
            title: Titre du tableau (optionnel)
            footnotes: Dictionnaire des notes de bas de page (optionnel)

        Returns:
            EnhancedTable
        """
        # Convertir les en-tetes bruts
        enhanced_headers = []
        for header_row in headers:
            enhanced_headers.append(
                [
                    TableCell(
                        value=str(cell) if cell else "",
                        is_header=True,
                        is_numeric=self._is_numeric(str(cell) if cell else ""),
                    )
                    for cell in header_row
                ]
            )

        # Si headers est une liste plate, l'envelopper
        if headers and not isinstance(headers[0], list):
            enhanced_headers = [
                [
                    TableCell(value=str(h), is_header=True, is_numeric=self._is_numeric(str(h)))
                    for h in headers
                ]
            ]

        # Convertir les lignes brutes
        enhanced_rows = []
        for row in rows:
            enhanced_rows.append(
                [
                    TableCell(
                        value=str(cell) if cell else "",
                        is_numeric=self._is_numeric(str(cell) if cell else ""),
                        footnote_refs=self._extract_footnote_refs(str(cell) if cell else ""),
                    )
                    for cell in row
                ]
            )

        table_id = f"tableau_p{page_num}_{len(rows)}"

        enhanced = EnhancedTable(
            table_id=table_id,
            page_number=page_num,
            title=title,
            headers=enhanced_headers,
            data_rows=enhanced_rows,
            footnotes=footnotes or {},
        )

        enhanced.table_type = self._classify_table_type(enhanced)

        return enhanced

    def _is_numeric(self, value: str) -> bool:
        """Verifier si une valeur est numerique (incluant pourcentages et devises)."""
        if not value:
            return False

        # Supprimer le formatage courant
        cleaned = re.sub(r"[,$%\s\u00a0()]", "", value)
        cleaned = cleaned.replace("M$", "").replace("G$", "")

        try:
            float(cleaned.replace(",", "."))
            return True
        except ValueError:
            return False

    def _extract_footnote_refs(self, value: str) -> list[str]:
        """Extraire les references de notes de bas de page d'une valeur de cellule."""
        refs = []

        # Patterns de notes de bas de page courants: 1 2 3 4 5, (1), [1], *
        superscript_pattern = re.compile(r"[1234567890]+")
        bracket_pattern = re.compile(r"\((\d+)\)|\[(\d+)\]")
        asterisk_pattern = re.compile(r"\*+")

        for match in superscript_pattern.findall(value):
            # Convertir exposant en nombres reguliers
            superscript_map = {
                "1": "1",
                "2": "2",
                "3": "3",
                "4": "4",
                "5": "5",
                "6": "6",
                "7": "7",
                "8": "8",
                "9": "9",
                "0": "0",
            }
            ref = "".join(superscript_map.get(c, c) for c in match)
            refs.append(ref)

        for match in bracket_pattern.findall(value):
            ref = match[0] or match[1]
            if ref:
                refs.append(ref)

        for match in asterisk_pattern.findall(value):
            refs.append("*" * len(match))

        return refs

    def _classify_table_type(self, table: EnhancedTable) -> str | None:
        """Classifier le type de tableau base sur le contenu."""
        # Combiner titre et premieres lignes pour la classification
        text_to_check = table.title or ""

        for row in table.headers:
            text_to_check += " " + " ".join(cell.value for cell in row)

        for row in table.data_rows[:5]:
            text_to_check += " " + " ".join(cell.value for cell in row)

        # Verifier contre les patterns
        for table_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if pattern.search(text_to_check):
                    return table_type

        return None

    def identify_key_indicators(self, table: EnhancedTable) -> list[dict]:
        """
        Identifier les indicateurs bancaires cles dans un tableau.

        Args:
            table: EnhancedTable a analyser

        Returns:
            Liste des indicateurs identifies avec leurs valeurs
        """
        indicators = []

        for row_idx, row in enumerate(table.data_rows):
            if not row:
                continue

            first_cell = row[0].value.strip()

            for indicator in self.KEY_INDICATORS:
                if indicator.lower() in first_cell.lower():
                    # Indicateur trouve - extraire les valeurs des autres colonnes
                    values = [cell.value for cell in row[1:] if cell.value.strip()]

                    indicators.append(
                        {
                            "indicateur": first_cell,
                            "cle_correspondante": indicator,
                            "valeurs": values,
                            "index_ligne": row_idx,
                            "page": table.page_number,
                        }
                    )
                    break

        return indicators

    def compare_tables(self, table1: EnhancedTable, table2: EnhancedTable) -> dict:
        """
        Comparer deux tableaux pour trouver les differences structurelles.

        Args:
            table1: Premier tableau (ancien)
            table2: Deuxieme tableau (nouveau)

        Returns:
            Dictionnaire avec les resultats de comparaison
        """
        result = {
            "structure_modifiee": False,
            "nouvelles_lignes": [],
            "lignes_supprimees": [],
            "nouvelles_colonnes": [],
            "colonnes_supprimees": [],
            "changements_valeurs": [],
        }

        # Comparer le nombre de colonnes
        if table1.col_count != table2.col_count:
            result["structure_modifiee"] = True
            if table2.col_count > table1.col_count:
                result["nouvelles_colonnes"] = list(range(table1.col_count, table2.col_count))
            else:
                result["colonnes_supprimees"] = list(range(table2.col_count, table1.col_count))

        # Comparer le nombre de lignes
        if table1.row_count != table2.row_count:
            result["structure_modifiee"] = True

        # Obtenir les identifiants de lignes (valeurs de la premiere colonne)
        rows1 = {row[0].value.strip(): row for row in table1.data_rows if row}
        rows2 = {row[0].value.strip(): row for row in table2.data_rows if row}

        # Trouver les lignes nouvelles et supprimees
        result["nouvelles_lignes"] = [key for key in rows2.keys() if key and key not in rows1]
        result["lignes_supprimees"] = [key for key in rows1.keys() if key and key not in rows2]

        return result


def extract_tables_from_pdf(pdf_path: str) -> list[EnhancedTable]:
    """
    Fonction utilitaire pour extraire tous les tableaux d'un PDF.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        Liste d'objets EnhancedTable
    """
    from .docling_processor import DoclingProcessor

    processor = DoclingProcessor()
    extractor = TableExtractor()

    # Ceci devra etre appele avec les parametres appropries
    # Pour l'instant, retourner une liste vide comme espace reserve
    logger.info(f"Extraction de tableaux demandee pour: {pdf_path}")
    return []

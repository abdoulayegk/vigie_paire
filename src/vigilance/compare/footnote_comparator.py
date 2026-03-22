"""
Comparateur de notes de bas de page entre deux trimestres consécutifs.

Les notes de bas de page des rapports bancaires contiennent souvent des
informations critiques qui ne figurent pas dans les chiffres eux-mêmes :
changements de méthodologie de calcul, nouvelles exigences réglementaires
(Bâle III, BSIF, IFRS 9), révisions de définitions d'indicateurs, etc.

Ce module détecte trois types de changements entre les notes du trimestre
précédent (T1) et celles du trimestre courant (T2) :

- **Nouvelle note** (``new_footnote``) : note présente en T2 mais absente en T1,
  sans note similaire en T1 (similarité textuelle < seuil).
- **Note supprimée** (``removed_footnote``) : note présente en T1 mais absente
  en T2, sans note similaire en T2.
- **Note modifiée** (``modified_footnote``) : note présente dans les deux
  trimestres avec la même référence, mais dont le contenu a changé
  (similarité < seuil).

Chaque changement est classifié par catégorie (``REGULATORY``, ``INDICATOR``,
``OTHER``) et par niveau de significativité (``MAJOR``, ``MODERATE``, ``MINOR``)
en fonction des mots-clés détectés dans le texte.

Utilisation typique
-------------------
.. code-block:: python

    from vigilance.compare.footnote_comparator import FootnoteComparator

    comparator = FootnoteComparator()
    changes = comparator.compare_footnotes(
        footnotes_t1={"1": "Calculé selon Bâle III."},
        footnotes_t2={"1": "Calculé selon Bâle IV."},
        table_id="table_28a",
    )
    for change in changes:
        print(change.change_type, change.significance, change.description)
"""

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from vigilance.utils.text_normalize_base import normalize_text_base

logger = logging.getLogger(__name__)


@dataclass
class FootnoteChange:
    """Représente un changement détecté dans une note de bas de page.

    Attributs
    ---------
    change_type:
        Type de changement : ``"new_footnote"``, ``"removed_footnote"`` ou
        ``"modified_footnote"``.
    footnote_ref:
        Référence normalisée de la note (ex. ``"1"``, ``"a"``, ``"note1"``).
    table_id:
        Identifiant du tableau auquel appartient la note (peut être ``None``).
    description:
        Message lisible décrivant le changement (ex. ``"Nouvelle note de bas
        de page (1)"``).
    old_text:
        Texte de la note en T1 (tronqué à 500 caractères). ``None`` pour les
        nouvelles notes.
    new_text:
        Texte de la note en T2 (tronqué à 500 caractères). ``None`` pour les
        notes supprimées.
    significance:
        Niveau de significativité : ``"MAJOR"``, ``"MODERATE"`` ou ``"MINOR"``.
    category:
        Catégorie du changement : ``"REGULATORY"`` (mots-clés réglementaires
        détectés), ``"INDICATOR"`` (mots-clés méthodologiques), ``"OTHER"``.
    """

    change_type: str  # "new_footnote", "removed_footnote", "modified_footnote"
    footnote_ref: str
    table_id: Optional[str]
    description: str
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    significance: str = "MINOR"
    category: str = "OTHER"

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "footnote_ref": self.footnote_ref,
            "table_id": self.table_id,
            "description": self.description,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "significance": self.significance,
            "category": self.category,
        }


class FootnoteComparator:
    """Compare les notes de bas de page entre deux rapports trimestriels.

    Les notes contiennent souvent des informations critiques sur les changements
    méthodologiques et les mises à jour réglementaires. Cette classe détecte les
    ajouts, suppressions et modifications de notes en utilisant une comparaison
    textuelle fuzzy pour gérer les légères variations de formulation.

    Attributs de classe
    -------------------
    METHODOLOGY_KEYWORDS:
        Mots-clés indiquant un changement de méthodologie de calcul.
    REGULATORY_KEYWORDS:
        Mots-clés indiquant une mise à jour réglementaire (Bâle, BSIF, etc.).

    Paramètres du constructeur
    --------------------------
    similarity_threshold:
        Seuil de similarité textuelle (0..1) en dessous duquel une note est
        considérée comme modifiée. Défaut : 0.8. Un seuil plus élevé de 0.92
        est appliqué automatiquement pour les notes courtes (< 50 caractères)
        afin de réduire les faux positifs.
    """

    METHODOLOGY_KEYWORDS = [
        "méthode",
        "méthodologie",
        "calcul",
        "formule",
        "changement",
        "modification",
        "révision",
        "ajustement",
    ]

    REGULATORY_KEYWORDS = [
        "bâle",
        "bsif",
        "réglementaire",
        "norme",
        "exigence",
        "conformité",
        "ligne directrice",
    ]

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self._short_footnote_similarity_threshold = 0.92

    def compare_footnotes(
        self, footnotes1: dict[str, str], footnotes2: dict[str, str], table_id: Optional[str] = None
    ) -> list[FootnoteChange]:
        """Comparer deux ensembles de notes et retourner la liste des changements.

        Algorithme
        ----------
        1. Normalise les deux ensembles de notes (clés et textes).
        2. Pour chaque note présente en T2 mais absente en T1, vérifie si une
           note similaire existe en T1 (fuzzy match). Si non → ``new_footnote``.
        3. Pour chaque note présente en T1 mais absente en T2, vérifie si une
           note similaire existe en T2. Si non → ``removed_footnote``.
        4. Pour les notes présentes dans les deux trimestres avec la même
           référence, compare le texte. Si similarité < seuil → ``modified_footnote``.

        Paramètres
        ----------
        footnotes1:
            Dictionnaire ``{référence: texte}`` des notes du trimestre précédent (T1).
        footnotes2:
            Dictionnaire ``{référence: texte}`` des notes du trimestre courant (T2).
        table_id:
            Identifiant du tableau pour contextualiser les changements détectés.

        Retourne
        --------
        Liste de ``FootnoteChange``, potentiellement vide si aucun changement
        n'est détecté.
        """
        changes = []

        norm1, raw1 = self._normalize_footnotes_with_raw(footnotes1)
        norm2, raw2 = self._normalize_footnotes_with_raw(footnotes2)

        for ref, text in norm2.items():
            if ref not in norm1:
                similar_ref = self._find_similar_footnote(text, norm1)
                if not similar_ref:
                    changes.append(
                        self._create_footnote_change(
                            "new_footnote", ref, raw2.get(ref, text), None, table_id
                        )
                    )

        for ref, text in norm1.items():
            if ref not in norm2:
                similar_ref = self._find_similar_footnote(text, norm2)
                if not similar_ref:
                    changes.append(
                        self._create_footnote_change(
                            "removed_footnote", ref, None, raw1.get(ref, text), table_id
                        )
                    )

        for ref in set(norm1.keys()) & set(norm2.keys()):
            text1 = norm1[ref]
            text2 = norm2[ref]
            similarity = SequenceMatcher(None, text1, text2).ratio()
            thresh = (
                self._short_footnote_similarity_threshold
                if max(len(text1), len(text2)) < 50
                else self.similarity_threshold
            )
            if similarity < thresh:
                changes.append(
                    self._create_footnote_change(
                        "modified_footnote",
                        ref,
                        raw2.get(ref, text2),
                        raw1.get(ref, text1),
                        table_id,
                    )
                )

        return changes

    def _normalize_footnotes_with_raw(
        self, footnotes: dict
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Normaliser les notes pour la comparaison tout en conservant le texte brut.

        Retourne un tuple ``(normalized, raw_display)`` où les deux dictionnaires
        sont indexés par la référence normalisée (minuscules, caractères
        non-alphanumériques supprimés). Le dictionnaire ``normalized`` contient
        le texte normalisé via ``normalize_text_base`` (pour la comparaison
        fuzzy), et ``raw_display`` contient le texte brut (pour l'affichage).
        """
        normalized: dict[str, str] = {}
        raw_display: dict[str, str] = {}
        for ref, text in footnotes.items():
            norm_ref = str(ref).strip().lower()
            norm_ref = re.sub(r"[^\w]", "", norm_ref)
            raw = re.sub(r"\s+", " ", str(text).strip())
            norm_text = normalize_text_base(raw)
            if norm_text:
                normalized[norm_ref] = norm_text
                raw_display[norm_ref] = raw
        return normalized, raw_display

    def _normalize_footnotes(self, footnotes: dict) -> dict[str, str]:
        n, _ = self._normalize_footnotes_with_raw(footnotes)
        return n

    def _find_similar_footnote(self, target: str, footnotes: dict[str, str]) -> Optional[str]:
        """Rechercher une note similaire dans un ensemble par comparaison textuelle.

        Utilisé pour détecter les cas où une note a été renommée (changement de
        référence) plutôt que réellement ajoutée ou supprimée. Retourne la
        référence de la note similaire si la similarité dépasse ``similarity_threshold``,
        ou ``None`` si aucune note similaire n'est trouvée.
        """
        target_lower = target.lower()
        for ref, text in footnotes.items():
            similarity = SequenceMatcher(None, target_lower, text.lower()).ratio()
            if similarity >= self.similarity_threshold:
                return ref
        return None

    def _create_footnote_change(
        self,
        change_type: str,
        ref: str,
        new_text: Optional[str],
        old_text: Optional[str],
        table_id: Optional[str],
    ) -> FootnoteChange:
        text_to_check = new_text or old_text or ""
        category = self._classify_footnote(text_to_check)
        significance = self._assess_significance(change_type, category, text_to_check)

        if change_type == "new_footnote":
            desc = f"Nouvelle note de bas de page ({ref})"
        elif change_type == "removed_footnote":
            desc = f"Note de bas de page supprimée ({ref})"
        else:
            desc = f"Note de bas de page modifiée ({ref})"

        return FootnoteChange(
            change_type=change_type,
            footnote_ref=ref,
            table_id=table_id,
            description=desc,
            old_text=old_text[:500] if old_text else None,
            new_text=new_text[:500] if new_text else None,
            significance=significance,
            category=category,
        )

    def _classify_footnote(self, text: str) -> str:
        """Classifier une note par catégorie selon les mots-clés détectés.

        Retourne ``"REGULATORY"`` si des mots-clés réglementaires sont trouvés
        (ex. ``bâle``, ``bsif``, ``norme``), ``"INDICATOR"`` si des mots-clés
        méthodologiques sont trouvés (ex. ``calcul``, ``révision``), ou
        ``"OTHER"`` dans les autres cas.
        """
        text_lower = text.lower()
        for keyword in self.REGULATORY_KEYWORDS:
            if keyword in text_lower:
                return "REGULATORY"
        for keyword in self.METHODOLOGY_KEYWORDS:
            if keyword in text_lower:
                return "INDICATOR"
        return "OTHER"

    def _assess_significance(self, change_type: str, category: str, text: str) -> str:
        """Évaluer le niveau de significativité d'un changement de note.

        Règles appliquées (par ordre de priorité) :
        - Catégorie ``REGULATORY`` + ajout/suppression → ``"MAJOR"``.
        - Catégorie ``REGULATORY`` + modification → ``"MODERATE"``.
        - Texte contenant des mots-clés méthodologiques → ``"MODERATE"``.
        - Nouvelle note longue (> 200 caractères) → ``"MODERATE"``.
        - Tous les autres cas → ``"MINOR"``.
        """
        if category == "REGULATORY":
            return "MAJOR" if change_type != "modified_footnote" else "MODERATE"
        if any(kw in text.lower() for kw in self.METHODOLOGY_KEYWORDS):
            return "MODERATE"
        if change_type == "new_footnote":
            return "MODERATE" if len(text) > 200 else "MINOR"
        return "MINOR"


def compare_footnotes(
    footnotes1: dict, footnotes2: dict, table_id: str = None
) -> list[FootnoteChange]:
    """Comparer deux ensembles de notes via le comparateur standard du module.

    Fonction utilitaire de niveau module qui instancie un ``FootnoteComparator``
    avec les paramètres par défaut et appelle ``compare_footnotes``. Pratique
    pour un usage ponctuel sans avoir à gérer l'instance de la classe.

    Paramètres
    ----------
    footnotes1:
        Dictionnaire ``{référence: texte}`` des notes du trimestre précédent (T1).
    footnotes2:
        Dictionnaire ``{référence: texte}`` des notes du trimestre courant (T2).
    table_id:
        Identifiant du tableau (optionnel, pour contextualiser les changements).

    Retourne
    --------
    Liste de ``FootnoteChange`` décrivant les changements détectés.
    """
    comparator = FootnoteComparator()
    return comparator.compare_footnotes(footnotes1, footnotes2, table_id)

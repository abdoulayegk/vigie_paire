import logging
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrphanMatch:
    """Représente une paire de lignes (T1, T2) identifiée comme un renommage."""

    t1_indicator: str
    t2_indicator: str
    confidence: str  # high, medium
    reasoning: str


class GenAIOrphanMatcher:
    """
    Module spécialisé pour la comparaison sémantique des lignes orphelines.
    N'est appelé QUE pour les lignes qui n'ont pas trouvé de correspondance exacte.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None

        # Initialisation lazy du client
        if self.api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("Package 'openai' manquant. GenAIOrphanMatcher désactivé.")

    def match_orphans(
        self, orphans_t1: List[str], orphans_t2: List[str], context_description: str = ""
    ) -> List[OrphanMatch]:
        """
        Tente d'apparier sémantiquement des lignes orphelines.

        Args:
            orphans_t1: Liste des indicateurs restants en T1 (candidats suppression)
            orphans_t2: Liste des indicateurs restants en T2 (candidats ajout)
            context_description: Contexte optionnel (ex: titre du tableau)

        Returns:
            Liste des paires (t1, t2) identifiées comme renommages.
        """
        # Sécurité : Si pas de client ou listes vides, on ne fait rien.
        if not self.client or not orphans_t1 or not orphans_t2:
            return []

        try:
            return self._query_gpt(orphans_t1, orphans_t2, context_description)
        except Exception as e:
            logger.error(
                f"Erreur GenAIOrphanMatcher: {e}. Repli sur comportement par défaut (pas de matching)."
            )
            return []

    def _query_gpt(self, list_t1: List[str], list_t2: List[str], context: str) -> List[OrphanMatch]:
        system_prompt = """You are a financial analyst specializing in reconciling banking reports.
Your task is to identify SEMANTIC RENAMES between two lists of unmatched row headers (orphans).

CONTEXT:
We have already performed exact string matching. The lines remaining in List 1 and List 2 did not match exactly.
Some might be the SAME logical line but renamed (e.g. "Gross Loans" -> "Loans (Gross)").

INSTRUCTIONS:
1. Compare every item in List 1 against List 2.
2. Identification criteria:
   - High Confidence: Minor rephrasing, abbreviation expansion, language correction, or standard financial synonym.
   - NO MATCH: If the meaning differs significantly or if it's clearly a different financial item.
3. Be conservative. It is better to return NO MATCH than a false positive.

OUTPUT FORMAT (JSON):
{
    "matches": [
        {
            "t1_source": "exact string from list 1",
            "t2_target": "exact string from list 2",
            "confidence": "high" or "medium",
            "reasoning": "brief explanation"
        }
    ]
}"""

        user_prompt = f"""
Context: {context}

List 1 (Potential Removals):
{json.dumps(list_t1, indent=2, ensure_ascii=False)}

List 2 (Potential Additions):
{json.dumps(list_t2, indent=2, ensure_ascii=False)}

Find renames.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        results = []
        for m in data.get("matches", []):
            # Validation post-hallucination : on vérifie que les strings existent vraiment
            if m["t1_source"] in list_t1 and m["t2_target"] in list_t2:
                results.append(
                    OrphanMatch(
                        t1_indicator=m["t1_source"],
                        t2_indicator=m["t2_target"],
                        confidence=m["confidence"],
                        reasoning=m["reasoning"],
                    )
                )

        return results

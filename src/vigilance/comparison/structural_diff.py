import logging
import json
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class StructuralRowComparator:
    """
    Compare la première colonne (indicateurs) de deux tableaux appariés (T1 et T2).
    Utilise GPT-4o pour identifier intelligemment :
    - Les lignes ajoutées
    - Les lignes supprimées
    - Les lignes renommées (traitement sémantique)

    Ignore le réarrangement ou les ajustements textuels mineurs (ex: "Total" vs "Total (M$)").
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-5.2"

    def compare_rows(
        self, t1_rows: List[str], t2_rows: List[str], context_title: str
    ) -> Dict[str, Any]:
        """
        Fonction principale. Compare deux listes d'en-têtes de lignes.
        """
        if not t1_rows and not t2_rows:
            return {"status": "no_data"}

        # system_prompt = """You are a specialized financial analyst tool.
        # Your job is to compare two lists of row headers from banking regulatory reports (Quarter 1 vs Quarter 2).

        # GOAL: Identify STRUCTURAL CHANGES (Added, Removed, Renamed) and ANALYZE them.

        # RULES:
        # 1. Ignore reordering. If a row moved from pos 5 to pos 10, it is NOT a change.
        # 2. Ignore minor formatting/spelling changes (e.g. "Total Capital" == "Total capital ($M)").
        # 3. Identify RENAMES if the meaning is identical but text changed significantly.
        # 4. "Added": A meaningful line present in List 2 but NOT in List 1.
        # 5. "Removed": A meaningful line present in List 1 but NOT in List 2.

        # ANALYSIS CRITERIA:
        # - "new_idea": Boolean. True if the change introduces a significantly new financial concept, product, or risk factor not previously reported. False if it's just a granularity change, a simple rename, or a formatting update.
        # - "justification": String. A brief explanation (in French) of why you classified it as a new idea or declared it as a change. Mention if it looks like a rename or a new requirement.

        # OUTPUT FORMAT (JSON ONLY):
        # {
        #     "changes": [
        #         {
        #             "type": "added" | "removed" | "renamed",
        #             "element": "The row text (or 'Old -> New' for renames)",
        #             "new_idea": true | false,
        #             "justification": "Explanation in French"
        #         }
        #     ]
        # }
        # Return {"changes": []} if no meaningful changes are found.
        # """

        system_prompt = """You are a specialized financial regulatory comparison engine.

You compare TWO LISTS of FIRST-COLUMN row headers extracted from banking regulatory reports
(Quarter 1 vs Quarter 2).

IMPORTANT:
The input lists already represent candidate row headers,
but extraction may have introduced noise (footnotes, column numbers, etc.).
Your job is to detect REAL structural changes only.

--------------------------------------------------
GOAL
--------------------------------------------------

Identify structural changes between List 1 (T1) and List 2 (T2):

- added
- removed
- renamed

Do NOT detect numerical value changes.
Do NOT detect ordering changes.
Only detect true structural row-level changes.

--------------------------------------------------
CRITICAL CLEANING RULES (MANDATORY)
--------------------------------------------------

1) FIRST-COLUMN PURITY
Assume that some rows may contain:
- Footnote numbers
- Cross-column numeric spillovers
- OCR contamination

If a row ends with an isolated number and:
- The number is NOT preceded by a semantic cue (see rule 4),
- The number looks like a footnote or column index,
Then REMOVE that trailing number before comparison.

Examples:
"Divers 4" → "Divers"
"Titrisation de créances hypothécaires 3" → clean to:
"Titrisation de créances hypothécaires"

2) IGNORE FOOTNOTE MARKERS
Remove:
- Superscripts (¹²³⁴)
- Trailing isolated digits
- Bracketed references
- Small numeric markers at end of label

3) IGNORE CROSS-COLUMN CONTAMINATION
If a number appears to come from another column (value column),
ignore it completely in the label comparison.

4) SEMANTIC NUMERIC PROTECTION (DO NOT REMOVE THESE)
If a number is immediately preceded by one of these cues,
it is semantic and MUST be preserved:

Keywords:
- series
- level
- type
- class
- tranche
- category
- group
- phase
- pillar
- ratio
- basel I / II / III / IV
- série
- niveau
- type
- classe
- tranche
- catégorie / categorie
- groupe
- phase
- pilier
- bâle / bale

Examples:
"Series 32" != "Series 47"
"Basel III" != "Basel II"
"Pilier 1" != "Pilier 2"

These are real structural distinctions.

--------------------------------------------------
MATCHING PRINCIPLES
--------------------------------------------------

5) GLOBAL MATCHING (NOT POSITIONAL)
Ignore row order completely.
A row moved from position 5 to 12 is NOT a change.

6) CANONICAL EQUIVALENCE
Treat rows as identical if differences are only:
- capitalization
- punctuation
- minor formatting
- added currency markers (e.g. ($M))
- spacing differences

7) RENAME DETECTION (STRICT)
Classify as "renamed" ONLY if:
- The financial meaning is clearly identical
- The wording changed significantly
- It is not merely formatting or noise

Do NOT overuse rename.

8) ADDED
A meaningful financial concept present in T2
and not found anywhere in T1 after cleaning.

9) REMOVED
A meaningful financial concept present in T1
and not found anywhere in T2 after cleaning.

10) IF UNCERTAIN
If the comparison is ambiguous or unclear,
DO NOT guess.
Do not fabricate changes.

--------------------------------------------------
ANALYSIS CRITERIA
--------------------------------------------------

For each detected change:

- new_idea:
  true  → if it introduces a materially new financial concept,
           regulatory requirement, product, or risk factor.
  false → if it is only structural, regrouping, or rename.

- justification:
  Short explanation in French explaining:
  - Why it is added / removed / renamed
  - Whether it represents a new regulatory or financial concept
  - Or why it is purely structural

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
--------------------------------------------------

{
  "changes": [
    {
      "type": "added" | "removed" | "renamed",
      "element": "Row text OR 'Old -> New' for renames",
      "new_idea": true | false,
      "justification": "Explication en français"
    }
  ]
}

Return:
{"changes": []}

if no meaningful structural changes are found.
"""

        user_prompt = f"""
        Context Table: {context_title}
        
        List 1 (Previous Quarter):
        {json.dumps(t1_rows, indent=2)}
        
        List 2 (Current Quarter):
        {json.dumps(t2_rows, indent=2)}
        
        Return JSON object with "changes" list.
        """

        try:
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
            return json.loads(content)

        except Exception as e:
            logger.error(f"Erreur lors de la comparaison GPT: {e}")
            return {"error": str(e), "changes": []}

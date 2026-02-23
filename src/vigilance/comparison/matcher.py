import logging
from typing import List, Dict, Optional, Any
import numpy as np
from openai import OpenAI
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class TableFingerprint:
    table_id: str
    page_number: int
    title: str
    headers: List[str]
    first_col_sample: List[str]
    embedding: List[float] = None


class SemanticTableMatcher:
    """
    Matches tables across different reports (T1 vs T2) using Semantic Fingerprinting.
    This handles cases where:
    - Tables move pages (p.40 -> p.42)
    - Titles change slightly ("Credit Risk" -> "Credit Risk Exposure")
    - Headers are reformatted
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"

    def _generate_fingerprint_text(self, table_data: Dict[str, Any]) -> str:
        """
        Construct a rich textual representation of the table identity.
        We combine Title + Headers + First few row indicators.
        """
        title = table_data.get("title") or "Untitled Table"
        headers = " | ".join(table_data.get("headers", []))

        # Take first 5 rows of the first column as a 'content signature'
        rows = table_data.get("rows", [])
        first_col_sample = []
        for row in rows[:5]:
            if row and len(row) > 0:
                first_col_sample.append(str(row[0]))

        content_sig = " | ".join(first_col_sample)

        # The fingerprint string
        return f"Title: {title}. Headers: {headers}. Content: {content_sig}"

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of strings."""
        if not texts:
            return []
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]

    def create_fingerprints(self, tables: List[Dict[str, Any]]) -> List[TableFingerprint]:
        """Process a list of Docling-extracted tables into Fingerprints."""
        fingerprints = []
        texts = []

        for tab in tables:
            # Prepare metadata
            rows = tab.get("rows", [])
            first_col = [r[0] for r in rows if r][:5]

            fp = TableFingerprint(
                table_id=tab.get("table_id", "unknown"),
                page_number=tab.get("page_number", 0),
                title=tab.get("title") or "",
                headers=tab.get("headers", []),
                first_col_sample=first_col,
            )
            fingerprints.append(fp)
            texts.append(self._generate_fingerprint_text(tab))

        # Batch embed
        if texts:
            embeddings = self._get_embeddings_batch(texts)
            for i, emb in enumerate(embeddings):
                fingerprints[i].embedding = emb

        return fingerprints

    def match_tables(
        self,
        t1_tables: List[Dict[str, Any]],
        t2_tables: List[Dict[str, Any]],
        threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Main function: Matches T1 tables to T2 tables.
        Returns visual diff mappings.
        """
        logger.info(f"Matching {len(t1_tables)} T1 tables vs {len(t2_tables)} T2 tables")

        if not t1_tables or not t2_tables:
            return []

        fp1 = self.create_fingerprints(t1_tables)
        fp2 = self.create_fingerprints(t2_tables)

        matches = []

        # Create matrices
        emb1 = np.array([f.embedding for f in fp1])
        emb2 = np.array([f.embedding for f in fp2])

        # Compute similarity matrix (N x M)
        sim_matrix = cosine_similarity(emb1, emb2)

        # Greedy matching strategy
        # We want to find the best T2 match for each T1 table

        matched_t2_indices = set()

        for i, f1 in enumerate(fp1):
            # Get best match for this T1 table
            row_sims = sim_matrix[i]
            best_idx = np.argmax(row_sims)
            score = row_sims[best_idx]

            match_result = {
                "t1_id": f1.table_id,
                "t1_title": f1.title,
                "t1_page": f1.page_number,
                "score": float(score),
                "status": "unmatched",
            }

            if score >= threshold:
                f2 = fp2[best_idx]
                match_result.update(
                    {
                        "t2_id": f2.table_id,
                        "t2_title": f2.title,
                        "t2_page": f2.page_number,
                        "status": "matched" if score > 0.9 else "renamed_likely",
                    }
                )
                matched_t2_indices.add(best_idx)
            else:
                match_result["status"] = "deleted_in_t2"

            matches.append(match_result)

        # Detect NEW tables in T2 (indices not matched)
        for j, f2 in enumerate(fp2):
            if j not in matched_t2_indices:
                matches.append(
                    {
                        "t2_id": f2.table_id,
                        "t2_title": f2.title,
                        "t2_page": f2.page_number,
                        "status": "new_in_t2",
                        "score": 0.0,
                    }
                )

        return matches

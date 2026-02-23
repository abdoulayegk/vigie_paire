import logging

logging.basicConfig(level=logging.INFO)

from vigilance.models.table_models import TableArtifact
from vigilance.compare.indicator_comparator import match_decision

t1 = TableArtifact(
    table_id="tableau_7",
    bank_code="bns",
    extraction_method="docling",
    page_pdf=37,
    section="risk_management",
    title="",
    rows=[],
    first_column_indicators=[],
    headers=[],
)

t2 = TableArtifact(
    table_id="tableau_7",
    bank_code="bns",
    extraction_method="docling",
    page_pdf=43,
    section="risk_management",
    title="",
    rows=[],
    first_column_indicators=[],
    headers=[],
)

res = match_decision(t1, t2)
print("Is match?", res.is_match)
print("Reason:", res.reason)
print("Score:", res.score)

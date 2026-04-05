"""Scan comparison.json files for pure date strings falsely flagged as added/removed."""
import json
import glob
import re

date_re = re.compile(
    r"^(?:Au\s+)?\d{1,2}\s+"
    r"(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
    r"\s+\d{4}\s*$",
    re.IGNORECASE,
)

count = 0
for f in sorted(glob.glob("outputs/comparisons/*/2*_vs_*/comparison.json")):
    with open(f) as fh:
        data = json.load(fh)
    bank = f.split("/")[2].upper()
    for p in data.get("pair_comparisons", []):
        td = p.get("technical_diff", {})
        for cat in ("indicators_added", "indicators_removed"):
            for ind in td.get(cat, []):
                val = ind.get("value", "")
                if date_re.match(val.strip()):
                    tid = p.get("current_table_id", "") or p.get("previous_table_id", "")
                    label = "AJOUTÉ" if "added" in cat else "SUPPRIMÉ"
                    print(f"{bank:5} | {tid:20} | {label:10} | \"{val}\"")
                    count += 1

print(f"\nTotal false date indicators: {count}")

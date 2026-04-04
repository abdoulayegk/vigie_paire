"""Quick script to dump all indicator changes from all bank comparisons."""
import json
import glob
import os

for cpath in sorted(glob.glob("outputs/*/comparison.json")):
    bank = os.path.basename(os.path.dirname(cpath))
    with open(cpath) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"BANK: {bank}")
    print(f"{'='*60}")

    for pair in data.get("pair_comparisons", []):
        ct = pair.get("current_table", {})
        title = ct.get("title", "") or pair.get("title", "")
        td = pair.get("technical_diff", {})
        added = td.get("indicators_added", [])
        removed = td.get("indicators_removed", [])
        renamed = td.get("indicators_renamed", [])
        if added or removed or renamed:
            print(f"\n  --- {title} ---")
            for a in added:
                val = a.get("value", a) if isinstance(a, dict) else a
                print(f"    ADDED: {val}")
            for r in removed:
                val = r.get("value", r) if isinstance(r, dict) else r
                print(f"    REMOVED: {val}")
            for rn in renamed:
                print(f"    RENAMED: {rn.get('previous','')} -> {rn.get('current','')}")

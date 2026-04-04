#!/usr/bin/env python3
"""Audit script to analyze table matching quality across all 6 banks."""

import json
import sys


def load_tables(path):
    with open(path) as f:
        data = json.load(f)
    tables = data if isinstance(data, list) else data.get("tables", [])
    return {t.get("table_id", ""): t for t in tables}


def get_indicators(t):
    inds = t.get("indicators", [])
    names = set()
    for ind in inds:
        name = ind.get("name", "") if isinstance(ind, dict) else str(ind)
        names.add(name.lower().strip())
    names.discard("")
    return names


def print_table_summary(t, prefix=""):
    tid = t.get("table_id", "")
    title = t.get("title", "")[:100]
    section = t.get("section", "")[:60]
    rows = t.get("row_count", "?")
    page = t.get("page", "?")
    summary = t.get("table_summary", "")[:120]
    first_ind = t.get("first_indicator", "")[:60]
    inds = get_indicators(t)
    fn_count = t.get("footnote_count", 0)
    print(f"{prefix}ID: {tid} | Page: {page} | Rows: {rows} | FN: {fn_count}")
    print(f"{prefix}Title: {title}")
    print(f"{prefix}Section: {section}")
    print(f"{prefix}Summary: {summary}")
    print(f"{prefix}First indicator: {first_ind}")
    print(f"{prefix}Indicators ({len(inds)}): {sorted(list(inds))[:10]}")
    print()


def audit_bank(bank_name, comp_path, t1_path, t2_path):
    t1 = load_tables(t1_path)
    t2 = load_tables(t2_path)
    with open(comp_path) as f:
        comp = json.load(f)

    matching = comp.get("matching", {})
    added = matching.get("tables_added", [])
    removed = matching.get("tables_removed", [])
    pairs = matching.get("matched_pairs", [])

    added_ids = [a["table_id"] for a in added]
    removed_ids = [r["table_id"] for r in removed]

    print("=" * 80)
    print(f"{bank_name} DEEP DIVE")
    print(f"T1: {len(t1)} tables | T2: {len(t2)} tables")
    print(f"Matched: {len(pairs)} | Added: {len(added)} | Removed: {len(removed)}")
    print("=" * 80)

    # Show matched pairs with confidence
    print(f"\nMATCHED PAIRS ({len(pairs)}):")
    for p in pairs:
        pid = p.get("previous_table_id", "")
        cid = p.get("current_table_id", "")
        conf = p.get("match_confidence", 0)
        reason = p.get("reason", "")[:100]
        pt = t1.get(pid, {})
        ct = t2.get(cid, {})
        ptitle = pt.get("title", "")[:60]
        ctitle = ct.get("title", "")[:60]
        print(f"  {pid} <-> {cid} (conf={conf:.2f})")
        print(f'    T1: "{ptitle}" | T2: "{ctitle}"')
        # Check indicator overlap for matched pairs
        p_inds = get_indicators(pt)
        c_inds = get_indicators(ct)
        overlap = p_inds & c_inds
        union = p_inds | c_inds
        jaccard = len(overlap) / max(len(union), 1)
        if jaccard < 0.3:
            print(
                f"    *** LOW OVERLAP: {len(overlap)}/{len(union)} (Jaccard={jaccard:.2f}) ***"
            )

    print(f"\nADDED T2 tables ({len(added)}):")
    for tid in added_ids:
        if tid in t2:
            print_table_summary(t2[tid], "  ")

    print(f"\nREMOVED T1 tables ({len(removed)}):")
    for tid in removed_ids:
        if tid in t1:
            print_table_summary(t1[tid], "  ")

    # Cross-reference: find potential matches between added and removed
    if added_ids and removed_ids:
        print(
            "\n--- CROSS-REFERENCE: Potential matches between added(T2) and removed(T1) ---"
        )
        for a_id in added_ids:
            a = t2.get(a_id, {})
            a_inds = get_indicators(a)
            for r_id in removed_ids:
                r = t1.get(r_id, {})
                r_inds = get_indicators(r)
                overlap = a_inds & r_inds
                union = a_inds | r_inds
                jaccard = len(overlap) / max(len(union), 1)
                print(f"  T2 {a_id} vs T1 {r_id}:")
                print(f"    T2 title: {a.get('title', '')[:80]}")
                print(f"    T1 title: {r.get('title', '')[:80]}")
                print(f"    T2 section: {a.get('section', '')[:60]}")
                print(f"    T1 section: {r.get('section', '')[:60]}")
                print(
                    f"    T2 rows: {a.get('row_count', '?')} | T1 rows: {r.get('row_count', '?')}"
                )
                print(
                    f"    Indicator overlap: {len(overlap)}/{len(union)} (Jaccard={jaccard:.2f})"
                )
                if overlap:
                    print(f"    Shared: {sorted(list(overlap))[:8]}")
                print()

    # Also check: removed T1 tables vs ALL T2 tables (maybe matched to wrong one)
    print(
        "\n--- REMOVED T1 vs ALL T2: Could a removed T1 table match an already-matched T2? ---"
    )
    matched_t2_ids = {p["current_table_id"] for p in pairs}
    matched_t1_ids = {p["previous_table_id"] for p in pairs}

    for r_id in removed_ids:
        r = t1.get(r_id, {})
        r_inds = get_indicators(r)
        r_title = r.get("title", "").lower().strip()

        best_jaccard = 0
        best_match = None
        for tid, t in t2.items():
            t_inds = get_indicators(t)
            overlap = r_inds & t_inds
            union = r_inds | t_inds
            jaccard = len(overlap) / max(len(union), 1)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = tid

        if best_match:
            best_t = t2[best_match]
            status = "ALREADY MATCHED" if best_match in matched_t2_ids else "UNMATCHED"
            already_to = ""
            if best_match in matched_t2_ids:
                for p in pairs:
                    if p["current_table_id"] == best_match:
                        already_to = f" (paired with T1:{p['previous_table_id']})"
            print(f'  Removed T1 {r_id}: "{r.get("title", "")[:60]}"')
            print(
                f"    Best T2 candidate: {best_match} (Jaccard={best_jaccard:.2f}) [{status}]{already_to}"
            )
            print(f'    T2 title: "{best_t.get("title", "")[:60]}"')
            if best_jaccard > 0:
                r_only = r_inds - get_indicators(best_t)
                t_only = get_indicators(best_t) - r_inds
                print(
                    f"    Overlap: {len(r_inds & get_indicators(best_t))}/{len(r_inds | get_indicators(best_t))}"
                )
            print()

    # Check for unmatched T2 tables not in added list
    all_t2_accounted = matched_t2_ids | set(added_ids)
    unaccounted_t2 = set(t2.keys()) - all_t2_accounted
    if unaccounted_t2:
        print(
            f"\n*** UNACCOUNTED T2 tables (not matched, not added): {sorted(unaccounted_t2)} ***"
        )
        for tid in sorted(unaccounted_t2):
            print_table_summary(t2[tid], "  ")

    # Similarly for T1
    all_t1_accounted = matched_t1_ids | set(removed_ids)
    unaccounted_t1 = set(t1.keys()) - all_t1_accounted
    if unaccounted_t1:
        print(
            f"\n*** UNACCOUNTED T1 tables (not matched, not removed): {sorted(unaccounted_t1)} ***"
        )
        for tid in sorted(unaccounted_t1):
            print_table_summary(t1[tid], "  ")

    print("\n")


banks = {
    "RBC": {
        "comp": "outputs/RBC_2025T2_vs_2025T1/_comparisons/rbc/2025_t2_vs_2025_t1/comparison.json",
        "t1": "outputs/RBC_2025T2_vs_2025T1/_extractions/rbc/2025/t1/tables.json",
        "t2": "outputs/RBC_2025T2_vs_2025T1/_extractions/rbc/2025/t2/tables.json",
    },
    "CIBC": {
        "comp": "outputs/CIBC_2025T2_vs_2025T1/_comparisons/cibc/2025_t2_vs_2025_t1/comparison.json",
        "t1": "outputs/CIBC_2025T2_vs_2025T1/_extractions/cibc/2025/t1/tables.json",
        "t2": "outputs/CIBC_2025T2_vs_2025T1/_extractions/cibc/2025/t2/tables.json",
    },
    "BNS": {
        "comp": "outputs/BNS_2025T2_vs_2025T1/_comparisons/bns/2025_t2_vs_2025_t1/comparison.json",
        "t1": "outputs/BNS_2025T2_vs_2025T1/_extractions/bns/2025/t1/tables.json",
        "t2": "outputs/BNS_2025T2_vs_2025T1/_extractions/bns/2025/t2/tables.json",
    },
    "BNC": {
        "comp": "outputs/BNC_2025T2_vs_2025T1/_comparisons/bnc/2025_t2_vs_2025_t1/comparison.json",
        "t1": "outputs/BNC_2025T2_vs_2025T1/_extractions/bnc/2025/t1/tables.json",
        "t2": "outputs/BNC_2025T2_vs_2025T1/_extractions/bnc/2025/t2/tables.json",
    },
    "BMO": {
        "comp": "outputs/BMO_2026T1_vs_2025T3/_comparisons/bmo/2026_t1_vs_2025_t3/comparison.json",
        "t1": "outputs/BMO_2026T1_vs_2025T3/_extractions/bmo/2025/t3/tables.json",
        "t2": "outputs/BMO_2026T1_vs_2025T3/_extractions/bmo/2026/t1/tables.json",
    },
    "TD": {
        "comp": "outputs/TD_2026T1_vs_2025T3/_comparisons/td/2026_t1_vs_2025_t3/comparison.json",
        "t1": "outputs/TD_2026T1_vs_2025T3/_extractions/td/2025/t3/tables.json",
        "t2": "outputs/TD_2026T1_vs_2025T3/_extractions/td/2026/t1/tables.json",
    },
}

# Only audit problem banks first, then remaining
target = sys.argv[1] if len(sys.argv) > 1 else "ALL"
if target == "ALL":
    for bank_name, paths in banks.items():
        audit_bank(bank_name, paths["comp"], paths["t1"], paths["t2"])
else:
    paths = banks[target]
    audit_bank(target, paths["comp"], paths["t1"], paths["t2"])

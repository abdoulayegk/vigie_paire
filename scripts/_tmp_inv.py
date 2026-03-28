import json

# Check extraction source files
for pattern in [
    "outputs/TD_2026T1_vs_2025T3/T1-2026/indicators.json",
    "outputs/TD_2026T1_vs_2025T3/T3-2025/indicators.json",
    "outputs/TD_2026T1_vs_2025T3/_extractions/td/2026/t1/indicators.json",
    "outputs/TD_2026T1_vs_2025T3/_extractions/td/2025/t3/indicators.json",
]:
    try:
        with open(pattern) as f:
            data = json.load(f)
        print(f"=== {pattern} ===")
        if isinstance(data, list):
            # Find the ACTIFS LIQUIDES MOYENS table
            for tbl in data:
                title = tbl.get("title", "") or tbl.get("table_name", "")
                if "ACTIFS LIQUIDES MOYENS" in title.upper():
                    print(f"  Table: {title}")
                    print(f"  table_id: {tbl.get('table_id', '?')}")
                    print(f"  page: {tbl.get('page', '?')}")
                    indicators = tbl.get("indicators", [])
                    print(f"  indicators ({len(indicators)}):")
                    for i, ind in enumerate(indicators):
                        if isinstance(ind, dict):
                            print(f"    {i:2d}. {ind.get('name', '?')[:70]}")
                        else:
                            print(f"    {i:2d}. {str(ind)[:70]}")
                    print()
                    break
        elif isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:10]}")
            tables = data.get("tables", data.get("indicators", []))
            if isinstance(tables, list):
                for tbl in tables:
                    title = tbl.get("title", "") if isinstance(tbl, dict) else ""
                    if "ACTIFS LIQUIDES MOYENS" in title.upper():
                        print(f"  Found: {title}")
                        indicators = tbl.get("indicators", [])
                        print(f"  indicators ({len(indicators)}):")
                        for i, ind in enumerate(indicators):
                            name = (
                                ind.get("name", ind)
                                if isinstance(ind, dict)
                                else str(ind)
                            )
                            print(f"    {i:2d}. {str(name)[:70]}")
                        break
        print()
    except FileNotFoundError:
        pass

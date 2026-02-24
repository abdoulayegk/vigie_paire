# bank-peer-vigilance

Façade `src/vigilance/` avec moteurs désormais rangés sous `src/vigilance/`.

## Structure (résumé)

- `src/vigilance/extraction/`: moteur extraction PDF
- `src/vigilance/comparison/`: moteur comparaison historique
- `src/vigilance/compare/`: comparator strict intra-section (PR2)
- `extraction/` et `comparison/`: shims de compatibilité imports legacy
- `tests/unit/`

## Comparateur officiel

- Point d'entrée recommandé: `vigilance.compare.run_strict_intra_section_compare`
- Sortie harmonisée:
  - `pairs`
  - `added_tables`
  - `removed_tables`
  - `unmatched_t1`
  - `unmatched_t2`
  - `reasons`
- Compatibilité legacy:
  - `vigilance.comparison.indicator_comparator.run_strict_intra_section_compare` (wrapper)

## Règle de matching (stricte)

- Matching autorisé uniquement si sections identiques:
  - `capital_management` ↔ `capital_management`
  - `risk_management` ↔ `risk_management`
  - `regulatory_updates` ↔ `regulatory_updates`
- Tout cross-section est interdit (`cross_section_forbidden`)
- `unknown_section` ne matche jamais automatiquement

## Installation

Avec `pip`:

```bash
pip install -e .
```

Avec `uv`:

```bash
uv pip install -e .
```

Pour les dependances de dev (tests):

```bash
uv sync --extra dev
```

Regle pratique:
- ne pas lancer `uv sync` a chaque execution de l'app,
- lancer `uv sync` seulement quand `pyproject.toml` ou `uv.lock` change.

## Demarrage Dash (stable)

Commande recommandee:

```bash
cd /Users/balde/Desktop/vigie_paire
UV_CACHE_DIR=.uv-cache PYTHONPATH=src DASH_DEBUG=0 DOCLING_NUM_THREADS=4 uv run python -m app.app
```

Avec `fish`:

```fish
cd /Users/balde/Desktop/vigie_paire
set -x UV_CACHE_DIR .uv-cache
set -x PYTHONPATH src
set -x DASH_DEBUG 0
set -x DOCLING_NUM_THREADS 4
uv run python -m app.app
```

Ou via script helper (depuis n'importe quel dossier):

```bash
scripts/run_dash.sh
```

Variables utiles:
- `DASH_DEBUG` (defaut: `0`)
- `DASH_PORT` (defaut: `8050`)
- `DOCLING_NUM_THREADS` (defaut: `4`)
- `VISION_CACHE_DIR` (defaut: `outputs/vision_cache`) — repertoire du cache GPT-4o Vision pour les indicateurs de premiere colonne.
- `VISION_CROP_DIR` (defaut: `outputs/debug_crops/vision_fallback`) — repertoire des crops debug du fallback Vision.

## CLI

Détection des sections:

```bash
python -m vigilance.cli.run_ranges \
  --bank rbc \
  --pdf /path/to/report.pdf \
  --quarter t1-2025 \
  --config configs/bank_profiles.yaml \
  --out_root outputs/runs
```

Extraction des tableaux Docling sur ranges:

```bash
python -m vigilance.cli.run_tables \
  --bank rbc \
  --pdf /path/to/report.pdf \
  --quarter t1-2025 \
  --config configs/bank_profiles.yaml \
  --ranges_json outputs/runs/t1-2025/rbc/section_ranges.json \
  --out_root outputs/runs
```

Scripts installables équivalents:

```bash
vigilance-run-ranges --help
vigilance-run-tables --help
```

Sorties JSON:

- `outputs/runs/{quarter}/{bank}/section_ranges.json`
- `outputs/runs/{quarter}/{bank}/tables_docling.json`
  - chaque table contient `section` et `first_column_indicators`

## Benchmark runtime

Mesure mediane sur 3 runs:
- `T_boot` (commande -> "Dash is running")
- `T_first_screen` (commande -> endpoint `/` pret)
- `T_compare` (optionnel, comparaison PDF end-to-end)

Exemple startup uniquement:

```bash
python scripts/measure_runtime.py --runs 3 --output-json artifacts/perf/current.json
```

En environnement headless/sandbox (sans acces HTTP local):

```bash
python scripts/measure_runtime.py --runs 3 --skip-first-screen-check
```

Exemple complet avec comparaison:

```bash
python scripts/measure_runtime.py \
  --runs 3 \
  --pdf-t1 /path/to/t1.pdf \
  --pdf-t2 /path/to/t2.pdf \
  --bank rbc \
  --sections-t1 /path/to/sections_t1.json \
  --sections-t2 /path/to/sections_t2.json \
  --output-json artifacts/perf/current.json
```

Comparaison avec un baseline:

```bash
python scripts/measure_runtime.py \
  --runs 3 \
  --baseline-json artifacts/perf/baseline.json \
  --output-json artifacts/perf/current.json
```

## Audit strict intra-section

Commande locale:

```bash
uv run python scripts/audit_intra_section.py \
  --input tests/fixtures/strict_intra_section_sample.json \
  --output artifacts/intra_section_audit.json \
  --fail-on-violations
```

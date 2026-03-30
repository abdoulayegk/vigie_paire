# Bank Peer Vigilance

Application de vigie bancaire pour extraire, comparer et revoir des tableaux réglementaires issus de rapports PDF.

Le système repose sur trois blocs :
- extraction ciblée des sections pertinentes d’un rapport
- comparaison GPT des tableaux entre deux périodes
- interface Dash de revue analyste avec historique par run

## Architecture

Le pipeline officiel est le suivant :

```text
PDF
-> détection des sections utiles
-> extraction des tableaux
-> génération des artefacts canoniques
-> comparaison GPT sur tables.json
-> comparison.json
-> revue Dash
```

Artefacts d’extraction :
- `tables.json` : source canonique
- `indicators.json` : projection
- `footnotes.json` : projection

Artefact de comparaison :
- `comparison.json` : sortie backend officielle, historisée par `run_id`

## Structure utile

- `src/vigilance/extraction/` : moteur d’extraction PDF
- `src/vigilance/compare_gpt.py` : moteur de comparaison GPT
- `src/vigilance/cli/` : CLI d’extraction et de comparaison
- `src/vigilance/dash_app/` : application Dash
- `src/app/` : adaptateurs UI, orchestration Dash, historique et revue
- `configs/bank_profiles.yaml` : configuration banques / modèles / pipeline
- `tests/unit/` : tests unitaires

## Installation

```bash
uv sync --extra dev
```

Ou, si tu utilises `pip` :

```bash
pip install -e .
```

## Variables d’environnement

Le projet charge automatiquement `.env` au démarrage de Dash.

Variables principales :
- `OPENAI_API_KEY` : requis pour l’extraction Vision et la comparaison GPT
- `DASH_PORT` : port Dash, par défaut `8050`
- `DASH_DEBUG` : `0` ou `1`

## Lancer Dash

Depuis la racine du repo :

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m vigilance.dash_app.app
```

Puis ouvrir :

```text
http://127.0.0.1:8050
```

## Dossier `Inputs/` — Convention de Nomenclature

Ce dossier contient les rapports PDF des banques, organisés par **banque** puis par **année**.

### Structure Obligatoire

```
Inputs/
├── BNC/
│   └── 2025/
│       ├── BNC_2025_T1.pdf
│       └── BNC_2025_T2.pdf
├── RBC/
│   └── 2025/
│       ├── RBC_2025_T1.pdf
│       └── RBC_2025_T2.pdf
├── TD/
│   └── 2025/
│       ├── TD_2025_T1.pdf
│       └── TD_2025_T2.pdf
├── BMO/
├── BNS/
└── CIBC/
```

### Convention de Nommage des PDFs

**Format strict : `{BANQUE}_{ANNÉE}_{TRIMESTRE}.pdf`**

| Champ | Valeurs | Exemple |
|---|---|---|
| `{BANQUE}` | BNC, RBC, TD, BMO, BNS, CIBC | `BNC` |
| `{ANNÉE}` | 2024, 2025, … | `2025` |
| `{TRIMESTRE}` | T1, T2, T3, T4 | `T2` |

**Exemple complet :** `BNC_2025_T2.pdf`

### Utilisation

```bash
uv run python run_pipeline.py --bank BNC --year 2025 --quarter T2
```

```bash
uv run python run_pipeline.py --bank TD --year 2026 --quarter T1 --skip-extraction
```

Le pipeline trouvera automatiquement :
- **Courant :**  `Inputs/BNC/2025/BNC_2025_T2.pdf`
- **Précédent :** `Inputs/BNC/2025/BNC_2025_T1.pdf` (déduit automatiquement)

## CLI

### 1. Extraire un rapport

```bash
uv run vigilance-run-extract-report \
  --bank bnc \
  --pdf data/bnc/report.pdf \
  --year 2026 \
  --quarter t1
```

Sortie :

```text
outputs/extractions/{bank}/{year}/{quarter}/
  tables.json
  indicators.json
  footnotes.json
```

### 2. Comparer un rapport courant à sa période de référence métier

```bash
uv run vigilance-run-compare-gpt4o \
  --bank bnc \
  --year-current 2026 \
  --quarter-current t1
```

La période de référence est résolue automatiquement selon la règle métier :
- `T2-Y -> T1-Y`
- `T3-Y -> T2-Y`
- `T1-Y -> T3-(Y-1)`
- `T4-Y -> T4-(Y-1)`

Sortie :

```text
outputs/comparisons/{bank}/{current}_vs_{previous}/{run_id}/comparison.json
```

Le dossier de run contient aussi les PDF archivés utilisés pour la preuve Dash.

## Règles de comparaison

La comparaison GPT est volontairement limitée à :
- la première colonne des tableaux
- les footnotes associées

Le moteur ignore explicitement :
- les valeurs numériques
- les pourcentages et montants
- les dates
- les labels de période

Le résultat identifie :
- tableaux ajoutés / supprimés
- indicateurs ajoutés / supprimés / renommés
- footnotes ajoutées / supprimées / renommées

## Dash

L’interface Dash permet :
- l’upload de deux rapports
- la validation des sections détectées
- le lancement de l’extraction et de la comparaison
- la revue analyste par thème et priorité
- le rechargement d’un `comparison.json` historique

Les runs historiques sont isolés et non écrasables.

## Tests

Exécuter la suite unitaire :

```bash
uv run pytest
```

Exécuter un sous-ensemble ciblé :

```bash
uv run pytest tests/unit/test_compare_gpt.py
```

## Notes

- Les extractions sous `outputs/extractions/` servent de cache technique par période.
- La vérité historisée côté analyse est le dossier de run sous `outputs/comparisons/`.
- Le projet privilégie la robustesse de la revue analyste et l’auditabilité des résultats.

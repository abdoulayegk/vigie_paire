# Vigie Paires - Analyse des Rapports Bancaires

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Docling](https://img.shields.io/badge/Docling-Structure%20Extraction-orange.svg)](https://github.com/ibm/docling)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF%20Processing-red.svg)](https://pymupdf.readthedocs.io/)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-yellow.svg)](https://python-pillow.org/)
[![Tests](https://img.shields.io/badge/Tests-33%20passing-brightgreen.svg)](./tests/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)](./.github/workflows/ci.yml)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20Vision-black.svg)](https://openai.com/)

</div>

Système d'analyse automatisée des rapports trimestriels des 6 grandes banques canadiennes pour le Mouvement Desjardins

## Objectif

Remplacer l'outil Kofax Power par un système intelligent capable de :

- Identifier les modifications pertinentes entre trimestres
- Détecter les tendances adoptées par ≥3 banques
- Générer des rapports Excel exploitables
- Comparer structurellement les tableaux entre rapports trimestriels
- Analyser qualitativement les changements avec GenAI

---

## Prérequis

| Outil  | Version minimale |
| ------ | ---------------- |
| Python | 3.10+            |
| Git    | —                |

- **uv** est optionnel mais recommandé pour la reproductibilité et les workflows développeur.
- Installer **uv** si disponible : `pip install uv` ou [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)
- Si l'installation de **uv** est bloquée par les politiques de sécurité, utilisez l'option `pip` ci-dessous.
- Clé API OpenAI requise pour l'extraction Vision et la comparaison GPT-4o.

---

## Installation

### Option A — Installation avec `uv` (recommandée)

```bash
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
uv sync --group dev
cp .env.example .env        # puis renseigner OPENAI_API_KEY
```

```powershell
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
uv sync --group dev
copy .env.example .env      # puis renseigner OPENAI_API_KEY
```

### Option B — Installation avec `pip` (compatible environnements restreints)

#### Linux / macOS

```bash
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env        # puis renseigner OPENAI_API_KEY
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env      # puis renseigner OPENAI_API_KEY
```

`uv` reste préférable pour la reproductibilité et les workflows dev. `pip` suffit pour exécuter les pipelines applicatifs sans changement de code sur les scripts principaux.

`requirements.txt` couvre les dépendances runtime. Les dépendances de développement comme `pytest` et `reportlab` ne sont pas incluses dans le parcours `pip` standard.

---

## Variables d'environnement

Le projet charge automatiquement le fichier `.env` au démarrage.
Un modèle est fourni dans `.env.example` — il suffit de le copier et de renseigner la clé API.

```bash
# Linux / macOS
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

Ouvrir `.env` et remplacer `sk-...` par votre clé OpenAI.

Variables disponibles :

| Variable                 | Requis | Défaut | Description                    |
| ------------------------ | ------ | ------ | ------------------------------ |
| `OPENAI_API_KEY`         | Oui    | —      | Clé API OpenAI                 |
| `DASH_PORT`              | Non    | `8050` | Port de l'interface Dash       |
| `DASH_DEBUG`             | Non    | `0`    | Mode debug Dash (1 = activé)   |
| `DOCLING_NUM_THREADS`    | Non    | `4`    | Parallélisme extraction PDF    |
| `ENABLE_TABLE_CROP_DUMP` | Non    | `0`    | Dump images de crop (débogage) |

> Le fichier `.env` est dans `.gitignore` — ne jamais le commiter.

---

## Banques supportées

| Code | Banque                     |
| ---- | -------------------------- |
| bnc  | Banque Nationale du Canada |
| rbc  | Banque Royale du Canada    |
| td   | Banque Toronto-Dominion    |
| bmo  | Banque de Montréal         |
| bns  | Banque de Nouvelle-Écosse  |
| cibc | CIBC                       |

## Structure des entrées

Déposer les PDFs dans `Inputs/` selon la convention suivante :

```text
Inputs/
  TD/
    2026/
      TD_2026_T1.pdf
    2025/
      TD_2025_T3.pdf
  BNC/
    2025/
      BNC_2025_T1.pdf
      BNC_2025_T2.pdf
```

**Format de nommage strict :** `{BANQUE}_{ANNÉE}_{TRIMESTRE}.pdf`

Banques supportées : `BNC` `RBC` `TD` `BMO` `BNS` `CIBC`

Le trimestre de référence est déduit automatiquement :
`T2→T1` · `T3→T2` · `T1→T3(N-1)` · `T4→T4(N-1)`

---

## Exécuter le pipeline

### Commande recommandée — Pipeline complet (indicateurs + texte)

`run_full_pipeline.py` est le point d’entrée principal. Il enchaîne automatiquement le pipeline indicateurs (tableaux chiffrés) puis le pipeline texte (risques, capital, etc.). Depuis le venv activé, vous pouvez lancer les scripts directement avec `python ...`. Si vous utilisez `uv`, préfixez les mêmes commandes avec `uv run`.

```bash
# Flux complet : extraction + comparaison indicateurs + comparaison texte
python run_full_pipeline.py --banque BNC --annee 2025 --T2

# Réutiliser les extractions indicateurs existantes (tables.json déjà présents)
# Le pipeline texte fait toujours son extraction de texte
python run_full_pipeline.py --banque BNC --annee 2025 --T2 --sans-extraction

# Indicateurs seulement (sauter le pipeline texte)
python run_full_pipeline.py --banque BNC --annee 2025 --T2 --sans-texte

# Texte seulement (sauter le pipeline indicateurs)
python run_full_pipeline.py --banque BNC --annee 2025 --T2 --sans-indicateurs
```

Avec `uv`, utilisez les mêmes commandes en les préfixant avec `uv run`, par exemple :

```bash
uv run python run_full_pipeline.py --banque BNC --annee 2025 --T2
```

**Options de `run_full_pipeline.py` :**

| Option | Description |
| --- | --- |
| `--banque` | Code banque : `BNC`, `RBC`, `TD`, `BMO`, `BNS`, `CIBC` |
| `--annee` | Année du rapport courant (ex : `2025`) |
| `--T1` / `--T2` / `--T3` / `--T4` | Trimestre courant (flags exclusifs) |
| `--sans-extraction` | Saute l’extraction indicateurs — le pipeline texte s’exécute en entier |
| `--sans-comparaison` | Saute toutes les comparaisons GPT-4o (indicateurs et texte) |
| `--sans-indicateurs` | Ignore entièrement le pipeline indicateurs (tableaux chiffrés) |
| `--sans-texte` | Ignore entièrement le pipeline texte (risques, capital, etc.) |
| `--sortie` | Répertoire racine des résultats (défaut : `outputs/resultats`) |

> **Note `--sans-extraction` :** ce flag ne s’applique qu’au pipeline indicateurs (réutilise les `tables.json` existants). Le pipeline texte effectue toujours sa propre extraction de texte depuis les PDFs.

---

### Pipelines individuels (usage avancé)

Il est aussi possible de lancer chaque pipeline séparément.

**Pipeline indicateurs uniquement :**

```bash
# Extraction + Comparaison complètes des tableaux
python run_pipeline.py --bank TD --year 2026 --quarter T1

# Réutiliser l’extraction existante (tables.json déjà présent)
python run_pipeline.py --bank TD --year 2026 --quarter T1 --skip-extraction

# Sauter la comparaison (re-triage uniquement)
python run_pipeline.py --bank TD --year 2026 --quarter T1 --skip-comparison
```

**Pipeline texte uniquement :**

```bash
# Extraction + Comparaison sémantique par sous-sections (T2 vs T1)
python run_text_pipeline.py --bank BNS --year 2025 --T2

# Sauter la comparaison (extraction seulement)
python run_text_pipeline.py --bank BNS --year 2025 --T2 --skip-comparison
```

---

**Artefacts produits :**

```text
outputs/resultats/{banque}/{annee_q}_vs_{annee_prev_q}/comparison.json
outputs/resultats/{banque}/{annee_q}_vs_{annee_prev_q}/text_comparison.json
```

Les comparaisons indicateurs et texte partagent la même racine `outputs/resultats/`, ce que lit aussi l’interface Dash. Si vous avez d’anciens dossiers sous `outputs/comparisons/` ou `outputs/text_comparisons/`, fusionnez-les vers `outputs/resultats/` (mêmes sous-dossiers `banque/année_q_vs_année_q`). Un script automatise la fusion : `python scripts/migrate_outputs_to_resultats.py --dry-run` ou `uv run python scripts/migrate_outputs_to_resultats.py --dry-run`.

---

## Lancer l'interface Dash

### Option A — Avec `uv`

```bash
uv run python -m vigilance.dash_app.app
```

### Dash — Linux / macOS

```bash
# Alternative pratique sous Bash si vous utilisez uv
bash scripts/run_dash.sh
```

### Dash — Windows (PowerShell)

```powershell
uv run python -m vigilance.dash_app.app
```

### Option B — Avec `pip`

Le mode `pip` est officiellement documenté pour les pipelines CLI. Pour Dash, il faut aussi installer le projet localement dans le venv :

```bash
python -m pip install -e .
python -m vigilance.dash_app.app
```

Le script `bash scripts/run_dash.sh` dépend de `uv` et ne convient pas à un environnement `pip` pur.

Ouvrir ensuite : [http://localhost:8050](http://localhost:8050)

---

## Guide d'utilisation — Interface analyste

### 1. Charger une comparaison

Au démarrage, sélectionner un `comparison.json` dans le menu déroulant puis cliquer **Charger**.

### 2. File d'attente

Les tableaux sont triés automatiquement par priorité d'action :

| Priorité          | Signification                  |
| ----------------- | ------------------------------ |
| **Escalade**      | Intervention immédiate requise |
| **Investigation** | Analyse approfondie nécessaire |
| **Confirmation**  | Vérification de cohérence      |
| **Information**   | Changement mineur à documenter |

### 3. Valider les changements

Sélectionner un tableau dans la file. Le panneau de détail affiche les changements détectés :

- **Indicateurs ajoutés / supprimés / renommés**
- **Footnotes modifiées**

Pour chaque changement :

1. Consulter les **preuves visuelles** — aperçu PDF T1 / T2 avec surbrillance de la modification (magenta = changement actif, jaune = contexte)
2. Lire la **synthèse GenAI** (catégorie, niveau de risque, narratif)
3. Choisir une action : **Approuver** · **Rejeter** · **Passer**
4. Ajouter une note si nécessaire

L'interface avance automatiquement au changement suivant après validation.

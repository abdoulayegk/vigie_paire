# Vigie de Paire — Analyse de Rapports Bancaires

Système d'extraction, de comparaison et de revue des tableaux réglementaires issus de rapports trimestriels PDF.

> Projet interne — usage restreint.

---

## Prérequis

| Outil  | Version minimale |
| ------ | ---------------- |
| Python | 3.10+            |
| uv     | dernière         |
| Git    | —                |

- Installer **uv** : `pip install uv` ou [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)
- Clé API OpenAI requise pour l'extraction Vision et la comparaison GPT-4o.

---

## Installation

### Linux / macOS

```bash
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
uv sync --group dev
cp .env.example .env        # puis renseigner OPENAI_API_KEY
```

### Windows (PowerShell)

```powershell
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
uv sync --group dev
copy .env.example .env      # puis renseigner OPENAI_API_KEY
```

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

| Variable                 | Requis | Défaut | Description                        |
| ------------------------ | ------ | ------ | ---------------------------------- |
| `OPENAI_API_KEY`         | Oui    | —      | Clé API OpenAI                     |
| `DASH_PORT`              | Non    | `8050` | Port de l'interface Dash           |
| `DASH_DEBUG`             | Non    | `0`    | Mode debug Dash (1 = activé)       |
| `DOCLING_NUM_THREADS`    | Non    | `4`    | Parallélisme extraction PDF        |
| `ENABLE_TABLE_CROP_DUMP` | Non    | `0`    | Dump images de crop (débogage)     |

> Le fichier `.env` est dans `.gitignore` — ne jamais le commiter.

---

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

```bash
# Extraction + Comparaison complètes
uv run run_pipeline.py --bank TD --year 2026 --quarter T1

# Réutiliser l'extraction existante (tables.json déjà présent)
uv run run_pipeline.py --bank TD --year 2026 --quarter T1 --skip-extraction

# Sauter la comparaison (re-triage uniquement)
uv run run_pipeline.py --bank TD --year 2026 --quarter T1 --skip-comparison
```

**Options :**

| Option              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `--bank`            | Code banque : `BNC`, `RBC`, `TD`, `BMO`, `BNS`, `CIBC`       |
| `--year`            | Année du rapport courant                                     |
| `--quarter`         | Trimestre courant : `T1`, `T2`, `T3`, `T4`                   |
| `--skip-extraction` | Réutilise les `tables.json` existants                        |
| `--skip-comparison` | Saute la comparaison GPT-4o                                  |
| `--inputs-root`     | Répertoire des PDFs (défaut : `Inputs/`)                     |
| `--outputs-root`    | Répertoire de sortie (défaut : `Outputs/`)                   |

**Artefact produit :**

```text
outputs/comparisons/{banque}/{annee_q}_vs_{annee_prev_q}/comparison.json
```

---

## Lancer l'interface Dash

Commande de référence, identique sur toutes les plateformes :

```bash
uv run python -m vigilance.dash_app.app
```

### Linux / macOS

```bash
# Alternative pratique sous Bash
bash scripts/run_dash.sh
```

### Windows (PowerShell)

```powershell
uv run python -m vigilance.dash_app.app
```

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

### 4. Navigation

- Flèches `← →` pour naviguer entre les tableaux
- Flèches `‹ ›` pour naviguer entre les changements d'un tableau
- Clic direct sur un tableau dans la file pour y accéder

---

## Architecture

Le pipeline suit ce flux :

```text
PDF → Détection des sections → Extraction des tableaux (Vision GPT-4o)
    → Artefacts canoniques (tables.json)
    → Comparaison GPT-4o → comparison.json
    → Triage GenAI → Interface Dash
```

Voir `Architecture_Vigilance_Bancaire_Desjardins.pdf` pour le détail technique.

# Design Specification — Analyse Comparative Rapports Bancaires (T1/T2)

## 1. Vue d'ensemble

- **Fichier principal:** `app_bnc_validation.py`
- **Titre de page:** "Analyse Comparative Rapports T1/T2 - 2025"
- **Layout:** `wide`
- **Objectif:** Outil de comparaison de tableaux financiers extraits de rapports trimestriels (images PNG) entre deux trimestres, avec validation humaine (analyste) des changements détectés par l'IA (GPT-4o). Support multi-banques : BNC, TD, RBC, BNS (Scotiabank), CIBC, BMO.

---

## 2. Architecture globale de l'interface

L'application est divisée en **3 zones principales** :

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER  │  Sélecteur Banque  │  Titre  │  Tab 2 : Revue  │  Tab 3 : Export │
├────────────┬────────────────────────┬────────────────────────────┤
│  SIDEBAR   │  PANNEAU CENTRAL       │  PANNEAU DROITE            │
│  Contexte  │  File de Revue         │  Détail du Changement      │
│  d'Analyse │  (liste des items)     │  (aperçu + décision)       │
├────────────┴────────────────────────┴────────────────────────────┤
│  FOOTER — Validation Client (statistiques globales + exports)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Header / Navigation

- **Sélecteur de banque** (coin gauche du header) : `st.selectbox` intégré dans le header avec les 6 options :
  `["BNC", "TD", "RBC", "BNS", "CIBC", "BMO"]`, défaut `"BNC"`.
  - Le header adopte la **couleur primaire de la banque sélectionnée** (voir palette ci-dessous).
  - Le nom de la banque sélectionnée s'affiche en grand à gauche.
- **Sous-titre contextuel dynamique** — affiché sous le nom de la banque, mis à jour automatiquement à chaque changement de paramètre :
  ```
  BNC  •  T1 2025 vs T2 2025  |  Analyste : Jean Dupont
  ```
  - Rendu avec `st.markdown` en CSS (taille ~14px, couleur blanc semi-transparent `rgba(255,255,255,0.85)`).
  - Ancre l'analyste dans son contexte sans avoir à regarder la sidebar.
- **Mini-barre de progression** — affichée à droite du sous-titre contextuel, sur la même ligne :
  ```
  12 / 45 validés  ██████░░░░░░  27%
  ```
  - Rendu avec `st.progress` ou barre HTML inline.
  - Compteur `{validés} / {total} validés` en texte blanc à gauche de la barre.
  - Se met à jour en temps réel à chaque décision de l'analyste.
  - Permet au superviseur de voir l'avancement **sans quitter le header**.
- **Titre principal:** `"Analyse Comparative Rapports T1/T2 - 2025"` — texte blanc sur fond coloré.
- **Tabs de navigation** (style onglets) :
  - `2) Revue des Changements` (onglet actif par défaut)
  - `3) Exporter & Auditer`
- Style header : fond coloré selon banque sélectionnée, pleine largeur, hauteur fixe ~80px (agrandi pour accueillir le sous-titre et la barre de progression).

### Palette de couleurs par banque

| Banque | Couleur primaire | Code hex |
|--------|-----------------|----------|
| BNC | Bleu marine | `#1B3A6B` |
| TD | Vert | `#00A651` |
| RBC | Bleu royal | `#005DAA` |
| BNS | Rouge | `#C8102E` |
| CIBC | Rouge bordeaux | `#8B0000` |
| BMO | Bleu clair | `#0079C1` |

> Le sélecteur du header est synchronisé avec le champ **Banque** de la sidebar via `st.session_state.selected_bank`. Changer l'un met à jour l'autre.

---

## 4. Sidebar — "Contexte d'Analyse"

Titre de section : **"Contexte d'Analyse"** (gras, taille H3).

### Champs de configuration

| Label | Widget | Valeurs / Défaut |
|-------|--------|-----------------|
| **Banque** | `st.selectbox` | `["BNC", "TD", "RBC", "BNS", "CIBC", "BMO"]`, défaut `"BNC"` |
| **Clé API OpenAI** | `st.text_input(type="password")` | — (champ masqué) |
| **Année** | `st.text_input` | Défaut `"2025"` |
| **Trimestre T1** | `st.text_input` | Défaut `"Q1"` |
| **Trimestre T2** | `st.text_input` | Défaut `"Q2"` |
| **Nom de l'Analyste** | `st.text_input` | Ex. `"Jean Dupont"` |
| **Upload des Données** | `st.radio` | Deux options (ex. upload fichier / dossier local) |
| **Source des Données** | `st.text_input` ou `st.file_uploader` | Chemin ou fichier |

- Séparateur `st.divider()` entre chaque groupe logique.
- Les valeurs sont persistées dans `st.session_state`.

---

## 5. Panneau central — "2) Revue des Changements"

### 5.1 Barre de progression KPI

```
KPI ──────────────────────────────────── 0.5
```

- `st.slider` ou `st.progress` affichant un score KPI global (0.0 → 1.0).
- Label `"KPI"` à gauche, valeur numérique à droite.

---

### 5.2 Tableau de Bord — Chiffres Clés (Superviseur)

Bloc de métriques visuelles affiché **en haut du panneau central**, bien en évidence, avant la file de revue. Rendu avec `st.columns(6)` + `st.metric()` pour chaque indicateur.

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  Tableaux    │ Indicateurs  │ Indicateurs  │  Indicateurs │  Tableaux    │  Tableaux    │
│  Appariés    │   Ajoutés    │  Supprimés   │  Renommés    │   Ajoutés    │  Supprimés   │
│              │              │              │              │              │              │
│     38       │     12       │      7       │      4       │      3       │      2       │
│  ▲ sur 45    │              │              │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

| Métrique | Clé session | Description | Couleur delta |
|----------|-------------|-------------|---------------|
| **Tableaux Appariés** | `matched_tables` | Nb de tableaux T1↔T2 mis en correspondance | Neutre |
| **Indicateurs Ajoutés** | `total_indicators_added` | Nb total d'indicateurs apparus en T2 | Vert (`delta_color="normal"`) |
| **Indicateurs Supprimés** | `total_indicators_removed` | Nb total d'indicateurs disparus en T2 | Rouge (`delta_color="inverse"`) |
| **Indicateurs Renommés** | `total_indicators_renamed` | Nb d'indicateurs dont le libellé a changé | Orange (`delta_color="off"`) |
| **Tableaux Ajoutés** | `tables_added` | Nb de nouveaux tableaux présents uniquement en T2 | Vert (`delta_color="normal"`) |
| **Tableaux Supprimés** | `tables_removed` | Nb de tableaux présents en T1 absents en T2 | Rouge (`delta_color="inverse"`) |

**Règles d'affichage :**
- Chaque `st.metric` affiche : titre en français, valeur chiffrée en grand, et un sous-texte delta optionnel (ex. `"sur 45 tableaux"`).
- Les cartes métriques ont un fond légèrement coloré (`st.container` avec CSS inline) pour les distinguer visuellement.
- Un clic sur une carte filtre automatiquement la file de revue sur la catégorie correspondante (ex. clic sur "Indicateurs Ajoutés" → filtre `type_changement == "ajout"`).
- Les valeurs sont recalculées en temps réel depuis `st.session_state.report_data` à chaque rerun.

**Exemple de code indicatif :**
```python
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Tableaux Appariés",   report["_metadata"]["matched"],              f"sur {report['_metadata']['total_q1_tables']}")
col2.metric("Indicateurs Ajoutés", report["_metadata"]["total_indicators_added"],   delta_color="normal")
col3.metric("Indicateurs Supprimés", report["_metadata"]["total_indicators_removed"], delta_color="inverse")
col4.metric("Indicateurs Renommés",  report["_metadata"]["total_indicators_renamed"], delta_color="off")
col5.metric("Tableaux Ajoutés",    report["_metadata"]["tables_added"],          delta_color="normal")
col6.metric("Tableaux Supprimés",  report["_metadata"]["tables_removed"],        delta_color="inverse")
```

---

### 5.3 File de Revue — Structure en Cartes Hiérarchiques

**En-tête de la file:**

```
File de Revue
Total : 45   ● Validés   ● En Attente   30
```

- `st.metric` ou `st.markdown` avec badges colorés :
  - `● Validés` — vert
  - `● En Attente` — orange
  - `● Rejetés` — rouge
  - Nombre affiché à droite (ex. `30`)

**Filtres (ligne de filtres sous l'en-tête) :**

- `st.checkbox` — filtrer par section active (ex. `☑ Gestion du Capital`)
- Badge cliquable `+ Statut : En Attente` pour afficher/masquer selon statut



Remplace le `st.dataframe` par une **liste de cartes `st.container(border=True)`** pour une hiérarchie visuelle forte et un contenu immédiatement "scan-nable".

**Structure visuelle d'une carte de section :**

```
┌────────────────────────────────────────────────────────────────┐
│  ●  Gestion du Capital                Type : 1   2 modifiés   │
│                                                                │
│  └─ ⊙  Émission d'actions ordinaires rel. à l'acq. de CWB    │
│        Score : 0.95 ████████████ (vert)          [En Attente] │
│                                                                │
│  └─ ⊙  Ratio de levier financier                              │
│        Score : 0.80 █████████░░░ (orange)        [En Attente] │
└────────────────────────────────────────────────────────────────┘
```

**Règles de rendu :**

| Élément | Widget / Style |
|---------|---------------|
| **Carte de section** (niveau 1) | `st.container(border=True)` + fond gris très clair `#F8F9FA` |
| **Icône statut section** `●` | Cercle coloré CSS : vert (validé), orange (en attente), rouge (rejeté) |
| **Nom de section + Type** | `st.markdown` gras, taille H4 |
| **Compteur "X modifiés"** | Badge arrondi coloré (orange si > 0, vert si tout validé) |
| **Sous-carte indicateur** (niveau 2) | `st.container` indenté (~20px margin-left), fond blanc, bordure gauche colorée selon score |
| **Icône indicateur** `⊙` | Icône cible ou point coloré |
| **Libellé indicateur** | Texte tronqué à 60 chars avec tooltip complet au survol |
| **Barre de score** | Mini `st.progress` inline (vert si ≥ 0.85, orange si 0.6–0.84, rouge si < 0.6) |
| **Score numérique** | Affiché à droite de la barre (`0.95`) — permet de repérer les cas litigieux |
| **Badge statut** | `"En Attente"` / `"Validé"` / `"Rejeté"` — pill coloré à droite |

**Interaction :**
- Clic sur une **sous-carte indicateur** → sélectionne l'item et met à jour le panneau droit "Détail du Changement".
- La sous-carte sélectionnée est surlignée (bordure gauche épaisse, fond bleu très clair `#E8F0FE`).
- Clic sur une **carte de section** (en-tête) → déplie/replie les sous-cartes (`st.session_state` toggle).
- Les sections sans changements sont **repliées par défaut**.
- Les sections avec score < 0.8 sur au moins un indicateur sont **dépliées automatiquement** (cas litigieux prioritaires).

---

## 6. Panneau droit — "Détail du Changement"

> **Principe directeur :** Ce panneau est centré sur la **visualisation humaine**. Les changements purement numériques sont **ignorés et filtrés en amont** — seuls les changements de libellés textuels (ajout, suppression, renommage d'indicateurs) sont présentés. L'analyste valide en regardant les deux images en grand, côte à côte, et juge lui-même.

---

### 6.1 En-tête du changement sélectionné

```
Tableau : Variation des Fonds Propres          T1 : p.24  →  T2 : p.28
```

- Nom du tableau (texte) à gauche.
- Référence pages T1 → T2 à droite, format `T1 : p.X  →  T2 : p.Y`.
- **Pas de score de confiance affiché ici** — l'analyste forme son propre jugement visuel.

---

### 6.2 Changement textuel détecté

Affiche **uniquement le ou les libellés textuels** qui ont changé entre T1 et T2.
Les valeurs numériques dans les cellules sont **complètement ignorées**.

```
┌─────────────────────────────────────────────────────────────────┐
│  Changement détecté                                             │
│                                                                 │
│  T1 :  « Émission d'actions ordinaires »                        │
│  T2 :  « Émission d'actions ordinaires relatives à l'acq. CWB »│
│                                                                 │
│  Type : AJOUT / RENOMMAGE / SUPPRESSION                         │
└─────────────────────────────────────────────────────────────────┘
```

- Rendu avec `st.container(border=True)`.
- Texte T1 en gris barré si supprimé, texte T2 en vert si ajouté, les deux affichés si renommage.
- **Aucune valeur numérique n'apparaît dans ce bloc.**
- Si plusieurs changements textuels sur le même tableau : liste verticale de paires T1/T2.

---

### 6.3 Visualisation côte à côte — Images en grand

C'est **le cœur du panneau**. Les deux images PNG du tableau occupent **toute la largeur disponible**.

```
┌──────────────────────────────┐  ┌──────────────────────────────┐
│                              │  │                              │
│   IMAGE COMPLÈTE T1          │  │   IMAGE COMPLÈTE T2          │
│   (tableau PNG pleine taille)│  │   (tableau PNG pleine taille)│
│                              │  │                              │
│                              │  │                              │
│                              │  │                              │
└──────────────────────────────┘  └──────────────────────────────┘
   T1 — Page 24                      T2 — Page 28
```

- `st.columns([1, 1])` avec `st.image(use_container_width=True)` dans chaque colonne.
- Les images sont affichées à **pleine largeur de colonne**, sans limite de hauteur (scroll vertical si nécessaire).
- Sous-titre centré sous chaque image : `"T1 — Page 24"` / `"T2 — Page 28"` en `st.caption`.
- **Pas de surlignage automatique** — l'analyste voit les images brutes telles qu'elles sont.
- Bouton optionnel **"Agrandir"** sous chaque image → ouvre l'image en plein écran via `st.image` dans un `st.expander` ou modale CSS.

---

### 6.4 Décision de l'Analyste

Directement sous les images, sans intermédiaire :

```
[ ✓ Valider — Changement Réel ]     [ ✗ Rejeter — Faux Positif ]
```

- `st.button("✓ Valider — Changement Réel", type="primary")` → fond vert, pleine largeur de sa colonne.
- `st.button("✗ Rejeter — Faux Positif")` → fond rouge, pleine largeur de sa colonne.
- Rendu avec `st.columns(2)`.

**Commentaire optionnel** (affiché uniquement si l'analyste clique sur "Rejeter") :

```
Motif du rejet (Optionnel)
┌────────────────────────────────────────────┐
│  Ex. : libellé identique, numérotation...  │
└────────────────────────────────────────────┘
[ Confirmer le rejet ]
```

- `st.text_area` qui apparaît dynamiquement après clic sur Rejeter (`st.session_state` toggle).
- Bouton "Confirmer le rejet" pour valider avec le commentaire.
- Cela évite les rejets accidentels.

---

### 6.5 Navigation entre items

```
[ ◀  Précédent ]          Item 7 / 45          [ Suivant  ▶ ]
```

- `st.columns([1, 2, 1])` : bouton gauche, compteur centré, bouton droit.
- Compteur `"Item X / Y"` en `st.markdown` centré.
- Boutons désactivés (`disabled=True`) en début/fin de file.
- Raccourci clavier documenté dans une `st.caption` : `"← → pour naviguer"`.

---

### 6.6 Règles de filtrage — Ce qui est ignoré

Le pipeline IA **ne transmet pas** au panneau de détail les changements suivants :

| Type ignoré | Exemple |
|-------------|---------|
| Variation de valeur numérique | `1 234` → `1 456` |
| Variation de pourcentage | `12,3 %` → `14,1 %` |
| Variation de date seule | `31 jan. 2024` → `31 jan. 2025` |
| Cellule vide → valeur numérique | `—` → `892` |
| Reformatage numérique | `1 234 567` → `1,234,567` |

Seuls les **changements de libellés textuels** (noms de lignes, titres de colonnes, entêtes de section) sont remontés pour validation.

---

## 7. Footer — "Validation Client"

Barre pleine largeur, fond bleu marine, texte blanc.

```
Validation Client                              ⟳ Streamlit

Indicateurs Analysés : 45    Validés : 12    Rejetés : 10    Rejetés :
Validés : 145                Importer PDN    Importer JSON   Arporter JSON   Construit cv...
```

**Colonne gauche — Statistiques :**

| Métrique | Valeur |
|----------|--------|
| Indicateurs Analysés | 45 |
| Validés | 145 |

**Colonnes centrales — Actions :**

| Bouton | Action |
|--------|--------|
| `Validés : 12` | Filtre vue sur validés |
| `Importer PDN` | Import fichier PDN |
| `Rejetés : 10` | Filtre vue sur rejetés |
| `Importer JSON` | Import rapport JSON |
| `Rejetés :` | (statut) |
| `Arporter JSON` | Export rapport JSON |
| `Construit cv...` | Export CSV |

- `st.download_button` pour chaque export.
- `st.button` pour chaque import (déclenche `st.file_uploader` ou dialogue).

---

## 8. État de session (Session State)

| Clé | Type | Rôle |
|-----|------|------|
| `analysis_done` | bool | Analyse exécutée avec succès |
| `current_comparison` | str \| None | Clé `{bank}_{year}_{q1}_{q2}` |
| `report_data` | dict \| None | Rapport structuré complet |
| `api_key` | str | Clé API OpenAI |
| `analyst_name` | str | Nom de l'analyste |
| `selected_item_index` | int | Index de l'item sélectionné dans la file |
| `decisions` | dict | `{item_id: {"decision": "valider"|"rejeter", "comment": str}}` |
| `kpi_score` | float | Score KPI global calculé |
| `matched_tables` | int | Nombre de tableaux appariés T1↔T2 |
| `total_indicators_added` | int | Total indicateurs ajoutés dans T2 |
| `total_indicators_removed` | int | Total indicateurs supprimés dans T2 |
| `total_indicators_renamed` | int | Total indicateurs renommés |
| `tables_added` | int | Tableaux présents uniquement en T2 |
| `tables_removed` | int | Tableaux présents uniquement en T1 |
| `active_filters` | dict | Filtres actifs (section, statut) |

---

## 9. Modèles de données

### `TableFile` (dataclass)
```python
filename: str
section: str
quarter: str
page_number: int
table_order: int
full_path: str
```

### `ReviewItem` (dataclass)
```python
item_id: str
section: str
table_type: int          # 1 ou 2
indicator_name: str
q1_file: str
q2_file: str
confidence_score: float
verdict: str             # "AJOUT VRAI" | "SUPPRESSION VRAIE" | "FAUX POSITIF" | "INCERTAIN"
status: str              # "en_attente" | "valide" | "rejete"
analyst_decision: str | None
analyst_comment: str | None
has_indicator_change: bool
```

### `AnalysisReport` (dict structure)
```python
{
  "_metadata": {
    "bank": str,
    "year": int,
    "q1": str,
    "q2": str,
    "analyst": str,
    "total_q1_tables": int,
    "total_q2_tables": int,
    "matched": int,
    "match_rate": float,
    "kpi": float,
    "total_indicators_added": int,
    "total_indicators_removed": int,
    "total_indicators_renamed": int,
    "tables_added": int,
    "tables_removed": int
  },
  "{section_name}": {
    "changements_indicateurs": [ReviewItem],
    "tables_ajoutees": [...],
    "tables_supprimees": [...]
  }
}
```

---

## 10. Palette de couleurs & styles

| Élément | Couleur |
|---------|---------|
| Header / Footer fond | `#1B3A6B` (bleu marine) |
| Header / Footer texte | `#FFFFFF` |
| Bouton Valider | `#28A745` (vert) |
| Bouton Rejeter | `#DC3545` (rouge) |
| Bouton Appliquer / Primary | `#1B3A6B` (bleu marine) |
| Badge "En Attente" | `#FD7E14` (orange) |
| Badge "Validé" | `#DC3545` ou `#28A745` selon contexte |
| Ligne sélectionnée | `#E8F0FE` (bleu très clair) |
| Score confiance élevée (>0.8) | `#28A745` |
| Score confiance moyenne (0.5–0.8) | `#FD7E14` |
| Score confiance faible (<0.5) | `#DC3545` |
| Fond sidebar | `#F8F9FA` (gris très clair) |

---

## 11. Palette de widgets Streamlit

| Widget | Usage |
|--------|-------|
| `st.set_page_config` | Titre, layout wide, favicon BNC |
| `st.markdown` + HTML/CSS | Header custom, footer, badges colorés |
| `st.sidebar` | Contexte d'analyse (configuration) |
| `st.selectbox` | Banque |
| `st.text_input` | Année, T1, T2, Analyste, Clé API |
| `st.radio` | Source des données |
| `st.tabs` | Navigation Revue / Export |
| `st.columns` | Mise en page à 2-3 colonnes |
| `st.container(border=True)` | Cartes pour chaque item de la file |
| `st.dataframe` | Tableau de la file de revue (interactif) |
| `st.checkbox` | Filtres de section/statut |
| `st.image` | Aperçu des tableaux PNG côte à côte |
| `st.button` | Valider, Rejeter, Appliquer, Précédent, Suivant |
| `st.text_area` | Commentaire analyste |
| `st.progress` | Barre KPI |
| `st.metric` | Compteurs (total, validés, rejetés) |
| `st.success` / `st.error` / `st.warning` / `st.info` | Verdict système |
| `st.download_button` | Export JSON / CSV |
| `st.file_uploader` | Import JSON / PDN |
| `st.divider` | Séparateurs visuels |
| `st.spinner` | Indicateur pendant l'analyse IA |
| `st.balloons` | Animation succès analyse |

---

## 12. Flux de travail (User Flow)

```
1. Configuration sidebar
   └─ Banque, Année, T1, T2, Analyste, API Key, Source données

2. Lancement de l'analyse (bouton "Lancer l'Analyse")
   └─ Pipeline IA : matching → extraction indicateurs → vérification
   └─ Résultats stockés dans session_state.report_data

3. Revue des changements (Tab 2)
   ├─ File de revue : navigation item par item
   ├─ Filtres par section / statut
   └─ Décision analyste pour chaque item (Valider / Rejeter + commentaire)

4. Export & Audit (Tab 3)
   ├─ Export JSON rapport complet
   ├─ Export CSV résumé
   └─ Statistiques d'audit (taux validation, temps passé, KPI)
```

---

## 13. Conventions de nommage des fichiers images

- **CIBC / RBC / BNC:** `sectionName_quarter_pageNumber_tableOrder.png`
  - Ex. : `financial_results_q1_5_2.png`
- **Scotiabank:** `BANK_sectionName_p-pageNumber_tableOrder.png`
  - Ex. : `SCOTIA_Gestion_du_risque_p10_28.png`

---

## 14. Fichiers et dossiers attendus

- **Entrée :**
  - `{base_path}/{bank}_{year}_{q1}_table_images/` → fichiers `.png`
  - `{base_path}/{bank}_{year}_{q2}_table_images/` → fichiers `.png`
- **Sortie automatique :**
  - `{output_dir}/{bank}_{year}_{q1}_vs_{q2}_analysis.json`
  - `{output_dir}/{bank}_{year}_{q1}_vs_{q2}_decisions.json` (décisions analyste)

---

## 15. Notes d'implémentation

- Le **footer** est rendu avec `st.markdown` + CSS personnalisé (`position: fixed; bottom: 0`).
- Le **header** est rendu avec `st.markdown` + CSS pour masquer le header Streamlit par défaut et le remplacer.
- La **sélection d'item** dans la file de revue utilise `st.session_state.selected_item_index` mis à jour via `st.button` ou `st.dataframe` avec `on_select`.
- Les **décisions** sont sauvegardées en temps réel dans `st.session_state.decisions` et persistées en JSON sur disque à chaque action.
- Le **KPI** est recalculé à chaque décision : `kpi = (validés + rejetés) / total_items`.
- L'affichage des images côte à côte utilise `use_column_width=True` pour s'adapter à la largeur du panneau.

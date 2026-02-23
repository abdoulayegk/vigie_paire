# Plan d’implémentation — Matching robuste des tableaux (T1 ↔ T2) à partir des JSON Docling

Contexte : tu disposes de deux fichiers JSON (T1 et T2) extraits avec Docling, limités aux sections pertinentes.
Objectif : **matcher les tableaux 1↔1** entre T1 et T2 de manière fiable, **sans dépendre** de `page_number` ni de `table_number` (qui peut être ambigu), et **réduire drastiquement** les faux positifs.

---

## 1) Principes de design (à respecter)

- **Ne jamais matcher sur les valeurs** (chiffres, montants, ratios) : elles changent naturellement.
- Utiliser en priorité la **structure métier** :
  - libellés de la **1ère colonne** (`first_column[*].text_norm`)
  - éventuellement `features.anchors`, `features.n_indicators`, titre normalisé
- `table_number` = **signal faible** (uniquement pour réduire les candidats, jamais comme clé).
- Matching final **1↔1** via **assignment global** (Hungarian / min-cost), pas « greedy ».

---

## 2) Entrées / Sorties

### Entrées

- `td_2025t1_extraction.json` (T1)
- `td_2025t2_extraction.json` (T2)

Chaque table contient typiquement :

- `section` (via le conteneur `sections.<name>.tables[]`)
- `table_uid`, `table_id`
- `table_title`, `headers[]`
- `first_column[]` avec `text_norm` (+ parfois `note_refs`)
- `features` : `n_indicators`, `indicator_set_hash`, `anchors[]`, etc.

### Sorties attendues

Un objet (ou fichier) `matching_result.json` avec :

- `matches[]` : liste de paires (T1_table_uid ↔ T2_table_uid) + score + explication
- `added_tables[]` : tables présentes en T2 sans match
- `removed_tables[]` : tables présentes en T1 sans match
- `uncertain[]` : paires candidates zone grise (à valider)
- `diagnostics[]` : raisons de non-match / faux split suspect / collisions

---

## 3) Pipeline d’implémentation (vue d’ensemble)

1. **Chargement + indexation** des tables par section
2. **Normalisation** des titres et libellés (dénormalisation des footnotes)
3. **Construction d’une signature** table (features dérivées)
4. **Génération des candidats** (bloquer par section, filtrer intelligent)
5. **Scoring multi-signal** table↔table
6. **Assignment 1↔1** (Hungarian) + seuils (match / uncertain / reject)
7. **Post-traitements** :
   - added/removed
   - détection split/fusion (diagnostic)
   - logs et exports

---

## 4) Étape 1 — Chargement et structuration des données

### 4.1. Lire les JSON

- `doc = json.load(...)`

### 4.2. Transformer en structure interne

Créer une structure uniforme :

```json
{
  "section": "gestion_capital",
  "table_uid": "...",
  "table_title": "...",
  "headers": [...],
  "first_col_norm": ["...", "..."],
  "anchors": ["...", "..."],
  "n_indicators": 12,
  "indicator_set_hash": "sha1:...",
  "raw": { "...": "..." }
}
```

---

## 5) Étape 2 — Normalisation (critical)

### 5.1. Nettoyer les footnote markers dans les labels

But : éviter les faux positifs « Actions ordinaires » vs « Actions ordinaires 2 ».

Règles minimales :

- supprimer **chiffres isolés en fin de label** (`\s+\d+$`)
- supprimer exposants finaux (`¹²³…`)
- supprimer symboles de note finaux (`*`, `†`, `‡`) si présents
- garder les nombres **métier** quand ils font partie du sens (ex. `Série 2023-9`, `10 %`, `Bâle III`)

Produire :

- `raw_label`
- `clean_label` (utilisé pour matching)
- `footnote_refs` (conservé pour comparaison séparée)

### 5.2. Normaliser le titre

- lowercase + enlever accents
- supprimer `TABLEAU 24`, `Page 37`, `Premier/Deuxième trimestre`, dates « Au 30 avril 2025 »
- réduire espaces

---

## 6) Étape 3 — Signature et features dérivées

Pour chaque table :

### 6.1. Ensembles / fingerprints

- `label_set` : set des `clean_label`
- `anchor_set` : set(`features.anchors`) si dispo (sinon vide)
- `title_tokens` : set de tokens significatifs du titre normalisé (min 3 chars)
- `size` : `n_indicators` (ou len(first_col_norm))

### 6.2. Option utile : top-k anchors

Garder les anchors les plus distinctifs (ex. éviter `total`, `actifs` si trop fréquents) :

- DF (document frequency) par section
- ignorer les anchors ultra fréquents

---

## 7) Étape 4 — Génération des candidats (réduction combinatoire)

Pour chaque table T1 dans une section donnée :

- Candidats T2 = tables **dans la même section**
- Si `table_number` existe en T1 :
  - **prioriser** les candidats T2 avec le même `table_number`
  - mais si plusieurs (collision), on départage au score
  - si aucun bon score, on élargit à tout T2 de la section
- Si `indicator_set_hash` existe et match exact :
  - candidat unique (score quasi certain)
  - on garde quand même une validation (sanity) au score

---

## 8) Étape 5 — Scoring multi-signal (table ↔ table)

### 8.1. Signaux recommandés

1. **Soft overlap des labels** (poids fort)
   - pour chaque label de T1, trouver le meilleur match dans T2 (fuzzy `token_set_ratio`), moyenne
   - faire aussi T2→T1, puis moyenne des deux (symétrique)
2. **Jaccard anchors** (poids moyen)
3. **Similarité titre** (poids faible)
4. **Similarité taille** (poids faible)

### 8.2. Score final (proposition)

`score = 0.70*s_labels + 0.15*s_anchors + 0.10*s_title + 0.05*s_size`

Règles “garde-fous” (anti faux match) :

- si `s_labels < 0.55` → refuser le match, même si titre proche
- si `abs(n1-n2)/max(n1,n2) > 0.60` → pénaliser fortement

---

## 9) Étape 6 — Assignment 1↔1 (Hungarian)

### Pourquoi ?

Empêche :

- une table T2 « attractive » d’être matchée à plusieurs tables T1
- la propagation de faux positifs

### Méthode

- Construire matrice `S[i,j] = score(T1_i, T2_j)` (par section)
- Coût = `1 - S`
- Hungarian (min-cost) → mapping 1↔1 optimal
- Appliquer seuils :
  - **match sûr** : `score ≥ 0.65`
  - **incertain** : `0.55 ≤ score < 0.65`
  - **reject** : `< 0.55`

---

## 10) Étape 7 — Post-traitements et diagnostics

### 10.1. Added / Removed

- `removed` = tables T1 non matchées (après seuil)
- `added` = tables T2 non matchées

### 10.2. Détection split/fusion (diagnostic, option)

But : repérer quand Docling segmente 1 table en 2.

Heuristique :

- Une table T1 a deux candidats T2 avec `s_labels` modérés (ex. ~0.45 chacun)
- et `labels(T2a) ∪ labels(T2b)` couvre bien `labels(T1)` (coverage > 0.80)
  → `diagnostic: split probable`

### 10.3. Expliquer chaque match (audit-ready)

Pour chaque match, conserver :

- score total + sous-scores
- top 5 labels communs
- top 5 labels non retrouvés
- taille T1 vs T2

---

## 11) Comparaison des indicateurs (après le matching de tables)

Une fois le match table↔table décidé :

- comparer les labels (clean) en mode set :
  - `added_indicators = T2 - T1`
  - `removed_indicators = T1 - T2`
- pour « rename » :
  - chercher des paires proches (fuzzy) parmi added/removed avec score > 0.85
- garder `raw_label` et `footnote_refs` séparément :
  - la présence/absence d’une note ne doit pas créer un faux ajout/suppression d’indicateur

---

## 12) Stratégie de validation (fiabilité)

### 12.1. Bench minimal

Sur une banque (TD) :

- matcher 1 section
- échantillonner 10 matches (manuel)
- mesurer :
  - precision = % matches corrects
  - recall = % tables réellement matchées

### 12.2. Logs à produire

- taux de match par section
- distribution des scores
- top 10 « uncertain »
- top 10 « collisions » (plusieurs candidats proches)

---

## 13) Checklist “production-ready”

- [ ] Normalisation labels + extraction footnotes (no false positives)
- [ ] Score multi-signal + garde-fous
- [ ] Hungarian 1↔1
- [ ] seuils match/uncertain/reject
- [ ] diagnostics split/fusion
- [ ] exports JSON + logs + rapport synthèse

---

## 14) Paramètres initiaux recommandés (à ajuster)

- `threshold_match = 0.65`
- `threshold_uncertain = 0.55`
- `min_label_score_gate = 0.55`
- poids : `labels 0.70`, `anchors 0.15`, `title 0.10`, `size 0.05`

Pour RBC/CIBC :

- baisser `threshold_match` à 0.60 (au début)
- activer diagnostics split/fusion plus agressifs
- renforcer `labels` (0.75–0.85) et réduire `title`

---

## 15) Livrables à coder (ordre)

1. `normalize.py`
   - `clean_label()`, `normalize_title()`, `parse_label()`
2. `features.py`
   - `build_table_features(table)`
3. `scoring.py`
   - `soft_overlap()`, `jaccard()`, `table_score()`
4. `matching.py`
   - `build_score_matrix()`, `hungarian_assign()`, `apply_thresholds()`
5. `diagnostics.py`
   - `detect_splits()`, `explain_match()`
6. `run_match.py`
   - CLI: `python run_match.py --t1 ... --t2 ... --out matching_result.json`

---

### Notes importantes

- Cette approche est **déjà justifiée sur TD** : tu as des matches parfaits via `indicator_set_hash` (ex. Tableau 24) et des matches robustes quand le hash change (Bâle III) grâce aux labels + anchors.
- Elle est **généralisable** : RBC/CIBC demandent surtout le diagnostic split/fusion + seuils ajustés.

# Documentation technique - Matching des tableaux T1/T2 et reduction des faux positifs

Ce document decrit la procedure complete de matching des tableaux entre rapports T1 et T2 (ex. trimestres successifs), ainsi que tous les mecanismes mis en place pour reduire les faux positifs.

---

## 1. Contexte et objectifs

### 1.1 Problematique

Les rapports financiers trimestriels contiennent des tableaux structures (capital, risques, etc.). La comparaison T1 vs T2 necessite d'identifier **quels tableaux correspondent** entre les deux periodes. Les valeurs numeriques changent naturellement ; le matching doit reposer sur la **structure metier** : libelles de la premiere colonne, titres, schema.

### 1.2 Contraintes

- Ne jamais matcher sur les **valeurs** (chiffres, montants, ratios).
- `table_number` est un **signal faible** : utile pour prioriser, jamais comme cle unique.
- Eviter les faux positifs (match errone entre tableaux differents).
- Assignment **1-vers-1** : une table T2 ne peut matcher qu'une seule table T1.

---

## 2. Pipeline general

```
Chargement JSON (vigie_extract)
        |
        v
Indexation par section (capital_management, risk_management, etc.)
        |
        v
Normalisation des labels et titres
        |
        v
Construction des features (anchors, indicator_set_hash)
        |
        v
Pour chaque section : candidats T2 dans la meme section
        |
        v
Scoring multi-signal (match_decision)
        |
        v
Assignment 1-vers-1 (Hungarian)
        |
        v
Post-traitement : rescue, diagnostics, added/removed
```

---

## 3. Normalisation des labels

### 3.1 Labels canoniques (_canonical_indicator_label)

Chaque libelle de premiere colonne est normalise avant comparison :

1. **Exclusions** : lignes date seule, totaux, unites (`is_date_only_line`, `is_non_indicator_line`).
2. **Normalisation** : accents, minuscules, ponctuation via `normalize_label`.
3. **Strip des notes** : suppression des chiffres finaux non semantiques (ex. "Actions ordinaires 2" -> "actions ordinaires") sauf si semantique (ex. "Bale III", "Serie 2023-9").

### 3.2 Alignement anchors / indicator_set

Les **anchors** et l'**indicator_set** utilisent la meme normalisation (labels canoniques) pour coherence entre :

- Jaccard sur les labels (`_jaccard`)
- Jaccard sur les anchors (`_jaccard_anchors`)
- Score plan et diagnostics split/fusion

L'**indicator_set_hash** reste calcule via vigie_extract pour le fast path et la compatibilite.

---

## 4. Signaux et scoring (plan Phases 1-6)

### 4.1 Features derivees

Pour chaque table :

| Feature | Description | Utilisation |
|---------|-------------|-------------|
| `anchors` | Labels canoniques (via `_indicator_set`) | s_anchors, Jaccard |
| `indicator_set_hash` | Hash SHA1 des indicateurs (vigie_extract) | Fast path |
| `indicator_set` | Set des labels canoniques | Overlap, containment |

### 4.2 Fast path - indicator_set_hash (Phase 2)

Si les hashes sont identiques et qu'il n'y a pas de conflit de `table_number` :

- Match automatique avec score 1.0.
- Raison : `indicator_set_hash_exact`.
- Ne s'applique que dans une **section connue** (`same_known`).

### 4.3 Signaux principaux

| Signal | Poids defaut | Description |
|--------|--------------|-------------|
| `s_labels` | 0.70 | Overlap soft des labels (fuzzy token_set_ratio) |
| `s_anchors` | 0.15 | Jaccard des anchors (labels canoniques) |
| `s_title` | 0.10 | Similarite des titres normalises |
| `s_size` | 0.05 | Similarite structurelle (nombre d'indicateurs) |

Formule retenue (active par defaut avec `use_plan_score_formula: true`) :

```
score = w_labels * s_labels + w_anchors * s_anchors + w_title * s_title + w_size * s_size
```

- `s_labels` : overlap effectif des libelles (Jaccard + soft fuzzy).
- `s_anchors` : Jaccard des ancres (labels canoniques).
- `s_title` : similarite des titres normalises.
- `s_size` : similarite structurelle (colonnes/lignes).

Poids par defaut : `weight_s_labels` 0.70, `weight_s_anchors` 0.15, `weight_s_title` 0.10, `weight_s_size` 0.05. Les overrides par banque peuvent utiliser ces cles ou les anciennes (`weight_label_overlap` -> `weight_s_labels`, `weight_title` -> `weight_s_title`, `weight_structure` -> `weight_s_size`) pour compatibilite.

---

## 5. Mecanismes anti-faux positifs

### 5.1 Blocage cross-section

Les tableaux de **sections differentes connues** ne peuvent jamais matcher.

- Raison : `cross_section_forbidden`.
- Exemple : Capital vs Risques = rejet systematique.

### 5.2 Conflit table_number

Si les deux tables ont un `table_number` different (ex. Tableau 10 vs Tableau 28) :

- Le hash exact est **ignore** (pas de fast path).
- Le match est **refuse** sauf titre tres specifique (override strict).

Override possible uniquement si :

- `title_similarity >= 0.85`
- Corps du titre tres similaire (`body_similarity >= 0.92`)
- Corps assez specifique (longueur, nombre de mots)
- `effective_label_overlap >= overlap_threshold`
- `structure_similarity >= 0.50`
- Difference de taille bornee : `|#T1 - #T2| / max(#T1, #T2) <= title_override_max_size_ratio` (defaut 0.25), pour eviter de matcher un grand tableau avec un petit malgre le titre.

### 5.3 Garde-fou overlap minimal

- `overlap_floor_min` : seuil minimal d'overlap (defaut 0.35).
- `min_label_overlap_reject` : si le score indique match mais `soft_indicator_overlap < 0.55` -> rejet.
- Raison : `low_label_overlap_reject`.

### 5.4 Garde-fou taille (Phase 4)

Si `|n1 - n2| / max(n1, n2) > 0.60` (ratio de difference de taille) :

- Rejet du match.
- Raison : `size_mismatch_reject`.

### 5.5 Validation post-matching (GenAI / Vision)

Les validateurs post-matching (Vision pour paires de tableaux, GenAI pour renommages d'indicateurs)
sont configurables via la section `validation` dans `bank_profiles.yaml`. Voir
[docs/comparison-validation.md](comparison-validation.md) pour le schema et les valeurs recommendees.

### 5.6 Anti-greedy margin (uncertain_competition)

Quand les **deux meilleurs candidats** ont un score tres proche (ecart < `margin_threshold` = 0.10) :

- Aucun des deux n'est choisi.
- La table T1 est marquee `unmatched` avec raison `uncertain_competition`.
- Evite de trancher arbitrairement entre deux candidats ambigus.

### 5.6 Seuils match / probable / reject

| Niveau | Score min | Condition supplementaire |
|--------|-----------|---------------------------|
| Match | 0.70 (`match_score_v2`) | Pas de rejet par garde-fou |
| Probable | 0.62 (`probable_score_v2`) | `indicator_containment >= 0.45` |
| No match | - | Sinon |

### 5.7 Section inconnue (unknown_section)

- Penalite : `unknown_section_penalty` = 0.15 sur le score.
- Exigence renforcee : `indicator_containment >= 0.65` et `score >= 0.74` pour valider un match (configurable : `unknown_match_min_containment`, `unknown_match_min_score` ; defauts renforces 0.72 et 0.78).
- Signal fort requis : en section inconnue, un match doit en outre satisfaire **au moins une** des conditions : `title_similarity >= unknown_match_min_title_similarity` (defaut 0.75) **ou** `structure_similarity >= unknown_match_min_structure` (defaut 0.65). Sinon raison `unknown_section_penalized`.

### 5.8 Regles permissives par banque (RBC, CIBC)

Pour **RBC** et **CIBC**, les regles suivantes sont plus permissives et peuvent augmenter le risque de faux positifs sur certains cas :

- **Few indicators** : pour les tableaux avec au plus 6 indicateurs (parametre `few_indicators_max_count`), un plancher d'overlap plus bas (`overlap_floor_min_few_indicators` = 0.25) est utilise, ce qui permet de matcher des tableaux petits avec moins de preuves.
- **Header/footer titles** : les titres de type en-tete/pied de page (ex. "24 Banque Royale du Canada Premier trimestre...") beneficient du meme plancher reduit quand les deux tableaux ont ce type de titre.

Ces regles peuvent etre assouplies (seuils plus stricts) apres un audit manuel des faux positifs par banque. Un parametre optionnel `few_indicators_require_title_match` pourrait exiger un titre tres proche pour valider un match lorsqu'on utilise le plancher 0.25.

---

## 6. Assignment 1-vers-1 (Hungarian)

### 6.1 Principe

Le matching utilise **Hungarian par section connue**, puis **greedy sur le reste** (sections inconnues et tables residuelles). Pour chaque section connue, une matrice de scores `S[i,j]` est construite entre tables T1 et T2. L'algorithme Hungarian (maximisation du score total via `linear_sum_assignment`) determine l'assignation optimale 1-vers-1. Les tables restantes (sections inconnues ou non couvertes) sont traitees par un fallback greedy avec contrainte 1-vers-1 (`used_t2_uids`).

### 6.2 Effets

- Une table T2 "attractive" ne peut etre matchée qu'a une seule table T1.
- Limite la propagation de faux positifs.

### 6.3 Hungarian post-seuil (optionnel)

Un mode alternatif (`use_post_hungarian_threshold: true`) applique les seuils **apres** l'Hungarian pour une optimisation globale :

1. **Matrice** : toutes les paires non bloquees par les garde-fous reçoivent leur score reel (y compris sous 0.70). Seules les paires bloquees (cross_section, table_number_conflict, low_label_overlap_reject, size_mismatch_reject) ont score = -inf.

2. **Hungarian** : l'algorithme maximise la somme totale sur toutes les paires admissibles.

3. **Post-filtrage** : pour chaque paire assignee, application des seuils :
   - `score >= match_score_v2` (0.70) -> match
   - `score >= probable_score_v2` (0.62) -> probable
   - `score < hungarian_min_score` (0.62) -> rejet, liberation pour rescue

4. **uncertain_competition** : conserve comme en mode standard.

Parametres (`configs/bank_profiles.yaml`, section `matching_thresholds`) :

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `use_post_hungarian_threshold` | false | Activer le mode Hungarian puis seuil |
| `hungarian_min_score` | 0.62 | Seuil minimal apres Hungarian (rejet en dessous) |

---

## 7. Rescue et post-traitement

### 7.1 Single rescue

Pour les tables T1 non matchées apres l'Hungarian :

- Recherche du meilleur candidat T2 restant (meme section ou compatible).
- Match si `indicator_containment >= 0.65` et `title_similarity >= 0.40`.
- Raison : `single_rescue`.

### 7.2 Split/merge rescue

- **Split** : une table T1 correspond a l'union de deux tables T2.
- **Merge** : deux tables T1 correspondent a une table T2.

Conditions :

- `union_containment >= 0.80` (labels T1 couverts par union T2 ou inverse).
- `header_schema_similarity >= 0.65`.

---

## 8. Diagnostic split probable

### 8.1 Heuristique

Une table T1 non matchée peut etre un **split** : son contenu est reparti sur deux tables T2.

Detection :

- Parmi les candidats T2 (top K dans `debug_unmatched_candidates`, K = `split_diagnostic_max_candidates`, defaut 5), chercher exactement 2 candidats avec :
  - `s_labels` dans la bande [0.35, 0.55] (ni trop haut, ni trop bas).
- L'union des labels des 2 candidats couvre >= 80% des labels T1.
- Limitation : un split dont les deux candidats T2 sont au-dela du top K ne sera pas detecte.

Si oui : diagnostic `split_probable` avec les deux t2_uid concernes.

### 8.2 Limitation importante

Le diagnostic s'appuie sur **debug_unmatched_candidates**, limite aux **3 meilleurs candidats** par score. Un split impliquant des candidats classes au-dela du top 3 ne sera **pas detecte**. C'est une limitation a documenter pour le tuning.

---

## 9. Explain match (Phase 5)

Pour chaque match, option `include_explanation` :

- Score total et sous-scores (labels, anchors, title, size).
- Top 5 labels communs.
- Top 5 labels manquants dans T2.
- Top 5 labels manquants dans T1.
- Nombre d'indicateurs par table.

---

## 10. Configuration et seuils

### 10.1 Fichier de configuration

`configs/bank_profiles.yaml` : section `matching_thresholds` ou `matching.thresholds`.

Overrides par banque : `banks.<code>.matching_overrides`.

### 10.2 Seuils par defaut (_DEFAULTS)

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `overlap_threshold` | 0.55 | Seuil overlap pour candidat valide |
| `overlap_floor_min` | 0.35 | Plancher overlap |
| `margin_threshold` | 0.10 | Anti-greedy : ecart min entre top 2 |
| `match_score_v2` | 0.70 | Score minimum pour match |
| `probable_score_v2` | 0.62 | Score minimum pour probable |
| `min_label_overlap_reject` | 0.55 | Rejet si overlap trop faible |
| `size_mismatch_reject_threshold` | 0.60 | Rejet si ratio taille > 60% |
| `use_post_hungarian_threshold` | false | Mode Hungarian puis seuil (optimisation globale) |
| `hungarian_min_score` | 0.62 | Seuil minimal apres Hungarian pour accepter une paire |
| `unknown_section_penalty` | 0.15 | Penalite section inconnue |
| `unknown_match_min_containment` | 0.65 | Containment min en section inconnue |
| `unknown_match_min_score` | 0.74 | Score min en section inconnue |

### 10.3 Poids plan (formule score)

| Parametre | Defaut |
|-----------|--------|
| `weight_s_labels` | 0.70 |
| `weight_s_anchors` | 0.15 |
| `weight_s_title` | 0.10 |
| `weight_s_size` | 0.05 |

---

## 11. Structure du code

| Module / fonction | Rôle |
|-------------------|------|
| `indicator_comparator.py` | Comparateur principal, match_decision, Hungarian |
| `_get_table_features` | Anchors + indicator_set_hash |
| `_compute_pair_score_with_guard_rails` | Score brut + garde-fous (pour mode post-seuil) |
| `_indicator_set` | Labels canoniques d'une table |
| `match_decision` | Decision match/probable/no_match |
| `match_tables_intra_section` | Pipeline par section |
| `run_strict_intra_section_compare` | Point d'entree avec rescue et diagnostics |
| `vigilance.config.get_matching_thresholds` | Chargement des seuils |
| `vigie_extract_schema` | Hash et parse first column |

### 11.1 Integration Dash

La **source de verite** pour l'UI est le resultat de `run_strict_intra_section_compare` via `app.comparison_runner.run_comparison_with_sections` : le payload canonique utilise strictement `strict["pairs"]`, `strict["added_tables"]`, `strict["removed_tables"]`. Chaque entree de `table_comparisons` provient d'une paire retournee par le moteur et expose `match_reason`, `rescue_type`, `match_score`. Les metadonnees du payload incluent `algorithm_used` (hungarian ou greedy). Aucune logique de matching parallele n'est utilisee dans l'app Dash ; tout passe par le comparateur strict.

---

## 12. Raisons de match / non-match (reference)

| Raison | Signification |
|--------|---------------|
| `indicator_set_hash_exact` | Hash identique, match certain |
| `table_number_match` | Numero de tableau + overlap OK |
| `indicator_overlap_match` | Overlap suffisant, pas de conflit |
| `title_override_match` | Override malgre conflit table_number |
| `table_number_low_overlap_rescue` | Titre/structure fort malgre overlap faible |
| `date_title_structure_rescue` | Titres date-only tres similaires |
| `single_rescue` | Rescue apres Hungarian |
| `split_merge_rescue` | Match split ou merge |
| `cross_section_forbidden` | Sections differentes |
| `table_number_conflict` | Numeros differents, pas d'override |
| `low_label_overlap_reject` | Overlap trop faible |
| `size_mismatch_reject` | Difference de taille trop forte |
| `unknown_section_penalized` | Section inconnue, criteres non atteints |
| `uncertain_competition` | Anti-greedy : top 2 trop proches |
| `generic_title_insufficient_signals` | Les deux titres sont generiques et containment/score insuffisants |

---

## 13. Validation et tuning

### 13.1 Bench et metriques (recommandation)

- **Jeu de reference** : constituer pour chaque banque (RBC, TD, CIBC, BNS au minimum) un echantillon de paires T1-T2 etiquetees manuellement (correct / incorrect). Format suggere : JSON listant `t1_uid`, `t2_uid`, `expected_match` (bool).
- **Script** : lancer le matching (ex. via `run_strict_intra_section_compare` sur les memes PDF/sections), comparer les paires produites aux etiquettes, calculer precision (% des paires retournees qui sont correctes), rappel (% des vrais matches qui sont retrouves) et F1 par banque.
- **Non-regression** : avant de valider des changements de seuils (phase 1-2), exiger une non-regression (ou un objectif cible) sur ce bench.

### 13.2 Autres

- **Logs** : distribution des scores, top "uncertain", collisions.
- **Par banque** : RBC/CIBC peuvent necessiter des seuils plus bas et un poids labels plus eleve (0.75-0.85).

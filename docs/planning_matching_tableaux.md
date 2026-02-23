# Plan : matching des tableaux et réduction des faux positifs

## Objectif

- **Matcher les tableaux** T1 vs T2 de façon fiable (même section, même tableau logique).
- **Réduire les faux positifs** (paires matchées à tort) tout en limitant les faux négatifs (vrais tableaux non matchés).

## État actuel (rappel)

- **Pipeline officiel** : `vigilance.compare` → `match_tables_intra_section` + `run_strict_intra_section_compare`.
- **Règles** : strict intra-section, Jaccard sur première colonne, numéro de tableau, titre, structure, contexte, page.
- **Problème potentiel** : le comparateur utilise `table.rows` pour les indicateurs ; en mode `labels_only`, `rows` est vide alors que `first_column_indicators` est rempli → Jaccard = 0.
- **Double couche** : `compare` (officiel) vs `comparison` (plusieurs matchers non unifiés).

---

## Phase 1 – Aligner extraction et comparateur (priorité haute)

| # | Action | Fichier / lieu | Détail |
|---|--------|----------------|--------|
| 1.1 | Utiliser `first_column_indicators` dans le calcul Jaccard quand `rows` vide | `compare/indicator_comparator.py` | Dans `_indicator_set(table)`, si `not table.rows` et `table.first_column_indicators`, construire l’ensemble à partir de `first_column_indicators` (normaliser en minuscules comme pour `row[0]`). Sinon garder la logique actuelle sur `rows`. |
| 1.2 | Propager `table_number` vers `TableArtifact` | `models/table_models.py` + `run_tables._to_artifacts` | Ajouter un champ optionnel `table_number: str | None` sur `TableArtifact`. Dans `_to_artifacts`, remplir depuis `getattr(table, "table_number", None)`. Le comparateur pourra l’utiliser en priorité par rapport à l’extraction regex du titre. |
| 1.3 | Optionnel : alimenter le comparateur depuis vigie_extract_v1 | Nouveau helper ou CLI | Si les JSON vigie_extract deviennent la source de vérité, ajouter une fonction qui charge un JSON vigie_extract et produit une liste de `TableArtifact` (ou un adaptateur) pour `match_tables_intra_section`. |

**Livrable** : le matching fonctionne correctement même en extraction labels_only, et peut s’appuyer sur `table_number` quand disponible.

---

## Phase 2 – Réduire les faux positifs (règles et seuils)

| # | Action | Fichier / lieu | Détail |
|---|--------|----------------|--------|
| 2.1 | Renforcer le seuil minimal d’overlap indicateurs | `compare/indicator_comparator.py` | Garder `overlap_floor = min(overlap_threshold, 0.30)` mais envisager de monter le défaut de `overlap_threshold` (ex. 0.55) ou d’ajouter un seuil absolu minimal (ex. 0.35) en dessous duquel aucun match n’est accepté, même avec titre/structure forts. |
| 2.2 | Exiger une contrainte supplémentaire pour les matchs “indicator_overlap_match” | `compare/indicator_comparator.py` | Pour le motif `indicator_overlap_match`, exiger en plus soit un accord sur le numéro de tableau (base), soit une similarité de titre ≥ seuil (ex. 0.50). Éviter de matcher deux tableaux très différents par titre uniquement grâce à un fort Jaccard (ex. tableaux génériques). |
| 2.3 | Resserrer la marge d’ambiguïté (anti-greedy) | `compare/indicator_comparator.py` | Augmenter `margin_threshold` (ex. de 0.07 à 0.10) pour refuser plus souvent les matchs lorsque deux candidats sont trop proches en score → moins de décisions hasardeuses. |
| 2.4 | Utiliser `table_number` en entrée du comparateur | `compare/indicator_comparator.py` | Si `TableArtifact` a `table_number`, l’utiliser en priorité pour `_extract_table_label` (ou une variante) au lieu de parser le titre. Réduit les erreurs d’extraction de numéro et améliore la cohérence. |
| 2.5 | Configurer les seuils par YAML | `config/` + `compare/indicator_comparator.py` | Exposer `overlap_threshold`, `overlap_floor`, `title_similarity_min`, `margin_threshold` (et éventuellement les seuils de `match_decision`) dans la config (ex. `get_matching_thresholds()` ou nouveau bloc `compare`) pour tuner sans toucher au code. |

**Livrable** : moins de paires incorrectes, seuils ajustables par config.

---

## Phase 3 – Qualité et observabilité (recommandé)

| # | Action | Fichier / lieu | Détail |
|---|--------|----------------|--------|
| 3.1 | Logger les matchs “limites” | `compare/indicator_comparator.py` | Pour chaque paire matchée, si le score est dans une bande basse (ex. 0.50–0.65) ou si la raison est `indicator_overlap_match` / `multi_signal_match`, logger un warning avec section, table_id, score et indicateurs en commun. Facilite l’audit des faux positifs. |
| 3.2 | Exposer les raisons dans la sortie | Déjà partiel | S’assurer que chaque entrée de `pairs` et `unmatched_*` contient bien `reason` et si possible `score`, pour analyse a posteriori et jeu de vérité. |
| 3.3 | Tests de non-régression sur jeux de paires | `tests/` | Ajouter des tests (ex. `test_compare_indicator_comparator.py`) avec des paires T1/T2 connues (match attendu / non-match attendu) pour éviter de casser le comportement en modifiant les seuils. |

**Livrable** : traçabilité des décisions et tests pour les changements de seuils.

---

## Phase 4 – Consolidation (optionnel, moyen terme)

| # | Action | Détail |
|---|--------|--------|
| 4.1 | Clarifier le rôle de `comparison` vs `compare` | Doc ou refactor | Documenter que `compare` est le chemin officiel pour T1/T2 ; `comparison` reste pour comparateurs spécialisés (structurel, preview, etc.). Éviter d’ajouter de nouveaux “match_tables” divergents dans `comparison` sans les brancher sur la même logique que `compare`. |
| 4.2 | Unifier la source des indicateurs | `compare` + `TableArtifact` | Partout dans le comparateur, utiliser une seule source “indicateurs” : soit dérivée de `rows` (première colonne), soit de `first_column_indicators` si rows vide (déjà prévu en Phase 1.1). |

---

## Ordre d’exécution suggéré

1. **Phase 1** (1.1, 1.2) – Sans cela, le matching peut être cassé en labels_only et sous-utilise les métadonnées.
2. **Phase 2** (2.1, 2.2, 2.4, 2.5) – Réduction directe des faux positifs et configurabilité.
3. **Phase 2** (2.3) – Ajustement fin de l’anti-greedy.
4. **Phase 3** (3.1, 3.2, 3.3) – Qualité et confiance.
5. **Phase 1** (1.3) + **Phase 4** – Si vous basculez sur vigie_extract comme source et souhaitez simplifier l’écosystème.

---

## Résumé des livrables

| Phase | Livrable principal |
|-------|---------------------|
| 1 | Comparateur alimenté par `first_column_indicators` si besoin ; `TableArtifact.table_number` utilisé pour le matching. |
| 2 | Seuils plus stricts et configurables ; moins de faux positifs. |
| 3 | Logs des matchs limites, raisons/score dans les sorties, tests de non-régression. |
| 4 | Rôle compare vs comparison clarifié ; une seule source d’indicateurs. |

Ce plan reste focalisé sur le matching des tableaux et la réduction des faux positifs, sans proposer d’algorithme de matching alternatif, en renforçant le pipeline actuel (`compare`) et en l’alignant sur l’extraction (vigie_extract / labels_only).

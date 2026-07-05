# Alignement des unités narratives du pipeline texte

## Objectif

Ce document décrit le fonctionnement actuel de l'alignement texte dans le pipeline narratif.

Le point important est le suivant:

**Le pipeline ne compare pas mécaniquement `Unit1` de T1 avec `Unit1` de T2.**

Quand une sous-section est découpée en unités narratives, le système cherche d'abord les unités qui portent le même sens, même si leur ordre a changé entre T1 et T2.

Cette logique sert à réduire les faux changements lorsque le rapport réordonne des paragraphes sans modifier le fond.

## Modules concernés

| Module | Rôle |
| --- | --- |
| `vigilance.text_analysis.subsection_units` | Découpe le markdown en sous-sections et en `NarrativeUnit`. |
| `vigilance.text_analysis.subsection_alignment` | Aligne les sous-sections, puis aligne les unités narratives quand c'est possible. |
| `vigilance.text_analysis.text_change_detection` | Exécute les comparaisons GPT à partir des tâches d'alignement produites. |
| `vigilance.text_analysis.change_records` | Produit les changements synthétiques `added` / `removed` sans appel GPT. |
| `vigilance.text_analysis.constants` | Porte les seuils techniques de découpage et d'alignement. |

## Vue d'ensemble

Le flux logique est:

```text
Markdown T1/T2
  -> découpage en sous-sections ###
  -> alignement des sous-sections T1/T2
  -> découpage du corps en NarrativeUnit
  -> alignement LLM des unités dans cette sous-section uniquement
  -> création de tâches de comparaison unitaires si le plan LLM est complet
  -> fallback sous-section complète si le matching est ambigu
  -> comparaison GPT
  -> enrichissement AMF et résumé
```

## Découpage en sous-sections

Le pipeline lit le markdown extrait des PDF et découpe sur les titres `###`.

Exemple:

```markdown
### Risques opérationnels

La Banque renforce la cybersécurité.

La Banque surveille les fournisseurs critiques.

### Risque de liquidité

La Banque maintient des sources diversifiées de financement.
```

Résultat:

```text
Sous-section 1: Risques opérationnels
Sous-section 2: Risque de liquidité
```

Si aucun `###` n'existe, le texte est traité comme une seule introduction `__intro__`.

## Découpage en unités narratives

Dans chaque sous-section, le corps est découpé en unités plus petites.

Le découpage respecte trois principes:

- les paragraphes séparés par des lignes vides deviennent des unités candidates;
- les puces deviennent des unités séparées;
- les très longs paragraphes sont découpés par phrases, sans couper au milieu d'une phrase.

Exemple:

```markdown
### Risques opérationnels

La Banque renforce la cybersécurité, la surveillance des menaces et les contrôles de sécurité de l'information.

La Banque surveille les fournisseurs critiques, les services impartis et les plans de continuité liés aux tiers.

La Banque améliore la gouvernance des données, la qualité des renseignements et la protection des renseignements personnels.
```

Résultat:

```text
T1 U1: cybersécurité
T1 U2: fournisseurs / tiers
T1 U3: données / renseignements personnels
```

Chaque unité conserve des métadonnées:

```text
section_key
heading
canonical_topic
unit_text
unit_index
word_count
char_len
hierarchy_path
```

## Alignement des sous-sections

Avant d'aligner les unités, le pipeline doit d'abord savoir quelles sous-sections comparer.

Exemple simple:

```text
T1: ### Risques opérationnels
T2: ### Risques opérationnels
```

Alignement:

```text
Risques opérationnels T1 -> Risques opérationnels T2
```

Si les titres changent, GPT peut proposer un alignement explicite:

```text
T1: ### Incidence des tarifs
T2: ### Incidence des tarifs douaniers
```

Alignement possible:

```text
Incidence des tarifs -> Incidence des tarifs douaniers
type: renamed
```

Sans alignement fiable, le pipeline ne force pas la comparaison entre titres différents.

## Alignement des unités narratives

Une fois deux sous-sections alignées, le pipeline peut descendre au niveau des unités narratives.

Le pipeline demande au LLM de produire un plan d'alignement borné à cette seule sous-section.

Le LLM reçoit:

```text
Grande section: gestion_risques
Sous-section T1: Risques opérationnels
Sous-section T2: Risques opérationnels
Unités T1: U1, U2, U3
Unités T2: U1, U2, U3
```

Il doit retourner:

```json
{
  "matches": [
    {
      "previous_unit_index": 1,
      "current_unit_index": 2,
      "confidence": "high",
      "reason": "Même sujet de cybersécurité."
    }
  ],
  "group_matches": [
    {
      "previous_unit_indexes": [3],
      "current_unit_indexes": [3, 4],
      "confidence": "high",
      "reason": "T2 découpe la même idée en deux paragraphes."
    }
  ],
  "removed_unit_indexes": [],
  "added_unit_indexes": [],
  "ambiguous_previous_unit_indexes": [],
  "ambiguous_current_unit_indexes": []
}
```

Règle centrale:

**Le LLM ne peut aligner que les unités fournies dans cette paire de sous-sections.**

Il ne peut pas aller chercher une unité dans une autre sous-section ou dans une autre grande section.

## Groupes d'unités locaux

Le découpage en unités n'est pas toujours identique entre deux rapports. Une idée peut être un seul paragraphe dans T1 et deux paragraphes dans T2.

Exemple:

```text
T1 U1: cybersécurité + gouvernance des données

T2 U1: cybersécurité
T2 U2: gouvernance des données
```

Le LLM peut retourner un groupe local:

```json
{
  "group_matches": [
    {
      "previous_unit_indexes": [1],
      "current_unit_indexes": [1, 2],
      "confidence": "high",
      "reason": "T2 sépare en deux unités le contenu combiné dans T1."
    }
  ]
}
```

Le pipeline construit alors une seule tâche de comparaison:

```text
body_t1 = T1 U1
body_t2 = T2 U1 + "\n\n" + T2 U2
alignment_type = llm_unit_group_match
```

Ce groupement reste strictement local à la sous-section alignée. Il ne réactive pas `split` ou `merged` entre sous-sections.

## Exemple: réordonnancement sans changement de fond

T1:

```text
U1: La Banque renforce la cybersécurité.
U2: La Banque surveille les fournisseurs critiques.
U3: La Banque améliore la gouvernance des données.
```

T2:

```text
U1: La Banque améliore la gouvernance des données.
U2: La Banque renforce la cybersécurité.
U3: La Banque surveille les fournisseurs critiques.
```

Matching incorrect à éviter:

```text
T1 U1 cyber   -> T2 U1 données
T1 U2 tiers   -> T2 U2 cyber
T1 U3 données -> T2 U3 tiers
```

Matching attendu:

```text
T1 U1 cyber   -> T2 U2 cyber
T1 U2 tiers   -> T2 U3 tiers
T1 U3 données -> T2 U1 données
```

Le test `test_build_comparison_tasks_aligns_narrative_units_by_llm_plan_not_position` couvre précisément ce scénario.

## Exemple: modification d'une unité

T1:

```text
U1: La Banque renforce la cybersécurité.
U2: La Banque surveille les fournisseurs critiques.
```

T2:

```text
U1: La Banque surveille les fournisseurs critiques avec des contrôles renforcés.
U2: La Banque renforce la cybersécurité.
```

Alignement:

```text
cyber T1 -> cyber T2
tiers T1 -> tiers T2
```

Comparaison attendue:

```text
cyber: unchanged
tiers: modified
```

Le changement sur les tiers est détecté parce que le sujet existe dans les deux versions, mais le contenu T2 ajoute des contrôles renforcés.

## Fallback conservateur

L'alignement unitaire n'est activé que si le plan LLM couvre toutes les unités de la sous-section.

Si le plan est incomplet ou ambigu, le pipeline revient à la comparaison complète de la sous-section.

Pourquoi?

Parce qu'une unité non appariée peut vouloir dire plusieurs choses:

- vrai ajout;
- vraie suppression;
- déplacement vers une autre sous-section;
- fusion avec un autre paragraphe;
- reformulation trop forte pour être alignée de façon fiable.

Dans ces cas, il est plus sûr de laisser GPT comparer la sous-section complète dans son contexte plutôt que de produire trop vite un `added` ou `removed`.

Exemple ambigu:

```text
T1 U1: cadre général de gestion des risques
T1 U2: contrôles internes

T2 U1: approche intégrée de gouvernance
T2 U2: nouvelle description opérationnelle
```

Le système ne force pas:

```text
U1 -> U1
U2 -> U2
```

Il revient à:

```text
Comparer toute la sous-section T1 avec toute la sous-section T2.
```

## Cas `split`, `merged` et `moved`

Les restructurations `split`, `merged` et `moved` sont rejetées dans le mode actuel.

La règle voulue est bloc-contre-bloc:

```text
Risque de marché T1 -> Risque de marché T2
Incidence des tarifs T1 -> Incidence des tarifs douaniers T2 si renommage fiable
```

Le pipeline évite:

```text
Description T1 -> Sécurité de l'information T2
Description T1 -> Risques liés aux tiers T2
Description T1 -> Risques géopolitiques T2
```

Exemple:

```text
T1:
### Description
cyber + tiers + géopolitique

T2:
### Sécurité de l'information
cyber

### Risques liés aux tiers
tiers

### Risques géopolitiques
géopolitique
```

Avant, GPT pouvait proposer:

```text
Description -> Sécurité de l'information
Description -> Risques liés aux tiers
Description -> Risques géopolitiques
type: split
```

Maintenant, ce plan est rejeté. Le résultat attendu devient:

```text
Description T1: removed
Sécurité de l'information T2: added
Risques liés aux tiers T2: added
Risques géopolitiques T2: added
```

Cela rend la sortie plus simple à auditer et évite qu'une ancienne section générale absorbe plusieurs nouvelles sous-sections.

## Changements synthétiques

Les changements synthétiques sont les changements produits sans appel GPT quand un élément est clairement ajouté ou supprimé.

Deux niveaux existent:

```text
Sous-section ajoutée/supprimée
Unité narrative ajoutée/supprimée
```

Le niveau unité est supporté dans le code, mais le comportement actuel reste conservateur: si l'alignement d'unités est incomplet, le pipeline préfère revenir à la comparaison de sous-section complète.

## Métadonnées produites

Quand une comparaison part au niveau unité, les changements produits peuvent recevoir:

```text
alignment_type
alignment_confidence
canonical_topic
previous_subsection_heading
current_subsection_heading
previous_unit_index
current_unit_index
previous_unit_indexes
current_unit_indexes
previous_hierarchy_path
current_hierarchy_path
```

Ces champs permettent de comprendre pourquoi deux fragments ont été comparés.

Exemple:

```json
{
  "alignment_type": "llm_unit_match",
  "alignment_confidence": 0.91,
  "canonical_topic": "cybersecurite",
  "previous_subsection_heading": "Risques opérationnels",
  "current_subsection_heading": "Risques opérationnels",
  "previous_unit_index": 1,
  "current_unit_index": 2,
  "previous_unit_indexes": [1],
  "current_unit_indexes": [2]
}
```

## Langue des sorties

Les champs visibles par l'analyste doivent être rédigés en français professionnel:

```text
reason
topic
change_summary
explanation
nouvelle_idee_justification
justification_posture
impact_it_justification
```

Les clés JSON, les codes AMF et les valeurs d'énumération restent au format technique imposé:

```text
diff_type = added / removed / modified / unchanged
status = ADDED / REMOVED / MODIFIED / EXISTS
impact_level = MAJEUR / MODERE / MINEUR
```

## Règles de prudence actuelles

Le comportement actuel privilégie la précision plutôt que le rappel.

Règles principales:

- ne jamais matcher par position seule;
- ne pas activer le niveau unité si le plan LLM est incomplet ou ambigu;
- rejeter `split`, `merged` et `moved`;
- ne jamais chercher une unité hors de sa sous-section alignée;
- conserver le fallback sous-section complète quand le plan LLM est fragile.

## Tests de validation

Les tests principaux sont dans `tests/unit/test_text_analysis_pipeline.py`.

Tests utiles:

```text
test_split_body_into_narrative_units_splits_long_paragraph_on_sentence_boundaries
test_split_body_into_narrative_units_preserves_long_bullets_as_units
test_build_comparison_tasks_aligns_narrative_units_by_llm_plan_not_position
test_build_comparison_tasks_supports_local_unit_group_match
test_compare_section_texts_sends_aligned_units_for_long_subsection
test_compare_section_texts_does_not_reclassify_same_section_moved_capital_unit
test_build_comparison_tasks_rejects_gpt_split_restructure
```

Commandes de validation:

```bash
uv run ruff check src/vigilance/text_analysis src/vigilance/text_analysis_pipeline.py tests/unit/test_text_analysis_pipeline.py
uv run pytest tests/unit/test_text_analysis_pipeline.py -q
uv run pytest -q
```

## Limites connues

Le matching d'unités est local à une paire de sous-sections déjà alignées.

Il ne cherche pas automatiquement une unité déplacée vers une autre sous-section si GPT n'a pas explicitement aligné cette restructuration. Cette limite est volontaire pour éviter de reclassifier à tort un vrai ajout ou une vraie suppression.

Une amélioration future possible serait d'ajouter un mode contrôlé de détection de mouvement inter-sous-section, avec seuil élevé et justification explicite.

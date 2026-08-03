# Découpage des fichiers monolithiques — carte de suivi

Document de travail de la branche `refactor/split-monoliths`.
Point de branchement : `e4cce93` (main).

## Objectif

Réduire les fichiers qui concentrent une part importante des 70 718 lignes de `src/vigilance`,
sans changer aucun comportement.

### Priorité 1 terminée — cœur de la comparaison

Le premier lot fonctionnel a été découpé sur la branche courante en quatre commits
indépendants. Les anciens chemins restent des façades compatibles afin de préserver
les imports existants et les points de `monkeypatch` utilisés par les tests.

| Phase | Façade | Avant | Après | Modules de responsabilité | Commit |
|---|---|---:|---:|---|---|
| Rapprochement | `comparison_matching.py` | 1 316 | 45 | `rapprochement_tableaux/contrats.py`, `normalisation_reponses.py`, `correction_reponses.py`, `moteur_rapprochement.py` | `8e38b4d` |
| Différences | `comparison_diff_gpt.py` | 1 171 | 44 | `differences_tableaux/normalisation_elements.py`, `comparaison_deterministe.py`, `comparaison_llm.py`, `filtrage_artefacts.py`, `comparaison_paire.py` | `11d83db` |
| Sections textuelles | `text_analysis/comparison.py` | 1 228 | 98 | `comparaison_sections/modeles.py`, `preparation_lots.py`, `execution_llm.py`, `resolution_alignements.py`, `traitement_fragments_orphelins.py`, `comparaison_section.py` | `10d82fd` |
| Pipeline complet | `compare_gpt.py` | 1 287 | 73 | `pipeline_comparaison/client_openai.py`, `ancrages_visuels.py`, `evenements_tableaux.py`, `traitement_paires.py`, `construction_resultat.py`, `orchestration.py` | `08ad09e` |

Responsabilités obtenues :

- le **rapprochement** valide, normalise, corrige puis apparie les tableaux ;
- les **différences** séparent la comparaison déterministe, l'analyse LLM et le
  filtrage des artefacts ;
- la **comparaison textuelle** sépare les modèles, les lots, l'exécution LLM,
  l'alignement et les fragments orphelins ;
- le **pipeline complet** ne conserve dans l'orchestrateur que l'enchaînement des
  étapes. Le transport OpenAI, les preuves visuelles, les événements non appariés,
  les paires et l'écriture du résultat sont isolés.

Validation du lot : `1 106 passed, 18 skipped`; Ruff et Bandit passent sur tous
les fichiers modifiés, ainsi que dans les hooks de chacun des quatre commits.

| Fichier | Lignes au départ | Cible |
|---|---|---|
| `extraction/section_locator.py` | 4 565 | ~12 modules de 150 à 600 lignes |
| `extraction/vision_full_extractor.py` | 3 569 | ~7 modules |
| `extraction/docling_processor.py` | 3 029 | phase 6 |
| `text_analysis/triage.py` | 2 103 | phase 6 |
| `dash_app/layouts/page_text_analysis.py` | 1 586 | phase 6 |

## Règles

1. Une phase = un commit séparé sur `refactor/split-monoliths`. Les phases restent
   sur cette même branche ; aucune PR distincte n'est requise.
2. Aucun changement de comportement dans un commit de découpage. Un bug trouvé
   en route fait l'objet d'un commit séparé.
3. Les modules d'origine restent en place comme **façades de re-export** : plusieurs
   tests importent des symboles privés (`_PROMPT_BASE`, `_parse_vision_result`,
   `_grade_extraction_quality`, `_viable_indicator_count`, `_structural_indicator_count`,
   `_extract_native_text_indicators`, `_normalize_footnote_marker_id`, `SECTION_PATTERNS`,
   `RISK_SUBSECTIONS`, `LocatedSection`, `TocEntry`, `VisualTextElement`).
   Sans façade, ces tests cassent en bloc et on ne distingue plus une régression
   d'un import mal recâblé.
4. Rien n'est jamais committé sous `outputs/` sur cette branche (237 Mo de binaires
   suivis y produiraient des conflits de merge irrésolubles). Un garde-fou local est
   posé dans `.git/hooks/pre-commit`, non suivi par git.
5. `main` et `dev` ne sont ni modifiées, ni fusionnées ici sans instruction explicite.

## Validation

Critère de sortie de chaque phase : **golden diff nul**, en plus de `pytest` et `ruff`.

Références figées hors dépôt dans `~/vigie-goldens/e4cce93/` (voir son README) :

- `locator/` — sortie de `locate_sections_in_pdf` pour les 36 PDF d'`Inputs/`.
  Déterministe, sans appel LLM, rejouable gratuitement. C'est la référence
  prioritaire pour tout ce qui touche `section_locator.py`.
  Capture initiale : 36/36 en succès, 14,4 min. **Déterminisme vérifié** : deux
  passes indépendantes sur RBC (dont deux rapports annuels T4) produisent une
  sortie identique au caractère près.
- `resultats/` — copie de `outputs/resultats` à `e4cce93`, référence end-to-end.

Observation figée par le golden, à ne pas confondre avec une régression : 16 des
36 PDF ne remontent que 2 sections sur 3 (`regulatory_updates` absent chez TD sur
tous les trimestres, chez CIBC et RBC sur certains). C'est le comportement actuel
— le localisateur a un `_bank_has_regulatory_section()` — pas un défaut introduit
par le découpage.

## État des phases

| Phase | Contenu | État |
|---|---|---|
| 0 | Garde-fous, packaging, goldens | fait |
| 1 | Extractions de données (patterns, modèles, prompts, schéma) | en cours |
| 2 | Fonctions pures (heuristiques qualité, parsing, TDM) | à venir |
| 3 | Découpage par responsabilité de `SectionLocator` | à venir |
| 4 | `locate_sections` en pipeline d'étapes | à venir |
| 5 | `extract()` de `VisionFullExtractor` | à venir |
| 6 | `docling_processor`, `triage`, fichiers Dash | à venir |

## Carte des blocs déplacés

Table de correspondance `source : lignes → destination`, à tenir à jour à chaque phase.
Elle sert au merge final : un correctif fait sur `main` dans un monolithe atterrit
sinon en conflit sur du code qui n'existe plus au même endroit.

### Phase 1 — extractions de données

Déplacements **à l'identique** (extraction par plage de lignes, aucune ligne retapée).
Les modules d'origine restent les façades publiques et re-exportent tout.

`section_locator.py` : 4 565 → 4 044 lignes

| Source (lignes d'origine) | Destination | Contenu |
|---|---|---|
| 26-195 | `extraction/locator/models.py` | `normalize_text`, `VisualTextElement`, `TocEntry`, `LocatedSection`, `SHARED_PAGE_TOP_THRESHOLD`, `SectionMapping` |
| 198-526 | `extraction/locator/patterns.py` | `SECTION_PATTERNS`, `FOLLOWING_SECTION_PATTERNS`, `SECTION_TITLE_ALIASES`, `T4_SECTION_TITLE_PROFILES` |
| 564-616 | `extraction/locator/patterns.py` | `RISK_SUBSECTIONS`, `TOC_PATTERNS` |

`vision_full_extractor.py` : 3 569 → 2 832 lignes

| Source (lignes d'origine) | Destination | Contenu |
|---|---|---|
| 40, 97-588 | `extraction/vision_full/prompts.py` | `_DEFAULT_REFERENCE_TEXT_MAX_CHARS`, les 4 prompts, identifiants de variante |
| 796-897 | `extraction/vision_full/prompts.py` | `_build_prompt`, `_build_precision_prompt` |
| 937-962 | `extraction/vision_full/prompts.py` | `_build_content`, `_build_repair_prompt` |
| 591-722 | `extraction/vision_full/schema.py` | modèles Pydantic, `_build_openai_json_schema`, `_validate_openai_strict_schema_contract` |

**Non déplacés volontairement** :

- `_load_bank_config` et ses dépendants (`_get_bank_section_names`, `BANK_SECTION_NAMES`)
  restent dans `section_locator.py`. La fonction résout la racine du projet par
  `Path(__file__).resolve().parents[3]` : la descendre d'un niveau dans un
  sous-package décalerait la profondeur et casserait le repli de configuration,
  visible seulement quand `vigilance.config.loader` échoue. Reporté en phase 3,
  avec l'ajustement traité explicitement et un test dédié.
- `_extract_native_text_indicators` reste dans `vision_full_extractor.py` : il
  appelle `_is_period_like_indicator`, `_is_weak_indicator` et
  `_looks_narrative_indicator`, qui partent en phase 2. Le déplacer maintenant
  créerait un import circulaire.

**Piège rencontré** : `SECTION_TITLE_ALIASES` et `T4_SECTION_TITLE_PROFILES` sont
des constantes *annotées* (`NOM: type = ...`). Un relevé des constantes par
`grep '^NOM ='` les manque. Pour les phases suivantes, utiliser
`grep -n '^_\?[A-Z_][A-Z_0-9]*\s*[:=]'`. Les tests ont attrapé l'omission
(11 échecs, `NameError`), corrigée par re-export.

### Phase 0 — aucun déplacement de code

| Changement | Nature |
|---|---|
| 8 répertoires vides supprimés (`extraction/vision`, `extraction/components`, `extraction/adapters`, `vigilance/graph`, `vigilance/export`, `quality/checks`, `dash_app/components/detail_widgets`, `dash_app/layouts/text_analysis`) | vestiges non suivis de la branche `refactor/langgraph-multiagent-architecture`, abandonnée |
| `pyproject.toml` : liste explicite → `packages.find` | correction d'un défaut de packaging préexistant (3 packages absents de la wheel) |

## Branche `refactor/langgraph-multiagent-architecture`

Non réutilisée. Vérification faite avant de démarrer : elle ne réduit aucun des deux
plus gros fichiers (`section_locator.py` +1 ligne, `docling_processor.py` inchangé) ;
ses « modules » sont des réimplémentations parallèles neuves, pas des extractions
(`export_comparison_to_excel()` ignore ses arguments et n'exporte rien,
`check_extraction_completeness()` teste si une clé est non vide alors que
`quality_gate.py` fait 1 132 lignes de contrôles réels) ; et ses imports sont morts
(`detect_section_key` et les trois `check_*` sont importés mais jamais appelés).
Elle ajoute par ailleurs `langchain-core`, `langchain-openai` et `langgraph` en
dépendances runtime, et remplace `triage.py` (2 103 lignes) par une façade de
73 lignes — un changement de comportement, pas un découpage.

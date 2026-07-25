# Unités atomiques pour les listes et les énumérations narratives

> **Vigilance bancaire — Pipeline de comparaison textuelle**
>
> Branche : `feat/atomic-comparison-units`
>
> Commit d'implémentation : `a3b0f05`
>
> Dépendance : `feat/text-canonical-cleanup-chunk-repair` — commit `11824c1`, PR #105
>
> Version du document : 1.0 — juillet 2026

---

## Table des matières

1. [Objectif](#1-objectif)
2. [Problème corrigé](#2-problème-corrigé)
3. [Position dans l'architecture](#3-position-dans-larchitecture)
4. [Modèle parent-enfants](#4-modèle-parent-enfants)
5. [Formats pris en charge](#5-formats-pris-en-charge)
6. [Règles de découpage](#6-règles-de-découpage)
7. [Règles d'alignement anti-bruit](#7-règles-dalignement-anti-bruit)
8. [Production des changements](#8-production-des-changements)
9. [Exemple réel BMO — liste Markdown](#9-exemple-réel-bmo--liste-markdown)
10. [Exemple réel TD — énumération romaine](#10-exemple-réel-td--énumération-romaine)
11. [Scénarios d'insertion, déplacement et remplacement](#11-scénarios-dinsertion-déplacement-et-remplacement)
12. [Métadonnées techniques](#12-métadonnées-techniques)
13. [Fichiers modifiés](#13-fichiers-modifiés)
14. [Validation et tests](#14-validation-et-tests)
15. [Limites connues](#15-limites-connues)
16. [Régénération des résultats](#16-régénération-des-résultats)

---

## 1. Objectif

Cette évolution améliore la granularité de comparaison des passages narratifs structurés.

Avant cette modification, une liste complète ou une longue phrase contenant une énumération pouvait former un seul chunk. Plusieurs idées indépendantes étaient alors comparées ensemble et produisaient généralement une seule fiche de changement.

L'objectif est désormais le suivant :

> Une unité comparable doit représenter une idée vérifiable, sans modifier le Markdown canonique et sans utiliser les marqueurs de liste comme preuve sémantique.

La solution introduit une couche d'unités atomiques entre le Markdown canonique et l'alignement TF-IDF/embeddings.

Elle couvre en priorité :

- les listes Markdown commençant par `-`;
- les listes numérotées;
- les énumérations alphabétiques;
- les énumérations romaines intégrées dans une phrase, par exemple `i) ...; ii) ...`;
- la relation entre un paragraphe introductif et les éléments qu'il annonce.

---

## 2. Problème corrigé

### 2.1 Comportement antérieur pour une liste

Une liste telle que :

```text
- Comprendre et gérer les risques.
- Préserver la réputation de la Banque.
- Maintenir une solide position de liquidité.
```

était regroupée en un seul chunk :

```text
Comprendre et gérer les risques.
Préserver la réputation de la Banque.
Maintenir une solide position de liquidité.
```

Conséquences :

- une seule puce modifiée faisait apparaître toute la liste comme modifiée;
- une nouvelle puce pouvait dégrader l'alignement de toute la liste;
- plusieurs changements indépendants étaient regroupés dans une seule fiche;
- l'analyste devait retrouver manuellement la puce réellement concernée.

### 2.2 Comportement antérieur pour une longue énumération

La section TD sur les facteurs pouvant affecter le cours des titres contient une phrase de plus de 400 mots avec quinze facteurs numérotés de `i)` à `xv)`.

Le chunking respectait l'intégrité grammaticale de la phrase. La phrase entière restait donc une unité unique, même si elle contenait quinze idées structurées.

Conséquences :

- les ajouts qualitatifs dans les facteurs `iii)` et `vi)` étaient regroupés;
- les treize facteurs inchangés alourdissaient le texte envoyé au modèle;
- le résultat était moins précis pour le retrieval, l'explication et la preuve.

### 2.3 Limite de la règle des 240 mots

La limite souhaitée de 240 mots ne doit pas provoquer une coupure arbitraire.

La règle demeure :

> Une phrase n'est jamais coupée uniquement pour satisfaire une taille maximale.

La présente évolution ajoute une nuance :

> Une phrase peut être décomposée lorsqu'elle contient des frontières structurelles explicites et vérifiables, comme `i)`, `ii)`, `iii)`.

La coupure se fait entre des éléments autonomes, jamais au milieu de leur contenu.

---

## 3. Position dans l'architecture

La modification est localisée après la canonicalisation et avant l'alignement.

```text
PDF T1 / PDF T2
    ↓
Extraction Docling
    ↓
Nettoyage et Markdown canonique
    ↓
Sections et sous-sections ###
    ↓
Blocs réparés : paragraphes et listes
    ↓
NOUVEAU — Détection des unités atomiques parent-enfants
    ↓
TF-IDF + embeddings
    ↓
Alignements locaux
    ↓
Diff exact ou validation LLM
    ↓
Triage et text_comparison.json
```

### 3.1 Ce qui ne change pas

Cette évolution ne modifie pas :

- l'extraction Docling;
- le contenu du Markdown canonique;
- la détection des sections;
- l'appariement des sous-sections;
- les mécanismes de réconciliation globale;
- le triage réglementaire;
- les formats Excel et Dash existants.

### 3.2 Frontière avec le PR #105

Le PR #105 améliore principalement :

- le nettoyage du Markdown canonique;
- la suppression des en-têtes et pieds de page;
- la réparation des frontières cassées;
- l'audit des transformations;
- la validation Vision des fusions ambiguës.

La présente branche réutilise ce Markdown nettoyé et intervient uniquement dans la couche de comparaison.

---

## 4. Modèle parent-enfants

Une liste ou une énumération est représentée par :

- une unité de contexte parent, lorsqu'une introduction autonome est disponible;
- plusieurs unités enfants;
- une relation explicite de chaque enfant vers son parent.

### 4.1 Exemple conceptuel

```text
Parent
└── Les objectifs de gestion des risques sont les suivants :
    ├── Enfant 1 — Comprendre et gérer les risques
    ├── Enfant 2 — Préserver la réputation
    ├── Enfant 3 — Diversifier les risques
    ├── Enfant 4 — Maintenir le capital et la liquidité
    └── Enfant 5 — Optimiser le rapport risque-rendement
```

### 4.2 Rôle du parent

Le parent :

- fournit le contexte métier de la liste;
- permet de regrouper les changements liés;
- évite de répéter la même introduction dans chaque embedding;
- demeure une unité distincte si son propre contenu doit être comparé.

### 4.3 Rôle des enfants

Chaque enfant :

- contient une seule idée structurée;
- conserve son extrait source;
- possède son propre identifiant de chunk;
- possède son propre alignement;
- peut produire une fiche de changement indépendante.

---

## 5. Formats pris en charge

### 5.1 Puces Markdown

```text
- Première obligation.
- Deuxième obligation.
- Troisième obligation.
```

Chaque ligne devient une unité `list_item`.

Les glyphes de puce issus du PDF, par exemple `•`, `▪`, `‰` ou certains caractères privés Docling, sont normalisés par la couche existante de traitement des listes.

### 5.2 Listes numériques

Les formes suivantes sont reconnues :

```text
1. Première mesure
2. Deuxième mesure
```

et :

```text
1) Première mesure
2) Deuxième mesure
```

### 5.3 Énumérations alphabétiques

```text
a) renforcer les contrôles;
b) revoir la gouvernance;
c) documenter les résultats.
```

### 5.4 Énumérations romaines

```text
i) risque de crédit;
ii) risque technologique;
iii) risque opérationnel.
```

Les formes parenthésées, par exemple `(i)`, `(ii)`, sont également reconnues.

### 5.5 Garde-fou de séquence

Une énumération n'est découpée que si au moins deux marqueurs forment une séquence consécutive valide.

Exemple découpé :

```text
i) première idée; ii) deuxième idée; iii) troisième idée.
```

Exemple non découpé :

```text
Le rapport renvoie à i) une définition et iii) une référence distincte.
```

Dans le second cas, les marqueurs ne sont pas consécutifs. Le passage reste un paragraphe normal afin d'éviter une inférence risquée.

---

## 6. Règles de découpage

### 6.1 Liste Markdown multi-item

Une liste est décomposée si :

- elle contient au moins deux lignes non vides;
- chaque ligne est reconnue comme un élément de liste;
- chaque élément contient du texte narratif.

Si la structure n'est pas entièrement validée, le chunker conserve le comportement historique et ne force pas le découpage.

### 6.2 Paragraphe introductif d'une liste

Le paragraphe immédiatement placé avant une liste devient le contexte parent lorsque :

- il contient au moins cinq mots;
- il se termine par `:`;
- il précède directement la liste.

Exemple :

> Notre approche en gestion des risques s'articule autour de cinq objectifs clés :

Ce passage devient `list_context`. Les cinq puces suivantes référencent son identifiant.

### 6.3 Introduction d'une énumération intégrée

Pour une énumération à l'intérieur d'un paragraphe, le texte placé avant le premier marqueur devient un contexte lorsque :

- il contient au moins cinq mots;
- il se termine par `:`, `.`, `!` ou `?`.

Une amorce grammaticale incomplète n'est pas transformée en paragraphe autonome.

### 6.4 Éléments enfants

Chaque enfant conserve :

- son marqueur source dans le texte de preuve;
- son contenu sans marqueur dans le texte de similarité;
- son ordre local;
- son parent;
- le contexte textuel du parent.

### 6.5 Paragraphes ordinaires

Un paragraphe sans liste ou énumération explicite continue de suivre le chunking existant :

- conservation du paragraphe simple;
- partition sémantique des paragraphes complexes;
- coupure uniquement entre des phrases complètes;
- arbitrage LLM des frontières ambiguës.

---

## 7. Règles d'alignement anti-bruit

### 7.1 Le contenu est prioritaire

Le marqueur `i)`, `ii)`, `1)` ou `-` ne constitue jamais une preuve de correspondance.

L'alignement utilise principalement :

- le contenu sans marqueur;
- la similarité TF-IDF;
- la similarité par embeddings;
- le contexte de sous-section;
- l'ordre documentaire comme signal secondaire.

### 7.2 Le marqueur est seulement un départage

Lorsque deux candidats ont déjà des contenus comparables, un marqueur identique peut servir de départage.

Le marqueur :

- n'augmente pas artificiellement le score de similarité;
- ne permet pas à une paire faible de franchir le seuil d'alignement;
- ne force pas `i) T1 ↔ i) T2`;
- n'est pas utilisé pour les puces génériques `-`.

### 7.3 Neutralisation dans les embeddings et TF-IDF

Deux textes sont distingués :

| Texte | Usage |
| --- | --- |
| Texte source | Preuve, affichage, audit |
| Texte de comparaison | TF-IDF, embeddings, décision exacte |

Exemple :

| Représentation | Valeur |
| --- | --- |
| Source | `iii) l'incidence de la résolution globale...` |
| Comparaison | `l'incidence de la résolution globale...` |

Cette séparation empêche une renumérotation de devenir un faux changement.

### 7.4 Compatibilité des rôles

Une unité de contexte ne peut pas être alignée avec un élément de liste.

Correspondances autorisées :

- contexte ↔ contexte;
- enfant ↔ enfant;
- unité ordinaire ↔ unité ordinaire;
- unité ordinaire ↔ enfant dans les cas de restructuration non ambigus.

Correspondance refusée :

- contexte ↔ enfant.

### 7.5 Faits numériques

La détection des faits numériques divergents utilise également le texte sans marqueur.

Ainsi, une liste renumérotée de `1)` vers `2)` ne devient pas ambiguë uniquement à cause de son numéro structurel.

### 7.6 Diff exact

Pour un alignement fort, le diff exact compare le contenu sans marqueur.

Exemple :

```text
T1 : i) le risque de crédit commercial
T2 : ii) le risque de crédit commercial
```

Résultat :

- contenu : identique;
- position : différente;
- décision : `unchanged`.

---

## 8. Production des changements

Chaque enfant possède son propre `chunk_id` et reçoit son propre `alignment_id`.

Le mécanisme de déduplication conserve une fiche par alignement. Deux enfants modifiés sous le même parent ne sont donc pas fusionnés.

### 8.1 Résultat attendu

```text
Parent : Facteurs pouvant affecter les titres de TD
├── iii) modifié — limite de l'actif aux États-Unis
└── vi) modifié — incapacité à atteindre les cibles financières
```

Les enfants inchangés peuvent être utilisés pendant la comparaison, mais ils ne sont pas conservés parmi les changements soumis au triage final.

### 8.2 Présentation actuelle

Les données contiennent maintenant la hiérarchie parent-enfants.

L'interface Dash actuelle n'affiche pas encore une arborescence visuelle. Elle présente les changements comme des fiches séparées sous la même sous-section.

Une évolution ultérieure de l'interface pourra regrouper les fiches au moyen de `parent_chunk_id_t1` et `parent_chunk_id_t2`.

---

## 9. Exemple réel BMO — liste Markdown

### 9.1 Source

Sous-section :

> Cadre d'appétit pour le risque

Introduction :

> Nous jugeons que la responsabilité de la gestion des risques incombe à chacun de nos employés et notre approche en gestion des risques s'articule autour de cinq objectifs clés, qui orientent toutes nos activités en ce domaine et s'inscrivent dans notre énoncé d'appétit pour le risque:

Liste :

1. Comprendre et gérer en n'assumant que les risques transparents et clairement définis.
2. Préserver la réputation de BMO.
3. Diversifier et restreindre les risques extrêmes.
4. Maintenir une situation solide en matière de capital et de liquidité.
5. Optimiser le rapport risque-rendement.

### 9.2 Unités produites

| Ordre | Type | Rôle | Contenu résumé |
| ---: | --- | --- | --- |
| 0 | `list_context` | contexte | Présentation des cinq objectifs |
| 1 | `list_item` | enfant | Comprendre et gérer les risques |
| 2 | `list_item` | enfant | Préserver la réputation |
| 3 | `list_item` | enfant | Diversifier les risques |
| 4 | `list_item` | enfant | Maintenir capital et liquidité |
| 5 | `list_item` | enfant | Optimiser risque-rendement |

### 9.3 Résultat réel BMO 2024–2025

Dans les Markdown examinés, les cinq objectifs sont identiques entre 2024 et 2025.

Résultat :

- cinq alignements forts enfant ↔ enfant;
- contexte inchangé;
- aucune fiche de changement retenue pour cette liste.

### 9.4 Scénario : une seule puce modifiée

Rapport précédent :

> Maintenir une situation enviable en matière de capital et de liquidité qui respecte les exigences réglementaires.

Rapport courant :

> Maintenir une situation enviable en matière de capital et de liquidité qui respecte les exigences réglementaires et renforce la capacité de la Banque à absorber des périodes de crise.

Résultat :

> BMO ajoute le renforcement de sa capacité à absorber des périodes de crise à son objectif relatif au capital et à la liquidité.

Les quatre autres objectifs restent inchangés et ne produisent pas de fiches.

---

## 10. Exemple réel TD — énumération romaine

### 10.1 Source

Sous-section :

> Valeur et cours de nos actions ordinaires et des autres titres

Le paragraphe contient une introduction et quinze facteurs `i)` à `xv)`.

Le nouveau chunking produit :

- une unité de contexte;
- quinze unités enfants;
- seize unités comparables au total.

### 10.2 Changement du facteur iii

Rapport précédent :

> iii) l'incidence de la résolution globale sur les activités, l'exploitation et la situation financière de la Banque

Rapport courant :

> iii) l'incidence de la résolution globale sur les activités, l'exploitation et la situation financière de la Banque, y compris l'incidence de la limite de l'actif de la Banque aux États-Unis

Résultat :

> TD précise que l'incidence de la résolution globale comprend celle de la limite imposée à l'actif de la Banque aux États-Unis.

### 10.3 Changement du facteur vi

Rapport précédent :

> vi) la différence entre les résultats réels de la Banque et ceux auxquels s'attendent les investisseurs et les analystes

Rapport courant :

> vi) la différence entre les résultats réels de la Banque et ceux auxquels s'attendent les investisseurs et les analystes, y compris l'incapacité à atteindre les cibles financières

Résultat :

> TD ajoute l'incapacité à atteindre les cibles financières parmi les facteurs pouvant créer un écart par rapport aux attentes des investisseurs et des analystes.

### 10.4 Granularité obtenue

| Unité | Décision |
| --- | --- |
| Contexte | Inchangé |
| `i)` | Inchangé |
| `ii)` | Inchangé |
| `iii)` | Modifié |
| `iv)`–`v)` | Inchangés |
| `vi)` | Modifié |
| `vii)`–`xv)` | Inchangés |

Le résultat contient deux changements indépendants au lieu d'une modification globale du paragraphe de plus de 400 mots.

---

## 11. Scénarios d'insertion, déplacement et remplacement

### 11.1 Insertion avec renumérotation

Rapport précédent :

```text
i) risque de crédit
ii) risque technologique
iii) risque opérationnel
```

Rapport courant :

```text
i) nouveau risque climatique
ii) risque de crédit
iii) risque technologique
iv) risque opérationnel
```

Alignement obtenu :

| Rapport précédent | Rapport courant | Décision |
| --- | --- | --- |
| — | `i)` risque climatique | Ajouté |
| `i)` risque de crédit | `ii)` risque de crédit | Inchangé |
| `ii)` risque technologique | `iii)` risque technologique | Inchangé |
| `iii)` risque opérationnel | `iv)` risque opérationnel | Inchangé |

Il n'y a pas de cascade de fausses modifications.

### 11.2 Réorganisation de puces

Rapport précédent :

```text
- Préserver la réputation.
- Maintenir la liquidité.
```

Rapport courant :

```text
- Maintenir la liquidité.
- Préserver la réputation.
```

Résultat :

- réputation ↔ réputation : inchangé;
- liquidité ↔ liquidité : inchangé;
- aucun changement qualitatif retenu.

### 11.3 Même marqueur, contenu remplacé

Rapport précédent :

> i) le risque de crédit commercial

Rapport courant :

> i) les attaques de cybersécurité externes

Le marqueur identique ne force pas la paire.

Si aucun équivalent n'existe ailleurs :

- risque de crédit : supprimé;
- risque cyber : ajouté.

### 11.4 Modification réelle du même élément

Rapport précédent :

> iii) incidence de la résolution globale sur la situation financière

Rapport courant :

> iii) incidence de la résolution globale sur la situation financière, y compris la limite de l'actif aux États-Unis

Le sujet principal est conservé et une précision est ajoutée.

Décision :

- même divulgation;
- `modified`;
- une fiche propre à l'élément `iii)`.

---

## 12. Métadonnées techniques

### 12.1 Métadonnées ajoutées à `TextChunk`

| Champ | Type logique | Rôle |
| --- | --- | --- |
| `comparison_text` | texte | Contenu utilisé pour TF-IDF, embeddings et diff exact |
| `unit_role` | `standalone`, `context`, `item`, `grouped` | Rôle hiérarchique |
| `parent_chunk_id` | identifiant optionnel | Parent de l'unité enfant |
| `atomic_marker` | texte optionnel | Marqueur source, par exemple `iii)` ou `-` |
| `parent_context` | texte | Contexte narratif du parent |

### 12.2 Métadonnées ajoutées aux changements

Les champs sont exposés séparément pour le rapport précédent et le rapport courant :

| Champ | Description |
| --- | --- |
| `unit_role_t1`, `unit_role_t2` | Rôle de l'unité |
| `parent_chunk_id_t1`, `parent_chunk_id_t2` | Identifiant du parent |
| `atomic_marker_t1`, `atomic_marker_t2` | Marqueur structurel |
| `parent_context_t1`, `parent_context_t2` | Contexte du parent |

Ces champs sont additionnels et ne cassent pas les consommateurs existants.

---

## 13. Fichiers modifiés

### 13.1 Nouveaux modules

| Fichier | Responsabilité |
| --- | --- |
| `src/vigilance/text_analysis/atomic_units.py` | Détection et décomposition des listes et énumérations |
| `src/vigilance/text_analysis/atomic_alignment.py` | Texte de similarité, compatibilité des rôles et départage par marqueur |

### 13.2 Modules adaptés

| Fichier | Modification |
| --- | --- |
| `src/vigilance/text_analysis/chunking.py` | Création des parents, enfants et métadonnées |
| `src/vigilance/text_analysis/chunk_alignment.py` | Similarité sans marqueur et alignement anti-bruit |
| `src/vigilance/text_analysis/comparison.py` | Diff exact sans marqueur et propagation des métadonnées |

### 13.3 Tests

| Fichier | Couverture |
| --- | --- |
| `tests/unit/test_atomic_comparison_units.py` | Scénarios atomiques BMO, TD, insertion, déplacement et remplacement |
| `tests/unit/test_semantic_chunking.py` | Nouveau comportement des puces Docling |
| `tests/unit/test_text_analysis_pipeline.py` | Intégration avec Markdown canonique et listes existantes |

---

## 14. Validation et tests

### 14.1 Résultat global

- 1 053 tests réussis;
- 17 tests ignorés;
- 0 échec;
- contrôles Ruff réussis;
- analyse Bandit réussie;
- contrôles pré-commit réussis.

### 14.2 Scénarios couverts

- liste Markdown BMO avec un parent et cinq enfants;
- énumération TD de quinze facteurs;
- listes romaines, numériques et alphabétiques;
- marqueurs non consécutifs;
- insertion d'un élément et renumérotation complète;
- réorganisation des puces;
- même marqueur avec contenu distinct;
- neutralisation des marqueurs dans les embeddings;
- décision `unchanged` pour une renumérotation sans changement de contenu;
- conservation de deux changements distincts sous un même parent;
- propagation des métadonnées parent-enfants.

---

## 15. Limites connues

### 15.1 Prose sans marqueurs

Un paragraphe qui contient plusieurs idées sans puces, numéros ou séparateurs explicites continue d'utiliser le chunking sémantique existant.

Cette branche ne cherche pas à découper arbitrairement toute la prose en idées abstraites.

### 15.2 Structures invalides ou incomplètes

Si les marqueurs ne forment pas une séquence valide, le système ne découpe pas.

Cette décision est conservatrice afin d'éviter des faux blocs.

### 15.3 Listes complexes

Les listes imbriquées, les éléments répartis sur plusieurs blocs Docling ou les listes dont les lignes ne sont pas toutes reconnues peuvent conserver le comportement historique.

Le système préfère un bloc plus large à une séparation non vérifiable.

### 15.4 Alignement généralisé `1 ↔ N`

Cette évolution couvre les insertions, renumérotations et déplacements grâce à l'alignement par contenu.

Elle ne remplace pas encore l'alignement généralisé `1 ↔ N` pour tous les cas de restructuration libre de prose.

Le mécanisme existant de réassemblage de chunks adjacents reste disponible pour certains cas presque identiques.

### 15.5 Interface

Les métadonnées hiérarchiques sont disponibles, mais Dash n'affiche pas encore explicitement un arbre parent-enfants.

### 15.6 Changements quantitatifs

Le découpage atomique ne décide pas de la pertinence métier.

Les changements exclusivement quantitatifs continuent d'être traités par la politique de triage aval. Ils ne doivent pas influencer les frontières des unités.

---

## 16. Régénération des résultats

### 16.1 Artefacts existants

Les fichiers `text_comparison.json` déjà présents ne sont pas automatiquement réécrits par cette modification.

Ils conservent les anciens résultats jusqu'à une nouvelle exécution de la comparaison.

### 16.2 Nouvelle exécution

Lors d'une nouvelle comparaison :

1. le Markdown canonique est chargé;
2. les sous-sections sont reconstruites;
3. les listes et énumérations deviennent des unités atomiques;
4. les unités sont alignées;
5. les changements sont régénérés;
6. `text_comparison.json` et l'export Excel sont réécrits.

### 16.3 Réextraction

Le découpage atomique agit sur le Markdown. Une réextraction PDF n'est pas requise uniquement pour activer cette logique si le Markdown canonique est déjà valide.

Une réextraction reste recommandée lorsque :

- le Markdown utilise une ancienne version de cache;
- les correctifs de canonicalisation du PR #105 doivent être appliqués;
- les fichiers contiennent encore des en-têtes, pieds de page ou frontières cassées.

### 16.4 Ordre d'intégration recommandé

1. fusionner le PR #105;
2. rebaser ou actualiser `feat/atomic-comparison-units` si nécessaire;
3. ouvrir le PR des unités atomiques;
4. exécuter la comparaison sur un échantillon BMO et TD;
5. vérifier les changements retenus dans JSON, Excel et Dash;
6. étendre ensuite la validation aux autres banques.

---

## Résumé

Cette évolution transforme les listes et énumérations explicites en unités comparables de première classe.

Le Markdown canonique demeure intact. Le marqueur reste disponible pour la preuve, mais il est neutralisé dans la similarité. Le contenu décide de l'alignement, le marqueur ne sert que de départage prudent.

Le résultat attendu est :

- moins de faux positifs lors des insertions et réorganisations;
- une fiche par idée qualitative modifiée;
- une meilleure preuve pour l'analyste;
- une hiérarchie parent-enfants exploitable par les futures évolutions de l'interface.

# Architecture technique du système de vigilance bancaire

## Métadonnées du document

| Champ | Valeur |
| --- | --- |
| Titre | Architecture cible du pipeline de vigilance bancaire |
| Périmètre | Extraction de tableaux réglementaires depuis des PDF, comparaison inter-trimestres, stockage d'artefacts JSON, revue analyste dans Dash |
| Positionnement | Document d'architecture cible, ancré sur l'implémentation observée dans le dépôt |
| Date | Mars 2026 |

---

## Résumé exécutif

Le système doit être conçu comme une chaîne batch de production d'artefacts, et non comme une application interactive qui calcule en direct pendant la revue. En pratique, la valeur métier n'est pas dans Dash lui-même. Elle est dans la fabrication, l'historisation et l'auditabilité des fichiers produits en amont : `tables.json`, `indicators.json`, `footnotes.json`, `comparison.json` et `manifest.json`.

Du point de vue d'un analyste bancaire, c'est la bonne architecture. Une analyse réglementaire doit être rejouable, traçable, stable et revue a posteriori. Si l'interface recalcule à la volée, on introduit un risque de non-reproductibilité : deux analystes peuvent ne pas revoir exactement le même résultat sur la même paire de rapports. Ce mode de fonctionnement est faible en gouvernance et faible en audit.

L'architecture cible est donc la suivante :

1. Les rapports PDF sont déposés dans `Inputs/`.
2. Le backend batch génère tous les artefacts JSON de manière déterministe dans leur emplacement de run.
3. Dash ne fait que lire ces artefacts persistés pour la revue.
4. La seule écriture autorisée côté Dash est l'état de revue analyste, stocké en sidecar à côté du `comparison.json`.

---

## Principe directeur

Le principe de conception doit être formulé sans ambiguïté :

**Le système de production fabrique des preuves. Dash ne fabrique pas les preuves, il les consulte.**

Conséquences directes :

- Le moteur d'extraction et le moteur de comparaison sont des composants backend hors UI.
- `tables.json` est la source canonique d'un trimestre.
- `indicators.json` et `footnotes.json` sont des projections dérivées, utiles pour audit et inspection, mais non canoniques.
- `comparison.json` est le livrable analytique officiel d'un run.
- Le dossier de run est l'unité de traçabilité métier.
- Dash est un consommateur de `comparison.json`, pas un producteur de calculs métier.

---

## Architecture cible

```{mermaid}
flowchart LR
  subgraph inputs [Entrées métier]
    PDFprev[PDF trimestre précédent]
    PDFcur[PDF trimestre courant]
  end

  subgraph batch [Plan de production hors Dash]
    ORCH[run_pipeline.py ou CLI dédiées]
    LOC[Localisation des sections]
    EXT[Extraction documentaire + Vision]
    STOREX[Stockage extraction par trimestre]
    CMP[Matching et diff sémantique]
    STORERUN[Écriture du run de comparaison]
  end

  subgraph artifacts [Artefacts persistés]
    TPREV[tables.json précédent]
    TCUR[tables.json courant]
    IPREV[indicators.json précédent]
    ICUR[indicators.json courant]
    FPREV[footnotes.json précédent]
    FCUR[footnotes.json courant]
    COMP[comparison.json]
    MANI[manifest.json]
    REVIEW[comparison.review_state.json]
  end

  subgraph ui [Dash]
    DASH[Lecture, tri, justification, revue]
  end

  PDFprev --> ORCH
  PDFcur --> ORCH
  ORCH --> LOC
  LOC --> EXT
  EXT --> STOREX
  STOREX --> TPREV
  STOREX --> TCUR
  STOREX --> IPREV
  STOREX --> ICUR
  STOREX --> FPREV
  STOREX --> FCUR
  TPREV --> CMP
  TCUR --> CMP
  CMP --> STORERUN
  STORERUN --> COMP
  STORERUN --> MANI
  COMP --> DASH
  MANI --> DASH
  DASH --> REVIEW
```

Ce schéma sépare volontairement :

- le plan de calcul ;
- le plan de stockage ;
- le plan de consultation.

Cette séparation est la bonne réponse à un besoin de gouvernance bancaire, de contrôlabilité et de réduction des ambiguïtés opérationnelles.

---

## Couches du système

### 1. Couche d'entrée documentaire

Les sources sont les PDF trimestriels déposés sous `Inputs/{BANK}/{YEAR}/`. La logique métier de référence temporelle est résolue en amont : `T2 -> T1`, `T3 -> T2`, `T1 -> T3 de l'année précédente`, `T4 -> T4 de l'année précédente`.

Cette couche ne contient aucune intelligence d'analyse. Elle garantit seulement que le couple documentaire est correct.

### 2. Couche de production batch

Le point d'entrée naturel est `run_pipeline.py`. Il orchestre :

1. la résolution du couple de trimestres ;
2. la localisation des PDF ;
3. l'extraction des deux rapports ;
4. la génération des artefacts trimestriels ;
5. la comparaison entre les deux extractions ;
6. l'écriture du dossier de run final.

Cette couche doit porter l'ensemble des appels LLM, des règles de matching, de la logique de diff, des métriques d'usage et de la persistance des résultats.

### 3. Couche d'artefacts canoniques

Cette couche est le coeur du système. Chaque période produit un jeu d'artefacts d'extraction ; chaque paire de périodes produit un artefact de comparaison.

Le point fondamental est le suivant : **tout composant aval doit consommer des fichiers, jamais des objets en mémoire issus d'une session Dash non persistée**.

### 4. Couche de revue analyste

Dash doit être traité comme une couche de consultation métier :

- chargement d'un `comparison.json` existant ;
- normalisation UI du payload si nécessaire ;
- affichage des changements et des preuves ;
- écriture éventuelle d'un fichier de revue sidecar.

Autrement dit, Dash peut enrichir l'expérience de revue, mais il ne doit pas recalculer la vérité métier.

---

## Contrats de données

| Artefact | Producteur | Nature | Rôle |
| --- | --- | --- | --- |
| `tables.json` | extraction batch | canonique | vérité d'extraction d'un trimestre |
| `indicators.json` | projection depuis `tables.json` | dérivé | audit rapide des libellés de lignes |
| `footnotes.json` | projection depuis `tables.json` | dérivé | audit rapide des notes de bas de page |
| `comparison.json` | moteur de comparaison batch | officiel | vérité de comparaison pour une paire de trimestres |
| `manifest.json` | orchestrateur batch | administratif | traçabilité du run et statut de génération |
| `comparison.review_state.json` | Dash | sidecar UI | état de revue analyste, sans impact sur la vérité métier |

### Hiérarchie de vérité

La hiérarchie doit être explicite :

1. `tables.json` est la vérité source de l'extraction.
2. `comparison.json` est la vérité source de la comparaison.
3. `indicators.json` et `footnotes.json` ne sont que des vues dérivées.
4. `comparison.review_state.json` n'est qu'un état utilisateur.

Cette hiérarchie est essentielle pour éviter qu'une UI, un export secondaire ou un cache local ne se mette à concurrencer la donnée officielle.

---

## Flux de bout en bout

### Étape 1. Constitution de la paire documentaire

Le système identifie le PDF courant et le PDF précédent selon la règle métier de référence. Cette étape relève de l'orchestration, pas de Dash.

### Étape 2. Extraction trimestrielle

Chaque PDF est traité isolément :

- localisation des sections utiles ;
- extraction des tableaux sur le périmètre utile ;
- sérialisation dans un répertoire trimestriel ;
- génération de `tables.json` ;
- dérivation de `indicators.json` et `footnotes.json`.

À ce stade, chaque trimestre devient un objet autonome, stable et rechargeable sans relire le PDF.

### Étape 3. Comparaison inter-trimestres

Le moteur de comparaison ne doit pas relire les PDF. Il doit charger uniquement les deux `tables.json` canoniques, puis exécuter :

- la préparation des cartes de matching ;
- le matching des tableaux ;
- le diff sémantique paire par paire ;
- l'agrégation des tableaux ajoutés, retirés et modifiés ;
- l'écriture de `comparison.json`.

Cela garantit que la comparaison se fait à partir d'une base figée et auditée.

### Étape 4. Revue dans Dash

Dash charge le `comparison.json` déjà présent sur disque, l'adapte si besoin au format canonical UI, puis présente :

- les tableaux appariés ;
- les tableaux ajoutés ou retirés ;
- les changements d'indicateurs ;
- les changements de footnotes ;
- les preuves et le contexte de revue.

La seule persistance légitime côté Dash est un sidecar de revue analyste.

---

## Lecture experte du besoin métier

Pour un usage de vigie bancaire, la séparation batch / revue n'est pas un détail technique. C'est une exigence de qualité analytique.

### Pourquoi Dash ne doit pas recalculer

Si Dash lance lui-même extraction et comparaison, on crée quatre faiblesses :

1. **Non-reproductibilité** : un même dossier peut être recalculé à des moments différents avec un état de code, de prompt ou de modèle différent.
2. **Confusion de responsabilité** : l'outil de revue devient en même temps un outil de production.
3. **Traçabilité affaiblie** : le point officiel de production n'est plus unique.
4. **Risque de modèle** : l'utilisateur peut croire qu'il lit un résultat stable alors qu'il visualise en fait une exécution opportuniste.

Dans un contexte de lecture de rapports réglementaires, c'est une mauvaise architecture de contrôle.

### Pourquoi les JSON doivent être le contrat central

Le bon niveau de vérité n'est ni la session Python, ni l'état navigateur, ni les structures Dash. Le bon niveau de vérité est le fichier persisté. C'est lui qui supporte :

- l'audit ;
- la preuve ;
- le rechargement ;
- la comparaison de runs ;
- la reprise après incident ;
- l'industrialisation nocturne.

---

## État observé dans le dépôt

À la lecture du code actuel, on observe deux modes de fonctionnement coexistants.

### Mode conforme à l'architecture cible

Le dépôt possède déjà les briques correctes pour un modèle batch + lecture seule :

- `run_pipeline.py` orchestre l'extraction, la comparaison et la génération du run.
- `src/vigilance/cli/run_compare_gpt4o.py` consomme des artefacts déjà extraits.
- `src/app/ui_io.py` sait lister et charger des `comparison.json` déjà présents dans `outputs/comparisons/`.
- `src/app/review_storage.py` persiste un état de revue sidecar à côté du fichier de comparaison.

### Mode hérité ou de développement

Le code Dash contient encore une capacité de lancer extraction et comparaison depuis l'interface, via `run_comparison_with_sections`.

Cette capacité peut être utile en développement, en démonstration ou en diagnostic. En revanche, elle ne doit pas être considérée comme le mode d'exploitation cible si l'objectif est bien :

**générer tous les JSON en amont, puis laisser Dash lire uniquement les fichiers enregistrés.**

---

## Architecture recommandée

### Décision 1. Le batch devient le seul producteur officiel des artefacts métier

Tous les appels d'extraction et de comparaison doivent partir de `run_pipeline.py` ou de CLIs backend équivalentes.

### Décision 2. Dash devient officiellement read-only sur la donnée métier

Dash peut charger, filtrer, annoter, exporter et sauvegarder un état de revue. Dash ne doit pas produire `tables.json`, `indicators.json`, `footnotes.json` ou `comparison.json` en mode production.

### Décision 3. Les runs deviennent immuables

Un run doit être interprétable comme une photographie complète d'une analyse à date, avec son `comparison.json`, ses artefacts trimestriels et son `manifest.json`.

### Décision 4. Les projections restent des dérivés, jamais des sources

Tout recalcul d'indicateurs ou de footnotes doit repartir de `tables.json`, pas d'une version enrichie côté UI.

### Décision 5. L'UX doit refléter cette séparation

Dans l'interface cible, le premier geste utilisateur n'est pas d'uploader deux PDF pour lancer un traitement. Le premier geste est de choisir un run déjà produit et de l'examiner.

---

## Recommandations de mise en oeuvre

1. Conserver `run_pipeline.py` comme point d'entrée principal de production.
2. Considérer les capacités de calcul dans Dash comme un mode debug et les masquer en production.
3. Renforcer le dossier de run comme unité d'audit complète.
4. Afficher dans Dash les métadonnées de run, de modèle et d'horodatage pour chaque `comparison.json` chargé.
5. Interdire toute ambiguïté sur la source affichée : la vue doit toujours mentionner le chemin ou l'identifiant du run lu.

---

## Conclusion

L'architecture correcte pour ce système n'est pas une architecture Dash centric. C'est une architecture backend centric, artifact first, review second.

Le backend génère la matière analytique. Le stockage la fige. Dash la rend exploitable pour l'analyste. C'est cette séparation qui donne au système sa crédibilité opérationnelle, sa robustesse de contrôle et sa valeur dans un contexte de vigie bancaire.

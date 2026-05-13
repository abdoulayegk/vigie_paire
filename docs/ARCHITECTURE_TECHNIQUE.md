# Architecture technique de Vigie de paire

## Métadonnées du document

| Champ | Valeur |
| --- | --- |
| Titre | Architecture cible de Vigie de paire |
| Périmètre | Extraction et comparaison inter-trimestres de **rapports bancaires** sur deux axes complémentaires : **tableaux réglementaires** et **texte narratif aligné AMF**. Stockage d'artefacts JSON et markdown. Revue analyste dans Dash. |
| Positionnement | Document d'architecture cible, ancré sur l'implémentation observée dans le dépôt |
| Date | Mars 2026 |

---

## Résumé exécutif

Le système est conçu comme une chaîne de traitement par lot qui produit des preuves, et non comme une application interactive qui calcule en direct pendant la revue. La valeur métier n'est pas dans l'interface : elle est dans la fabrication, l'historisation et l'auditabilité des fichiers produits en amont.

Deux pipelines coexistent, indépendants et complémentaires :

- **Pipeline 1 — Tableaux réglementaires.** Lit les rapports PDF, extrait les tableaux d'indicateurs (capital, liquidité, RWA, etc.), apparie les tableaux T1↔T2 et calcule les écarts. Livrables canoniques : `tables.json` (un par trimestre) et `comparison.json` (un par paire). Point d'entrée : `run_pipeline.py`.
- **Pipeline 2 — Texte narratif aligné AMF.** Lit les mêmes PDF, isole les blocs narratifs des sections en scope, écrit un markdown auditable (`text_extraction_{q}.md`) qui sert à la fois de preuve humaine et d'entrée GPT, puis compare T1 vs T2 et trie les changements selon la taxonomie AMF. Livrable canonique : `text_comparison.json`. Point d'entrée : `run_text_pipeline.py`.

Du point de vue d'un analyste bancaire, c'est la bonne architecture. Une analyse réglementaire doit être rejouable, traçable, stable et revue a posteriori. Si l'interface recalculait à la volée, on introduirait un risque de non-reproductibilité : deux analystes pourraient ne pas revoir exactement le même résultat sur la même paire de rapports. Ce mode de fonctionnement est faible en gouvernance et faible en audit.

L'architecture cible se résume ainsi :

1. Les rapports PDF sont déposés dans `Inputs/`.
2. Les deux moteurs de traitement génèrent leurs livrables JSON et markdown de manière déterministe dans leur dossier d'exécution.
3. Dash ne fait que lire ces artefacts persistés pour la revue.
4. La seule écriture autorisée côté Dash est l'état de revue analyste, stocké dans des fichiers d'accompagnement à côté des artefacts canoniques.

---

## Principe directeur

Le principe de conception doit être formulé sans ambiguïté :

**Le système de production fabrique des preuves. Dash ne fabrique pas les preuves, il les consulte.**

Conséquences directes :

- Le moteur d'extraction et le moteur de comparaison sont des composants applicatifs hors interface.
- `tables.json` est la source canonique d'un trimestre.
- `indicators.json` et `footnotes.json` sont des projections dérivées, utiles pour audit et inspection, mais non canoniques.
- `comparison.json` est le livrable analytique officiel d'une exécution.
- Le dossier d'exécution est l'unité de traçabilité métier.
- Dash est un consommateur de `comparison.json`, pas un producteur de calculs métier.

---

## Architecture cible — vue d'ensemble

Le schéma suivant donne la vue macro. Les deux pipelines partagent la même résolution de paire et la même couche Dash, mais ils sont **strictement indépendants** : on peut rejouer l'un sans l'autre.

```{mermaid}
flowchart LR
  subgraph inputs [Entrées métier]
    PDFprev[PDF trimestre précédent T1]
    PDFcur[PDF trimestre courant T2]
  end

  subgraph quarter [Résolution de paire]
    PAIR[quarter_logic<br/>T2→T1, T1→T3 N-1, T4→T4 N-1]
  end

  subgraph p1 [Pipeline 1 — Tableaux réglementaires]
    direction TB
    P1ENTRY[run_pipeline.py]
    P1EXT[Extraction tableaux<br/>Docling + GPT-4o Vision]
    P1CMP[Appariement + écart sémantique<br/>GPT-4o]
    P1ART[(tables.json × 2<br/>comparison.json<br/>manifest.json)]
    P1ENTRY --> P1EXT --> P1ART --> P1CMP --> P1ART
  end

  subgraph p2 [Pipeline 2 — Texte narratif AMF]
    direction TB
    P2ENTRY[run_text_pipeline.py]
    P2EXT[Extraction blocs narratifs<br/>Docling + classification heuristique]
    P2MD[(text_extraction_q.md × 2<br/>source de vérité auditée)]
    P2CMP[Comparaison sections<br/>GPT-4o sur .md]
    P2TRI[Triage AMF<br/>20 thèmes en scope]
    P2ART[(text_comparison.json)]
    P2ENTRY --> P2EXT --> P2MD --> P2CMP --> P2TRI --> P2ART
  end

  subgraph ui [Couche de revue]
    DASH[Dash — lecture seule sur artefacts]
    REVIEW[(comparison.review_state.json<br/>text_review_state.json)]
  end

  PDFprev --> PAIR
  PDFcur --> PAIR
  PAIR --> P1ENTRY
  PAIR --> P2ENTRY
  P1ART --> DASH
  P2ART --> DASH
  DASH --> REVIEW
```

Ce schéma sépare volontairement :

- le plan de calcul (deux pipelines indépendants) ;
- le plan de stockage (artefacts canoniques + dérivés) ;
- le plan de consultation (Dash lecture seule + état de revue).

Cette séparation est la bonne réponse à un besoin de gouvernance bancaire, de contrôlabilité et de réduction des ambiguïtés opérationnelles.

---

## Pipeline 1 — Tableaux réglementaires

**Objectif métier.** Détecter les changements quantitatifs et structurels dans les tableaux d'indicateurs réglementaires entre deux trimestres : capital, liquidité, fonds propres, ratios, ainsi que l'apparition / disparition de tableaux ou de notes de bas de page.

**Point d'entrée.** [`run_pipeline.py`](https://github.com/abdoulayegk/vigie_paire/blob/main/run_pipeline.py).

**Module principal.** {mod}`vigilance.comparison_runner` orchestre l'extraction puis la comparaison. {mod}`vigilance.compare_gpt` porte l'écart sémantique GPT. {mod}`vigilance.comparison_excel` produit l'export analyste.

```{mermaid}
flowchart LR
  subgraph in1 [Entrée]
    PDF1A[PDF T1]
    PDF1B[PDF T2]
  end

  subgraph extr1 [Extraction par trimestre]
    direction TB
    LOC1[Localisation sections utiles]
    DOC1[Docling — layout pages]
    VIS1[GPT-4o Vision — contenu tableaux]
    NORM1[Normalisation indicateurs<br/>utils/indicator_cleaner]
    STORE1[(tables.json<br/>indicators.json<br/>footnotes.json)]
    LOC1 --> DOC1 --> VIS1 --> NORM1 --> STORE1
  end

  subgraph cmp1 [Comparaison de paire]
    direction TB
    MATCH[comparison_matching<br/>appariement tableaux T1↔T2]
    DIFF[compare_gpt<br/>écart sémantique]
    NOISE[comparison_noise_filter<br/>retrait des changements non substantiels]
    METR[comparison_metrics<br/>scores + métadonnées]
    CANON[comparison_canonical<br/>schéma de sortie]
  end

  subgraph out1 [Livrables canoniques]
    COMP1[(comparison.json)]
    MANI1[(manifest.json)]
    XLSX1[(comparison.xlsx<br/>via comparison_excel)]
  end

  PDF1A --> LOC1
  PDF1B --> LOC1
  STORE1 --> MATCH
  MATCH --> DIFF --> NOISE --> METR --> CANON --> COMP1
  CANON --> MANI1
  COMP1 --> XLSX1
```

**Contrats de sortie (Pipeline 1).**

| Artefact | Producteur | Localisation | Rôle |
| --- | --- | --- | --- |
| `tables.json` | extraction par lot | `outputs/extractions/{bank}/{year}/{q}/` | vérité d'extraction d'un trimestre |
| `indicators.json` | projection | idem | audit rapide des libellés de lignes |
| `footnotes.json` | projection | idem | audit rapide des notes de bas de page |
| `comparison.json` | moteur de comparaison | `outputs/resultats/{bank}/{q2}_vs_{q1}/` | vérité de comparaison de la paire |
| `manifest.json` | orchestrateur | idem | traçabilité de l'exécution |
| `comparison.xlsx` | export analyste | idem | export Excel minimal pour revue |

---

## Pipeline 2 — Texte narratif aligné AMF

**Objectif métier.** Détecter les changements de **divulgation textuelle** entre deux trimestres, dans les sections en scope AMF (20 thèmes : divulgations ajoutées / retirées, modifications de méthodologie, capital, liquidité, ESG climatique, risque émergent, etc.). Les exclusions sont strictes : variations chiffrées propres à la banque, RWA, stress tests, comparatifs inter-banques, signaux purement numériques.

**Point d'entrée.** [`run_text_pipeline.py`](https://github.com/abdoulayegk/vigie_paire/blob/main/run_text_pipeline.py).

**Modules principaux.** {mod}`vigilance.text_analysis_pipeline` orchestre l'ensemble. {mod}`vigilance.amf_taxonomy` verrouille les 20 thèmes et le contrat Pydantic du triage GPT. {mod}`vigilance.text_extraction` écrit le markdown auditable. {mod}`vigilance.text_comparison` sérialise le livrable et produit l'export Excel.

```{mermaid}
flowchart LR
  subgraph in2 [Entrée]
    PDF2A[PDF T1]
    PDF2B[PDF T2]
  end

  subgraph extr2 [Extraction texte par trimestre]
    direction TB
    RES2[Résolution sections AMF en scope]
    DOC2[Docling — blocs page]
    CLA2[Classification heuristique<br/>narratif / tableau / footnote]
    MD2[(text_extraction_q.md<br/>source de vérité auditée)]
    AUD2[(text_extraction_audit.json<br/>traçabilité par bloc)]
    RES2 --> DOC2 --> CLA2 --> MD2
    CLA2 --> AUD2
  end

  subgraph cmp2 [Comparaison de paire]
    direction TB
    PARSE[Parse sections + sous-sections<br/>depuis le .md]
    GPT[GPT-4o — comparaison T1 vs T2<br/>par section]
    PAGE[Mapping fragments → pages]
    TRI[Triage AMF — 20 thèmes<br/>genai_triage + amf_taxonomy]
    FILT[Filtre nouvelle_idee<br/>substantielle + nouveauté + thème AMF]
  end

  subgraph out2 [Livrables canoniques]
    TXTC[(text_comparison.json)]
    XLSX2[(text_comparison.xlsx<br/>via text_comparison_excel)]
  end

  PDF2A --> RES2
  PDF2B --> RES2
  MD2 --> PARSE
  PARSE --> GPT --> PAGE --> TRI --> FILT --> TXTC
  TXTC --> XLSX2
```

**Pourquoi le `.md` est la source de vérité.** Le pipeline texte fait un choix d'architecture fort : le markdown extrait est à la fois la **preuve auditable par l'humain** ET **l'entrée directe** de GPT-4o pour la comparaison. Cette unicité garantit qu'un analyste qui ouvre le `.md` voit exactement ce que le modèle a lu. Supprimer le `.md` force une ré-extraction Docling au prochain run — c'est volontaire : le `.md` est l'unité de cache et l'unité d'audit.

**Contrats de sortie (Pipeline 2).**

| Artefact | Producteur | Localisation | Rôle |
| --- | --- | --- | --- |
| `text_extraction_{q}.md` | extraction texte par lot | `outputs/text_extractions/{bank}/{year}/{q}/` | source de vérité narrative auditée |
| `text_extraction_audit.json` | extraction texte par lot | idem | traçabilité bloc-à-bloc (inclus / exclu / raison) |
| `text_comparison.json` | comparaison texte par lot | `outputs/resultats/{bank}/{q2}_vs_{q1}/` | vérité de comparaison textuelle de la paire |
| `text_comparison.xlsx` | export analyste | idem | export Excel minimal (Nouvelle idée + Justification + commentaire) |
| `text_review_state.json` | Dash | idem | état de revue analyste, accompagne `text_comparison.json` |

**Définition unifiée d'une « nouvelle idée ».** Un changement est retenu (côté tableaux comme côté texte) si et seulement si les trois conditions sont réunies :

1. **Substantielle** — ce n'est pas une variation chiffrée propre à la banque ni un ajustement de forme ;
2. **Nouveauté informationnelle** — l'information n'était pas déjà présente au trimestre précédent ;
3. **Thème AMF** — le changement entre dans l'un des 20 thèmes en scope.

Cette définition est partagée entre les deux pipelines, ce qui garantit que la revue analyste manipule une notion unique quel que soit l'axe.

---

## Couches du système

### 1. Couche d'entrée documentaire

Les sources sont les PDF trimestriels déposés sous `Inputs/{BANK}/{YEAR}/`. La logique métier de référence temporelle est résolue en amont : `T2 -> T1`, `T3 -> T2`, `T1 -> T3 de l'année précédente`, `T4 -> T4 de l'année précédente`.

Cette couche ne contient aucune intelligence d'analyse. Elle garantit seulement que le couple documentaire est correct.

### 2. Couche de production par lot

Le point d'entrée naturel est `run_pipeline.py`. Il orchestre :

1. la résolution du couple de trimestres ;
2. la localisation des PDF ;
3. l'extraction des deux rapports ;
4. la génération des artefacts trimestriels ;
5. la comparaison entre les deux extractions ;
6. l'écriture du dossier d'exécution final.

Cette couche doit porter l'ensemble des appels LLM, des règles d'appariement, de la logique d'écart, des métriques d'usage et de la persistance des résultats.

### 3. Couche d'artefacts canoniques

Cette couche est le coeur du système. Chaque période produit un jeu d'artefacts d'extraction ; chaque paire de périodes produit un artefact de comparaison.

Le point fondamental est le suivant : **tout composant aval doit consommer des fichiers, jamais des objets en mémoire issus d'une session Dash non persistée**.

### 4. Couche de revue analyste

Dash doit être traité comme une couche de consultation métier :

- chargement d'un `comparison.json` existant ;
- normalisation UI du payload si nécessaire ;
- affichage des changements et des preuves ;
- écriture éventuelle d'un fichier d'accompagnement pour l'état de revue.

Autrement dit, Dash peut enrichir l'expérience de revue, mais il ne doit pas recalculer la vérité métier.

---

## Contrats de données — vue consolidée

Vue unifiée des deux pipelines.

| Pipeline | Artefact | Producteur | Nature | Rôle |
| --- | --- | --- | --- | --- |
| 1 | `tables.json` | extraction par lot | canonique | vérité d'extraction d'un trimestre |
| 1 | `indicators.json` | projection depuis `tables.json` | dérivé | audit rapide des libellés de lignes |
| 1 | `footnotes.json` | projection depuis `tables.json` | dérivé | audit rapide des notes de bas de page |
| 1 | `comparison.json` | moteur de comparaison de tableaux | officiel | vérité de comparaison de tableaux pour une paire |
| 1 | `manifest.json` | orchestrateur de traitement | administratif | traçabilité de l'exécution et statut de génération |
| 1 | `comparison.review_state.json` | Dash | accompagnement | état de revue analyste tableaux |
| 2 | `text_extraction_{q}.md` | extraction texte par lot | canonique | source de vérité narrative, lisible humain et entrée GPT |
| 2 | `text_extraction_audit.json` | extraction texte par lot | dérivé | traçabilité bloc-à-bloc (inclus / exclu / raison) |
| 2 | `text_comparison.json` | moteur de comparaison texte | officiel | vérité de comparaison textuelle pour une paire |
| 2 | `text_review_state.json` | Dash | accompagnement | état de revue analyste texte |

### Hiérarchie de vérité

La hiérarchie est explicite et identique pour les deux pipelines :

1. **Source canonique** — `tables.json` (pipeline 1), `text_extraction_{q}.md` (pipeline 2). C'est la vérité d'extraction d'un trimestre.
2. **Comparaison officielle** — `comparison.json` (pipeline 1), `text_comparison.json` (pipeline 2). C'est la vérité de comparaison d'une paire.
3. **Dérivés** — `indicators.json`, `footnotes.json`, `text_extraction_audit.json`. Ce sont des vues d'audit, jamais des sources.
4. **État utilisateur** — `*.review_state.json`. C'est un état de revue, sans impact sur la vérité métier.

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

- la préparation des cartes d'appariement ;
- l'appariement des tableaux ;
- l'analyse d'écart sémantique paire par paire ;
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

La seule persistance légitime côté Dash est un fichier d'accompagnement pour la revue analyste.

---

## Lecture experte du besoin métier

Pour un usage de vigie bancaire, la séparation entre traitement par lot et revue n'est pas un détail technique. C'est une exigence de qualité analytique.

### Pourquoi Dash ne doit pas recalculer

Si Dash lance lui-même extraction et comparaison, on crée quatre faiblesses :

1. **Non-reproductibilité** : un même dossier peut être recalculé à des moments différents avec un état de code, de consigne ou de modèle différent.
2. **Confusion de responsabilité** : l'outil de revue devient en même temps un outil de production.
3. **Traçabilité affaiblie** : le point officiel de production n'est plus unique.
4. **Risque de modèle** : l'utilisateur peut croire qu'il lit un résultat stable alors qu'il visualise en fait une exécution opportuniste.

Dans un contexte de lecture de rapports réglementaires, c'est une mauvaise architecture de contrôle.

### Pourquoi les JSON doivent être le contrat central

Le bon niveau de vérité n'est ni la session Python, ni l'état navigateur, ni les structures Dash. Le bon niveau de vérité est le fichier persisté. C'est lui qui supporte :

- l'audit ;
- la preuve ;
- le rechargement ;
- la comparaison d'exécutions ;
- la reprise après incident ;
- l'industrialisation nocturne.

---

## État observé dans le dépôt

À la lecture du code actuel, on observe deux modes de fonctionnement coexistants.

### Mode conforme à l'architecture cible

Le dépôt possède déjà les briques correctes pour un modèle de traitement par lot avec lecture seule :

- `run_pipeline.py` orchestre l'extraction, la comparaison et la génération du dossier d'exécution.
- La commande de comparaison consomme des artefacts déjà extraits.
- `src/app/ui_io.py` sait lister et charger des `comparison.json` déjà présents dans `outputs/comparisons/`.
- Le stockage de revue persiste un état d'analyse à côté du fichier de comparaison.

### Mode hérité ou de développement

Le code Dash contient encore une capacité de lancer extraction et comparaison depuis l'interface, via `run_comparison_with_sections`.

Cette capacité peut être utile en développement, en démonstration ou en diagnostic. En revanche, elle ne doit pas être considérée comme le mode d'exploitation cible si l'objectif est bien :

**générer tous les JSON en amont, puis laisser Dash lire uniquement les fichiers enregistrés.**

---

## Architecture recommandée

### Décision 1. Le traitement par lot devient le seul producteur officiel des artefacts métier

Tous les appels d'extraction et de comparaison doivent partir de `run_pipeline.py` ou de commandes applicatives équivalentes.

### Décision 2. Dash devient officiellement en lecture seule sur la donnée métier

Dash peut charger, filtrer, annoter, exporter et sauvegarder un état de revue. Dash ne doit pas produire `tables.json`, `indicators.json`, `footnotes.json` ou `comparison.json` en mode production.

### Décision 3. Les dossiers d'exécution deviennent immuables

Un dossier d'exécution doit être interprétable comme une photographie complète d'une analyse à date, avec son `comparison.json`, ses artefacts trimestriels et son `manifest.json`.

### Décision 4. Les projections restent des dérivés, jamais des sources

Tout recalcul d'indicateurs ou de footnotes doit repartir de `tables.json`, pas d'une version enrichie côté UI.

### Décision 5. L'UX doit refléter cette séparation

Dans l'interface cible, le premier geste utilisateur n'est pas de téléverser deux PDF pour lancer un traitement. Le premier geste est de choisir un dossier d'exécution déjà produit et de l'examiner.

---

## Recommandations de mise en oeuvre

1. Conserver `run_pipeline.py` comme point d'entrée principal de production.
2. Considérer les capacités de calcul dans Dash comme un mode diagnostic et les masquer en production.
3. Renforcer le dossier d'exécution comme unité d'audit complète.
4. Afficher dans Dash les métadonnées d'exécution, de modèle et d'horodatage pour chaque `comparison.json` chargé.
5. Interdire toute ambiguïté sur la source affichée : la vue doit toujours mentionner le chemin ou l'identifiant du dossier lu.

---

## Conclusion

L'architecture correcte pour ce système n'est pas centrée sur Dash. Elle est centrée sur le moteur de traitement, avec les preuves produites d'abord et la revue ensuite.

Le moteur de traitement génère la matière analytique. Le stockage la fige. Dash la rend exploitable pour l'analyste. C'est cette séparation qui donne au système sa crédibilité opérationnelle, sa robustesse de contrôle et sa valeur dans un contexte de vigie bancaire.

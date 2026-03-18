# Implémentation Structurée du Système GenAI de Comparaison

## 1. But du document

Ce document transforme le plan d'architecture en plan d'implémentation concret, orienté livraison.

Il sert à :

- cadrer les changements à faire dans le dépôt actuel
- clarifier l'ordre d'implémentation
- éviter les refontes inutiles
- maximiser la robustesse, la maintenabilité et la réutilisation
- produire un système crédible devant un superviseur


## 2. Vision d'ensemble de l'implémentation

Le système doit être construit autour de deux niveaux d'artefacts :

### 2.1 Artefacts canoniques par rapport

Ils représentent la source de vérité d'un trimestre donné.

Emplacement cible :

```text
outputs/extractions/{bank}/{year}/{quarter}/
```

Fichiers obligatoires :

- `extraction_snapshot.json`
- `tables.json`
- `meta.json`
- `indicators.json`
- `footnotes.json`


### 2.2 Artefacts dérivés par comparaison

Ils représentent le résultat d'un run de comparaison entre deux trimestres.

Emplacement cible :

```text
outputs/comparisons/...
```

Fichiers obligatoires de cette phase :

- `comparison.json`
- artefacts d'audit existants utilisés par le quality gate

Fichiers à préparer pour la phase suivante :

- `review_state.json`
- `final_report.json`


## 3. Principes d'implémentation retenus

### 3.1 Pas de nouvel appel GPT-4o

Les fichiers `indicators.json` et `footnotes.json` ne doivent jamais être générés via de nouveaux appels GPT.

Ils doivent être dérivés localement à partir des objets déjà extraits.


### 3.2 Évolution additive

L'implémentation doit :

- conserver la compatibilité avec Dash
- conserver la compatibilité avec les exports existants
- conserver les artefacts d'audit de comparaison actuels
- ne pas casser la logique de réutilisation des extractions


### 3.3 Une source de vérité par niveau

- Niveau rapport : `outputs/extractions/...`
- Niveau comparaison : `outputs/comparisons/...`


## 4. Découpage d'implémentation

L'implémentation doit être réalisée en quatre blocs.


## Bloc A — Stabilisation des artefacts d'extraction

### Objectif

Faire de l'extraction trimestrielle une sortie complète, autonome et réutilisable.

### Composants concernés

- `src/app/extraction_storage.py`
- `src/vigilance/extraction/vision_extraction_writer.py`

### Modifications à apporter

#### A.1 Sauvegarde canonique enrichie

Étendre la sauvegarde d'extraction pour produire automatiquement :

- `tables.json`
- `meta.json`
- `extraction_snapshot.json`
- `indicators.json`
- `footnotes.json`

dans le même dossier trimestriel.

#### A.2 Ajout de helpers mono-rapport

Dans `vision_extraction_writer.py`, introduire des helpers publics dédiés à un seul trimestre, distincts des writers actuels orientés comparaison.

Helpers à ajouter :

- `write_report_indicators_json(...)`
- `write_report_footnotes_json(...)`

Ces helpers doivent :

- accepter une seule liste de tables
- écrire une vue orientée rapport
- ajouter les métadonnées utiles : banque, année, trimestre, date

#### A.3 Chargement résilient

Lors du chargement d'une extraction stockée :

- si `tables.json` et `meta.json` existent et sont compatibles, l'extraction reste réutilisable
- si `indicators.json` ou `footnotes.json` manque, ils doivent être régénérés localement
- cette régénération ne doit pas nécessiter de nouvel appel GPT

### Résultat attendu

Un trimestre extrait est immédiatement exploitable pour :

- réutilisation
- audit
- comparaison future
- démonstration métier


## Bloc B — Séparation nette rapport / comparaison

### Objectif

Distinguer clairement les artefacts canoniques d'extraction des artefacts de comparaison, sans casser l'existant.

### Composants concernés

- `src/app/comparison_runner.py`

### Modifications à apporter

#### B.1 Conserver les artefacts de comparaison existants

Ne pas déplacer les fichiers actuellement produits sous `outputs/comparisons/...`.

Ils restent utiles pour :

- le quality gate
- l'audit de run
- la compatibilité avec le système actuel

#### B.2 Enrichir la provenance de la comparaison

Dans `comparison.json`, ajouter un bloc `meta.extraction_sources` contenant, pour chaque trimestre :

- `quarter`
- `mode`: `stored` ou `fresh`
- `artifact_dir`
- `tables_path`
- `indicators_path`
- `footnotes_path`
- `meta_path`
- `artifacts_present`

#### B.3 Décrire le mode de production

La comparaison doit permettre de savoir explicitement :

- si le trimestre a été réutilisé depuis le cache
- s'il a été recalculé
- s'il a été remplacé par une extraction stockée jugée meilleure

### Résultat attendu

Le fichier `comparison.json` devient traçable, défendable et exploitable techniquement.


## Bloc C — Renforcement des exports CSV/Excel

### Objectif

Faire des exports une sortie métier de premier rang dès cette phase.

### Composants concernés

- `src/app/review_export.py`

### Modifications à apporter

#### C.1 Standardisation de la source

L'export doit s'appuyer sur le `comparison.json` canonique enrichi.

#### C.2 Couverture métier complète

L'export doit couvrir correctement :

- tableaux appariés
- tableaux ajoutés
- tableaux supprimés
- indicateurs ajoutés
- indicateurs supprimés
- indicateurs renommés
- footnotes modifiées
- score de confiance
- statut review si présent
- commentaire si présent

#### C.3 Compatibilité stricte

Ne pas casser :

- le schéma CSV existant
- l'ordre des colonnes attendues
- la compatibilité Excel

#### C.4 Tolérance aux futurs champs review

Les futurs champs suivants doivent être tolérés s'ils apparaissent :

- `review_user`
- `review_timestamp`
- `edited_value`

mais ne doivent pas être obligatoires à ce stade.

### Résultat attendu

Même sans persistance complète de review, le système produit déjà un livrable exploitable pour l'analyste et le superviseur.


## Bloc D — Préparation de la review persistée

### Objectif

Préparer la phase suivante sans modifier lourdement Dash maintenant.

### Composants concernés

- `src/app/review_models.py`
- `src/app/review_state.py`

### Modifications à apporter

#### D.1 Étendre le modèle de review

Ajouter des champs optionnels :

- `review_user`
- `review_timestamp`
- `edited_value`

Ces champs doivent être ajoutés dans :

- le dataclass
- `to_dict()`
- `from_dict()`

#### D.2 Garder la logique actuelle

Ne pas implémenter encore :

- l'écriture disque de `review_state.json`
- le merge final avec un rapport final

#### D.3 Préparer la compatibilité future

Les objets de review doivent déjà être prêts à être persistés plus tard sans refonte de modèle.

### Résultat attendu

La phase suivante pourra ajouter `review_state.json` proprement, sans casser :

- Dash
- les exports
- les adapters de review


## 5. Ordre d'exécution recommandé

L'ordre d'implémentation doit être le suivant :

1. Bloc A
2. Bloc B
3. Bloc C
4. Bloc D

Raison :

- les artefacts d'extraction sont la base de tout le reste
- la comparaison dépend de cette provenance
- les exports doivent ensuite s'appuyer sur un contrat stable
- la review persistée peut être préparée sans devenir bloquante


## 6. Critères d'acceptation

### 6.1 Extraction

- une extraction fraîche écrit les cinq fichiers trimestriels attendus
- une extraction stockée compatible se recharge sans réappel GPT
- si `indicators.json` ou `footnotes.json` manque, ils sont régénérés localement

### 6.2 Comparaison

- `comparison.json` reste chargeable par Dash sans adaptation majeure
- `meta.extraction_sources` permet de savoir si chaque trimestre a été chargé ou recalculé
- les chemins vers les artefacts trimestriels sont cohérents

### 6.3 Exports

- le CSV/Excel reste lisible par le superviseur
- les colonnes existantes restent compatibles
- les cas métiers majeurs sont couverts :
  - ajout
  - suppression
  - renommage
  - tableau ajouté
  - tableau supprimé
  - footnote modifiée

### 6.4 Maintenabilité

- aucun doublon de logique métier d'extraction n'est introduit
- les nouveaux helpers sont dédiés et nommés clairement
- les artefacts produits sont compréhensibles par un autre ingénieur


## 7. Risques à éviter

### 7.1 Risques techniques

- regénérer `indicators.json` ou `footnotes.json` via GPT au lieu du local
- casser le format de `comparison.json` attendu par Dash
- casser le schéma CSV exporté
- mélanger artefacts canoniques et artefacts d'audit

### 7.2 Risques d'architecture

- déplacer prématurément les artefacts de comparaison existants
- mélanger stockage “par rapport” et stockage “par comparaison”
- introduire une persistance de review partielle mais incohérente


## 8. Phase suivante déjà préparée

Une fois cette implémentation terminée, la prochaine phase pourra ajouter proprement :

- `review_state.json`
- `final_report.json`
- merge brut + review
- exports finaux basés sur le résultat revu

Le tout sans revenir sur les décisions structurantes de cette phase.


## 9. Conclusion opérationnelle

Cette implémentation structurée permet de :

- garder GPT-4o au centre de la valeur métier
- fiabiliser la réutilisation trimestrielle
- améliorer la traçabilité des comparaisons
- préserver les exports attendus
- préparer le système à une vraie industrialisation

Elle constitue une étape réaliste, robuste et défendable, adaptée à un contexte de stage où l'objectif n'est pas seulement de faire fonctionner le système, mais de montrer une capacité à le structurer pour qu'il survive après le départ du stagiaire.

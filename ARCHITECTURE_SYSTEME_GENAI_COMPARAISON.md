# Architecture du Systeme GenAI de Comparaison de Rapports Trimestriels

## 1. Contexte et objectif

Le systeme a pour objectif de comparer automatiquement des rapports trimestriels bancaires canadiens en s'appuyant fortement sur les capacites multimodales et semantiques de GPT-4o.

Le besoin metier peut etre resume ainsi :

- extraire le contenu pertinent des tableaux presents dans les rapports PDF
- identifier les indicateurs de la premiere colonne
- extraire les notes de bas de tableau
- stocker les resultats d'extraction par banque et par trimestre
- apparier correctement les tableaux entre deux trimestres
- comparer les indicateurs et les footnotes entre deux rapports
- rendre les resultats disponibles rapidement dans une interface Dash
- permettre a l'analyste de valider, rejeter, corriger et commenter les changements
- produire un rapport final par banque apres revue humaine

Le systeme doit egalement permettre la reutilisation des extractions deja calculees. Par exemple, si le trimestre `T2` a deja ete extrait pour une comparaison `T2 vs T1`, alors lors d'une comparaison `T3 vs T2`, seule l'extraction de `T3` doit etre executee et l'extraction de `T2` doit etre reutilisee.


## 2. Principe directeur de conception

L'architecture retenue est une architecture **GenAI-first mais industrialisee**.

Cela signifie que GPT-4o joue un role central dans les taches d'intelligence documentaire et semantique, tandis que les traitements qui exigent stabilite, traçabilite et auditabilite demeurent deterministes.

### 2.1 Taches confiees a GPT-4o

GPT-4o est utilise pour :

- l'extraction Vision du contenu des tableaux
- l'extraction des indicateurs de premiere colonne
- l'extraction des notes de bas de tableau
- la validation semantique des cas ambigus d'appariement de tableaux
- la validation des cas de renommage probable d'indicateurs
- l'interpretation semantique de certaines modifications de footnotes
- la production d'un resume executif des changements significatifs

### 2.2 Taches a conserver deterministes

Les taches suivantes doivent rester deterministes :

- la persistance des donnees
- la gestion du cache
- la reutilisation des extractions existantes
- le calcul des differences
- le suivi des statuts de revue
- l'alimentation rapide de Dash
- la construction du rapport final

Cette separation est essentielle pour garantir :

- la reproductibilite
- la robustesse operationnelle
- la reduction des couts API
- la rapidite d'execution
- la possibilite d'auditer les resultats


## 3. Vision globale du pipeline

Le systeme doit fonctionner en deux temps :

### 3.1 Execution nocturne

Pendant la nuit, le systeme doit :

1. detecter les nouveaux rapports disponibles
2. verifier si chaque rapport a deja ete extrait
3. n'executer l'extraction que pour les rapports absents ou obsoletes
4. charger les trimestres deja extraits si possible
5. lancer les comparaisons requises
6. sauvegarder les resultats de comparaison
7. preparer les artefacts necessaires pour l'interface Dash
8. produire un pre-rapport final

### 3.2 Utilisation matinale par l'analyste

Le matin, l'analyste ne doit pas attendre le recalcul des traitements lourds. L'interface Dash doit :

1. charger directement les artefacts produits pendant la nuit
2. afficher les tableaux compares et les changements detectes
3. permettre la validation humaine
4. sauvegarder les decisions de revue
5. mettre a jour le rapport final

Ainsi, Dash devient une interface de visualisation et de revue, et non un moteur de calcul intensif.


## 4. Architecture fonctionnelle

L'architecture recommande decompose le systeme en cinq modules.

### 4.1 Module d'ingestion

Ce module est responsable de :

- recevoir les rapports PDF par banque, annee et trimestre
- stocker le document source
- calculer une empreinte du PDF
- verifier l'existence d'une extraction precedente
- enregistrer les metadonnees d'ingestion

Les informations minimales a associer a un rapport sont :

- code banque
- annee
- trimestre
- chemin du PDF
- hash du PDF
- date d'ingestion


### 4.2 Module d'extraction GenAI

Ce module est responsable de :

- detecter les tableaux dans le PDF
- recadrer les zones utiles
- appeler GPT-4o Vision sur chaque tableau
- recuperer le titre du tableau
- recuperer les en-tetes
- recuperer les indicateurs de premiere colonne
- recuperer les footnotes
- stocker les resultats d'extraction

Il s'agit du coeur GenAI du systeme au niveau documentaire.


### 4.3 Module de comparaison

Ce module est responsable de :

- charger deux extractions trimestrielles
- apparier les tableaux entre les deux rapports
- comparer les indicateurs
- comparer les footnotes
- solliciter GPT-4o uniquement sur les cas semantiquement ambigus
- persister le resultat de comparaison


### 4.4 Module de revue analyste

Ce module alimente Dash et permet :

- d'afficher les changements detectes
- de fournir un contexte de verification
- d'approuver ou rejeter une decision automatique
- de corriger une interpretation
- d'ajouter un commentaire


### 4.5 Module de generation du rapport final

Ce module reconstruit un resultat final a partir de :

- l'extraction brute
- la comparaison brute
- les decisions de revue humaine

Il produit ensuite un livrable final par banque.


## 5. Organisation recommandee du stockage

Afin de permettre la reutilisation, l'historisation et l'audit, il est recommande de structurer les donnees par banque, annee, trimestre et paire de comparaison.

Exemple :

```text
data/
  rbc/
    2025/
      T1/
        source/
          report.pdf
        extraction/
          extraction_meta.json
          tables.json
          indicators.json
          footnotes.json
      T2/
        source/
          report.pdf
        extraction/
          extraction_meta.json
          tables.json
          indicators.json
          footnotes.json
      comparisons/
        T2_vs_T1/
          comparison.json
          review_state.json
          final_report.json
```

Cette structuration apporte plusieurs avantages :

- une extraction unique par rapport
- une reutilisation simple d'un trimestre deja traite
- une comparaison persistante par paire de trimestres
- une separation claire entre brut, revue et final


## 6. Artefacts d'extraction par rapport

Chaque rapport extrait doit produire plusieurs fichiers specialises.

### 6.1 `tables.json`

Ce fichier represente la sortie d'extraction la plus riche.

Pour chaque tableau, il peut contenir :

- identifiant du tableau
- page
- titre
- section
- en-tetes
- lignes du tableau
- indicateurs bruts
- indicateurs normalises
- footnotes
- coordonnees spatiales eventuelles
- score de confiance
- warnings de qualite
- methode d'extraction


### 6.2 `indicators.json`

Ce fichier contient une vue simplifiee orientee comparaison des indicateurs.

Il peut contenir :

- identifiant du tableau
- titre
- section
- page
- liste d'indicateurs
- liste d'indicateurs normalises


### 6.3 `footnotes.json`

Ce fichier contient une vue simplifiee orientee comparaison des notes.

Il peut contenir :

- identifiant du tableau
- titre
- section
- page
- liste ordonnee des footnotes
- marqueurs normalises


### 6.4 `extraction_meta.json`

Ce fichier sert a la traçabilite et au controle du cache.

Il doit contenir au minimum :

- code banque
- annee
- trimestre
- hash du PDF
- date d'extraction
- modele utilise
- version de prompt
- version du pipeline
- version de schema
- statut global de l'extraction


## 7. Strategie de reutilisation des extractions

La reutilisation est une exigence majeure du systeme.

Avant de lancer une extraction, il faut verifier :

- si les fichiers d'extraction existent deja
- si le hash du PDF est identique
- si la version du pipeline est compatible
- si la version du schema est compatible
- si l'extraction precedente est complete

### Regle de reutilisation

On reutilise une extraction si :

- la banque est la meme
- l'annee est la meme
- le trimestre est le meme
- le PDF est identique
- la version du pipeline reste compatible

Sinon, on relance l'extraction.

### Exemple concret

Pour une comparaison `T2 vs T1` :

- extraction de `T1`
- extraction de `T2`

Pour une comparaison `T3 vs T2` :

- extraction de `T3`
- reutilisation de l'extraction de `T2`

Cette strategie permet de reduire considerablement :

- le temps de traitement
- le cout d'utilisation de GPT-4o
- la duplication des calculs


## 8. Utilisation de GPT-4o pour l'extraction

GPT-4o Vision doit etre considere comme le moteur principal d'extraction des tableaux.

Pour chaque recadrage de tableau, le modele doit retourner une structure contenant au minimum :

- le titre du tableau
- les en-tetes visibles
- les indicateurs de premiere colonne dans l'ordre visuel
- les notes de bas de tableau dans l'ordre visuel
- un score de confiance
- des signaux de troncature ou de mauvaise qualite

Cette approche est pertinente car :

- les tableaux financiers sont souvent visuellement complexes
- les notes sont parfois mal gerees par un OCR simple
- la hierarchie des libelles peut etre implicite
- la restitution de l'ordre visuel est cruciale pour la comparaison


## 9. Conception de l'appariement des tableaux

L'appariement des tableaux entre deux trimestres doit suivre une logique hybride.

### 9.1 Premier passage deterministe

Un score d'appariement peut etre construit a partir de :

- la section
- le numero de tableau
- la similarite des titres
- le recouvrement des indicateurs
- les signaux de structure de page

Ce premier passage permet de produire :

- des appariements a forte confiance
- des tableaux non appariees
- des cas ambigus


### 9.2 Validation semantique par GPT-4o

Seuls les cas ambigus doivent etre soumis a GPT-4o.

Le modele peut alors repondre a une question du type :

- ces deux tableaux correspondent-ils au meme tableau metier ?

Et retourner :

- oui / non / incertain
- un score de confiance
- une justification courte

Cette conception permet d'utiliser GPT-4o de maniere pertinente sans rendre le systeme trop couteux ou trop instable.


## 10. Conception de la comparaison des indicateurs

Une fois les tableaux appariees, on compare leurs indicateurs de premiere colonne.

### 10.1 Difference deterministe

La logique deterministe permet d'identifier :

- les indicateurs inchanges
- les indicateurs ajoutes
- les indicateurs supprimes
- les correspondances exactes ou normalisees

### 10.2 Validation de renommage par GPT-4o

Lorsque deux libelles sont proches mais non identiques, GPT-4o peut etre utilise pour determiner s'il s'agit :

- d'un simple renommage
- d'un changement de sens
- de deux indicateurs differents
- d'un cas incertain

Les statuts possibles deviennent alors :

- inchange
- ajoute
- supprime
- renomme
- incertain


## 11. Conception de la comparaison des footnotes

Les footnotes doivent etre comparees pour chaque paire de tableaux appariees.

### 11.1 Comparaison deterministe

On peut d'abord identifier :

- les notes ajoutees
- les notes supprimees
- les notes modifiees textuellement

### 11.2 Interpretation semantique par GPT-4o

GPT-4o devient utile lorsque :

- la numerotation change mais pas le sens
- la formulation est differente mais la note reste equivalente
- une note signale une modification methodologique reelle

Cette etape est importante car les footnotes portent souvent de l'information reglementaire ou methodologique.


## 12. Artefact de comparaison

Chaque paire de trimestres doit produire un artefact de comparaison unique, par exemple :

- `comparison.json`

Cet artefact doit contenir :

- l'identite de la comparaison
- un resume global
- les comparaisons par paire de tableaux
- la liste des tableaux ajoutes
- la liste des tableaux supprimes
- les metadonnees de validation et de qualite

### 12.1 Bloc `summary`

Le resume global doit contenir :

- le nombre de tableaux extraits
- le nombre de tableaux comparables
- le nombre de tableaux apparies
- le nombre de tableaux ajoutes
- le nombre de tableaux supprimes
- le nombre de cas ambigus
- le nombre d'indicateurs ajoutes, supprimes et renommes
- les compteurs de changements de footnotes

### 12.2 Bloc `table_comparisons`

Chaque entree represente une paire de tableaux appariees.

Elle peut contenir :

- identifiant du tableau du trimestre precedent
- identifiant du tableau du trimestre courant
- titres
- pages
- section
- indicateurs ajoutes
- indicateurs supprimes
- indicateurs renommes
- changements de footnotes
- statut global du tableau
- score de confiance ou statut de validation

### 12.3 Blocs `tables_added` et `tables_removed`

Ces blocs recensent :

- les tableaux presents uniquement dans le trimestre courant
- les tableaux presents uniquement dans le trimestre precedent

### 12.4 Bloc `meta`

Ce bloc doit contenir :

- date de generation
- version du modele
- version du prompt
- version du pipeline
- metriques de validation
- resume executif si necessaire


## 13. Gestion de la revue humaine

La revue humaine doit etre tracee dans un artefact distinct du brut.

Il ne faut jamais ecraser les donnees d'extraction brutes ni les donnees de comparaison brutes.

### 13.1 Fichier `review_state.json`

Ce fichier peut enregistrer, pour chaque element revu :

- identifiant de l'element
- type d'element
- statut de revue
- utilisateur
- horodatage
- commentaire
- valeur corrigee si l'analyste modifie un resultat

### 13.2 Statuts de revue recommandes

- `pending`
- `approved`
- `rejected`
- `edited`

### 13.3 Actions disponibles dans Dash

L'analyste doit pouvoir :

- approuver
- rejeter
- corriger
- commenter

Cette approche garantit une piste d'audit claire.


## 14. Construction du rapport final

Le rapport final doit etre produit a partir de trois couches :

1. extraction brute
2. comparaison brute
3. decisions de revue

Le resultat final par banque doit contenir :

- l'identite de la banque
- les trimestres compares
- les changements valides
- les changements rejetes ou corriges
- les commentaires analyste
- un eventuel resume executif

Les formats de sortie peuvent inclure :

- JSON
- CSV
- Excel
- PDF de synthese


## 15. Principe de conception de Dash

Dash ne doit pas recalculer les traitements lourds au moment de l'ouverture.

Au contraire, l'interface doit uniquement :

- charger les artefacts deja prepares
- afficher les resultats de comparaison
- afficher les details necessaires a la verification
- enregistrer les decisions de revue

Cette contrainte est essentielle pour garantir :

- un chargement rapide
- une experience utilisateur stable
- une independance vis-a-vis des appels API en temps reel


## 16. Gouvernance technique et auditabilite

Chaque artefact persiste doit etre versionne.

Les champs recommandes sont :

- `model_version`
- `prompt_version`
- `pipeline_version`
- `schema_version`
- `created_at`
- `source_pdf_hash`

Cette gouvernance est indispensable car :

- les prompts peuvent evoluer
- les schemas peuvent changer
- le comportement du modele peut varier
- les resultats doivent rester explicables


## 17. Pourquoi cette architecture est pertinente

Cette architecture repond aux attentes du superviseur pour plusieurs raisons :

- GPT-4o est reellement au coeur du systeme
- les traitements lourds peuvent etre executes la nuit
- les analystes peuvent consulter les resultats rapidement le matin
- les extractions sont reutilisables d'un trimestre a l'autre
- la revue humaine est integree
- le rapport final reste traçable et auditable

Elle evite egalement plusieurs erreurs classiques :

- recalculer les extractions a chaque comparaison
- appeler GPT-4o a l'ouverture de Dash
- melanger les corrections humaines avec les sorties brutes
- confier a GenAI des traitements qui doivent rester deterministes


## 18. Positionnement du systeme vis-a-vis de GPT-4o

Il est pertinent de presenter le systeme comme un **systeme de comparaison trimestrielle fortement base sur GPT-4o**, mais non entierement delegue a GPT-4o.

La formulation la plus juste est la suivante :

> GPT-4o constitue le moteur principal d'extraction multimodale et de validation semantique du systeme, tandis que les couches de persistance, de reexecution, de calcul de differences, de revue et de reporting final reposent sur une logique deterministe afin de garantir rapidite, fiabilite et traçabilite.

Cette formulation est techniquement solide et defendable.


## 19. Perimetre minimal de livraison

Si le temps de mise en oeuvre est limite, il est recommande de prioriser les composants suivants :

1. extraction persistante par rapport
2. reutilisation d'un trimestre deja extrait
3. comparaison persistante par paire de trimestres
4. revue analyste persistante dans Dash
5. generation d'un rapport final revu

Ce noyau constitue une premiere livraison coherente, exploitable et defendable devant un superviseur.


## 20. Conclusion

Le systeme propose repose sur une logique claire : utiliser GPT-4o la ou il apporte une valeur differenciante, c'est-a-dire dans l'extraction visuelle et l'interpretation semantique, tout en conservant une architecture batch, persistante et auditable pour l'orchestration globale.

Une telle architecture permet :

- une comparaison trimestrielle fiable
- une reutilisation efficace des calculs deja effectues
- une consultation rapide dans Dash
- une revue humaine robuste
- une production finale exploitable par banque

En pratique, cette approche constitue un compromis solide entre puissance GenAI, qualite logicielle et exigences operationnelles.

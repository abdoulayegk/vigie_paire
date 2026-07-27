# Correction de la sous-classification des changements MAJEUR, MODÉRÉ et MINEUR

## 1. Objet du document

Ce document décrit un problème de sous-classification observé dans le pipeline
d'analyse textuelle. Le système détecte correctement de nombreux changements,
mais certains changements ayant une portée métier, prudentielle ou
réglementaire sont classés **MINEUR** alors qu'ils devraient être présentés à
l'analyste comme **MODÉRÉ** ou **MAJEUR**.

L'objectif n'est pas de rendre automatiquement majeur tout changement contenant
un terme sensible. L'objectif est de faire dépendre le niveau d'impact du
**changement de sens métier démontré**, et non de la longueur du texte, de sa
ressemblance visuelle ou de la seule présence d'une « nouvelle idée ».

Ce document conserve le diagnostic fonctionnel initial et décrit désormais
l'architecture effectivement mise en place dans la branche de correctif. Il ne
présente pas de code. Les mécanismes et leurs garde-fous sont implémentés, mais
aucune campagne complète de validation métier sur les six banques n'a encore
été exécutée. Les niveaux recommandés dans les exemples demeurent donc des
hypothèses à confirmer par des analystes.

## 2. Problème observé

Les cas examinés montrent que le système trouve généralement le passage modifié,
mais sous-estime parfois son importance. Les erreurs se concentrent notamment
sur les changements suivants :

- changement de terminologie dans un domaine sensible;
- modification du périmètre d'un risque;
- reclassement d'un risque vers une autre catégorie;
- modification d'une définition;
- ajout ou retrait d'une responsabilité;
- modification d'un objectif de gestion du capital;
- retrait d'une information qualitative accompagné de variations numériques;
- suppression d'une divulgation ou d'un graphique;
- changement réglementaire ou changement de statut d'application d'une loi;
- réorganisation de plusieurs passages liés qui, pris séparément, paraissent
  mineurs, mais qui forment ensemble une évolution structurante.

Le problème n'est donc pas principalement un problème de détection. Il s'agit
d'un problème de **qualification métier et de hiérarchisation**.

## 3. Exemple représentatif : gestion du capital

Deux remplacements ont été signalés :

- « groupes d'exploitation » devient « unités d'exploitation »;
- « suffisance du capital » devient « adéquation des fonds propres ».

Pris isolément, chacun de ces remplacements peut correspondre à une
harmonisation terminologique. Ils ne doivent donc pas être automatiquement
classés MAJEUR sur la seule base des mots employés.

Cependant, le passage complet sur la répartition des fonds propres ajoute
également deux éléments :

- la répartition des fonds propres sert désormais explicitement à « guider la
  répartition des ressources »;
- l'objectif est formulé comme le fait de « surveiller et d'optimiser » les
  rendements ajustés en fonction des risques.

L'ensemble ne constitue plus une simple substitution de vocabulaire. Il peut
modifier la compréhension de la finalité du processus d'allocation du capital.
Ce dossier devrait donc être classé au minimum **MODÉRÉ**, et potentiellement
**MAJEUR** si le contexte confirme une modification du périmètre des unités, de
la responsabilité ou du processus d'allocation.

Le remplacement de « suffisance du capital » par « adéquation des fonds propres »
reste plus ambigu. Il peut signaler l'adoption d'une terminologie prudentielle
plus formelle, mais il peut aussi s'agir d'une harmonisation de traduction. En
l'absence de preuve d'une nouvelle méthode, d'un nouveau seuil, d'un nouveau
périmètre ou d'une nouvelle responsabilité, le classement recommandé est
**MODÉRÉ à examiner**, plutôt que MAJEUR automatique ou MINEUR automatique.

Cet exemple montre deux besoins :

1. examiner le sens du passage complet plutôt qu'une différence lexicale isolée;
2. regrouper les changements liés avant d'attribuer la sévérité finale.

## 4. Fonctionnement historique à l'origine de la sous-classification

Le pipeline historique séparait la décision en deux étapes.

La première étape demande au modèle de déterminer principalement :

- si le changement est pertinent;
- s'il constitue une nouvelle idée;
- les thèmes AMF concernés.

Le modèle n'attribue pas directement le niveau MAJEUR, MODÉRÉ ou MINEUR.

La deuxième étape applique une grille fixe :

| Résultat intermédiaire | Niveau attribué |
|---|---|
| Non pertinent | MINEUR |
| Pertinent sans nouvelle idée | MINEUR |
| Pertinent avec nouvelle idée | MODÉRÉ |
| Nouvelle idée associée à un thème prioritaire | MAJEUR |

Cette logique crée un goulot d'étranglement : dès que le modèle répond
« nouvelle idée = non », le changement ne peut plus devenir MODÉRÉ ou MAJEUR,
même s'il modifie un périmètre, une définition, une responsabilité ou la
transparence d'une divulgation.

Dans la branche de correctif, cette ancienne grille demeure calculée uniquement
comme point de comparaison et mécanisme de compatibilité pour les anciens
artefacts. Lorsqu'une décision directe de matérialité est disponible, c'est
elle qui alimente le niveau final. L'ancien niveau reste conservé dans l'audit
afin de mesurer les reclassements produits par la nouvelle approche.

La notion de « nouvelle idée » est utile, mais elle ne représente pas à elle
seule la matérialité. Les situations suivantes peuvent être importantes sans
être naturellement décrites comme une nouvelle idée :

- réduction du périmètre d'un risque;
- suppression d'une information;
- renommage d'une unité qui modifie son référent organisationnel;
- reclassement d'un risque existant;
- modification du statut d'une exigence réglementaire;
- retrait d'une responsabilité ou d'un contrôle;
- modification de la comparabilité d'un indicateur prudentiel.

## 5. Autres causes probables

### 5.1 Analyse trop locale

Les changements sont souvent qualifiés fragment par fragment. Plusieurs
modifications mineures en apparence peuvent pourtant former une évolution
majeure lorsqu'elles sont regroupées à l'échelle d'une sous-section, d'un thème
ou de l'ensemble du rapport.

Une modification répétée de « groupes » vers « unités », combinée à des
changements d'objectifs, de tableaux et de responsabilités, doit être évaluée
comme un ensemble cohérent.

### 5.2 Confusion entre même divulgation et même sens

Deux passages peuvent traiter de la même divulgation tout en contenant un
changement substantiel. La conclusion « même divulgation » ne doit pas être
interprétée comme « aucun changement de fond ».

Lorsqu'une étape antérieure affirme déjà qu'une modification « ne change pas le
sens fondamental », cette formulation peut influencer la qualification
ultérieure. L'évaluation de la matérialité doit être indépendante de la décision
d'alignement.

### 5.3 Absence d'un traitement explicite de l'incertitude

Le système tend à transformer l'absence de preuve claire en verdict MINEUR. Dans
un contexte de vigie, cela crée des faux négatifs coûteux.

Un changement sensible dont l'équivalence n'est pas démontrée devrait être
classé MODÉRÉ pour revue, et non MINEUR par défaut.

### 5.4 Exclusion trop large des variations propres à la banque

Une variation chiffrée propre à une banque peut être non pertinente. Toutefois,
un passage contenant des chiffres peut aussi retirer une information
qualitative sur :

- la diversification d'un portefeuille;
- une structure de limites;
- des équipes spécialisées;
- une méthode d'atténuation;
- un contrôle ou un processus de surveillance.

Le caractère numérique d'une partie du passage ne doit pas masquer la
suppression de son contenu qualitatif.

### 5.5 Propagation d'un verdict entre changements similaires

La déduplication est utile pour réduire le volume, mais deux occurrences d'un
même remplacement peuvent avoir des conséquences différentes selon leur
contexte. Un changement terminologique dans une phrase descriptive n'a pas
nécessairement la même importance que le même changement dans une responsabilité
du conseil ou dans un processus d'allocation du capital.

Le regroupement devrait servir à mesurer un effet cumulatif, et non seulement à
propager le verdict du premier fragment.

## 6. Principe directeur de la solution

La solution ne consiste pas à supprimer toutes les règles. Un classement sans
définitions stables serait difficile à reproduire et à auditer.

La solution consiste à :

1. conserver des règles strictes pour le bruit incontestable;
2. demander une analyse sémantique directe de l'impact métier;
3. séparer la pertinence, la nouveauté et la matérialité;
4. traiter l'incertitude de manière prudente;
5. consolider les changements liés avant le verdict final;
6. utiliser les décisions validées par les analystes comme précédents de
   calibration.

La politique générale proposée est la suivante :

> Dans un domaine sensible, un changement ne peut rester MINEUR que si son
> équivalence métier, son déplacement ou son caractère purement rédactionnel est
> démontré. En cas d'incertitude sémantique réelle, il est classé MODÉRÉ pour
> revue. Le niveau MAJEUR exige un effet substantiel démontré.

## 7. Dimensions à évaluer indépendamment

Pour chaque changement, l'analyse devrait produire quatre décisions
indépendantes.

### 7.1 Pertinence

Le changement apporte-t-il une information utile à la vigie prudentielle ou à
la gestion des risques de la banque analysée?

### 7.2 Nature du changement

Le changement correspond-il à :

- une équivalence terminologique;
- une clarification;
- une addition;
- un retrait;
- une modification de définition;
- une modification de périmètre;
- un reclassement;
- une modification de méthode;
- une modification de gouvernance ou de responsabilité;
- une modification réglementaire;
- une modification de contrôle ou de processus;
- une réorganisation éditoriale;
- un déplacement de contenu?

### 7.3 Matérialité

Quelle est l'incidence réelle sur la compréhension du dispositif, de la
divulgation ou du risque?

### 7.4 Confiance

Les textes fournis permettent-ils de conclure avec une confiance élevée,
moyenne ou faible? L'incertitude doit être visible et doit influencer le niveau
de revue.

Ces quatre dimensions ne doivent pas être déduites automatiquement les unes des
autres. Un changement peut être pertinent, ne pas être une nouvelle idée et
être néanmoins MODÉRÉ ou MAJEUR.

## 8. Nouvelle définition des niveaux

### 8.1 MINEUR

Le niveau MINEUR est approprié lorsque l'absence d'impact de fond est démontrée.

Exemples :

- correction typographique;
- formatage ou ponctuation;
- mise à jour d'une date sans changement de statut;
- variation chiffrée normale sans changement de méthode ni de périmètre;
- remplacement terminologique dont l'équivalence est confirmée par le contexte;
- déplacement intégral d'un passage sans perte d'information;
- changement de titre décrivant exactement le même contenu.

Un classement MINEUR dans un domaine sensible doit expliquer pourquoi les deux
formulations sont réellement équivalentes.

### 8.2 MODÉRÉ

Le niveau MODÉRÉ est le niveau prudent lorsque le changement peut influencer
l'analyse, mais que son effet substantiel n'est pas entièrement démontré.

Exemples :

- terminologie sensible dont le référent réel demeure incertain;
- retrait d'une information qualitative qui pourrait être reprise ailleurs;
- modification d'une divulgation réglementaire sans nouvelle obligation
  clairement identifiée;
- changement de structure qui pourrait être éditorial ou métier;
- suppression d'un graphique dont les données pourraient subsister sous une
  autre forme;
- évolution du statut d'application d'une loi;
- changement touchant le capital, les APR, la gouvernance, la conformité, la
  liquidité, le climat, la cybersécurité, l'IA, les tiers ou les modèles lorsque
  l'équivalence n'est pas démontrée.

### 8.3 MAJEUR

Le niveau MAJEUR est approprié lorsqu'un effet substantiel est démontré.

Exemples :

- modification de l'autorité décisionnelle ou de la responsabilité;
- ajout ou retrait d'un comité, d'un contrôle ou d'une ligne de défense;
- modification d'une méthode de calcul, d'un modèle, d'un ratio ou d'un seuil;
- modification du périmètre d'un risque ou d'une exposition;
- reclassement d'un risque dans une autre catégorie;
- modification structurante d'une définition;
- nouvelle obligation réglementaire applicable;
- changement de la logique d'allocation du capital ou des fonds propres;
- suppression complète d'une divulgation matérielle non reprise ailleurs;
- refonte démontrée d'un cadre de gestion des risques;
- combinaison de plusieurs changements concordants révélant une nouvelle
  approche de gestion.

La simple présence d'un mot lié au capital, au BSIF, au climat ou à la
cybersécurité ne suffit pas à attribuer MAJEUR.

## 9. Architecture implémentée des sept leviers

La branche de correctif met en œuvre les sept leviers retenus. Ils forment une
chaîne unique : relation documentaire factuelle, filtrage mécanique limité,
dossier de changements reliés, jugement direct, recherche de précédents,
contestation ciblée, puis mesure parallèle.

### Levier 1 — Décorréler la matérialité de la « nouvelle idée »

Le triage produit maintenant une décision directe de matérialité comprenant :

- le niveau MAJEUR, MODÉRÉ ou MINEUR;
- une à trois natures de changement;
- le degré d'équivalence métier;
- la confiance;
- la suffisance de la preuve;
- le statut confirmé, provisoire ou à confirmer;
- l'indication qu'une revue est requise;
- les preuves favorables et les contre-arguments.

Le champ « nouvelle idée » demeure disponible comme attribut descriptif, mais
il ne commande plus le niveau. Un changement pertinent déjà connu peut donc
être classé MAJEUR si une modification substantielle de périmètre, d'autorité,
de responsabilité, de méthode ou de contrôle est démontrée.

Les anciens artefacts restent lisibles et peuvent encore utiliser l'ancienne
grille comme repli. En revanche, toute nouvelle réponse de triage doit contenir
une décision directe complète et non nulle : une omission est refusée au lieu
d'être silencieusement convertie par l'ancienne logique. L'ancien niveau et le
nouveau niveau sont tous deux conservés pour l'audit.

### Levier 2 — Transformer les exclusions larges en indices

Les signaux d'acquisition, de calendrier, de variation numérique ou de forte
similarité ne constituent plus à eux seuls un veto après l'analyse métier. Ils
sont transmis au juge comme indices auditables.

L'exclusion automatique est réservée aux situations mécaniquement démontrées :

- formatage visuel réellement équivalent;
- déplacement de texte confirmé avec une confiance élevée, une couverture
  complète des fragments et une équivalence textuelle vérifiée;
- variation purement numérique, sans signal de méthode, de processus, de
  gouvernance, de seuil ou d'exigence.

Ainsi, un passage mentionnant une acquisition peut tout de même devenir MAJEUR
s'il ajoute en même temps un risque lié aux données, aux tiers ou aux contrôles.
De même, un changement de date atteint le juge de matérialité s'il modifie le
statut réel de mise en œuvre d'une exigence.

Un déplacement de confiance moyenne ou faible demeure visible et exige une
revue, même si les extraits paraissent identiques. De même, une paire entièrement
couverte mais reformulée reste disponible pour le jugement de matérialité :
« même divulgation » ne signifie pas « même sens ». Lorsqu'une divulgation est
partiellement modifiée, seules les portions textuellement équivalentes peuvent
être soustraites; les portions non équivalentes et leurs contextes avant/après
restent transmis au triage. Le contrôle d'équivalence conserve notamment les
séparateurs décimaux, les signes, les parenthèses comptables, les unités et les
opérateurs numériques afin qu'une variation de valeur ne soit jamais assimilée
à un simple changement de ponctuation. Il vérifie aussi la structure des
rapprochements : les mêmes acteurs et verbes réassociés différemment ne peuvent
pas être supprimés comme un déplacement inchangé. L'automatisation accepte un
texte complet équivalent ou le découpage d'un bloc en fragments entiers
équivalents; les assemblages croisés restent à confirmer.

### Levier 3 — Consolider sans propager aveuglément le verdict

Les changements situés dans la même sous-section peuvent recevoir un contexte
factuel réciproque. Toutefois, une sous-section commune ne suffit pas à créer
un dossier cumulatif. Seuls les changements sémantiquement proches et
compatibles forment un groupe consolidé.

Chaque membre est néanmoins classé séparément. Le verdict d'un représentant
n'est plus copié aux autres membres. Le résultat conserve :

- le niveau propre à chaque changement;
- les identifiants des membres du groupe;
- l'absence explicite de propagation;
- un jugement collectif indépendant, capable de reconnaître qu'une combinaison
  de changements concordants a une portée supérieure à chacun des fragments;
- un niveau consolidé au moins égal au niveau individuel le plus élevé, mais
  pouvant être relevé par ce jugement collectif;
- la pertinence consolidée du dossier.

La réconciliation globale des déplacements est réalisée avant le triage, ce qui
réduit aussi le risque de traiter comme suppression une information reprise
ailleurs.

### Levier 4 — Séparer la constatation factuelle du jugement métier

L'étape d'alignement doit uniquement déterminer si deux passages concernent la
même divulgation, des divulgations distinctes, un déplacement ou une relation
incertaine. Elle ne doit plus conclure que le changement est « sans effet » ou
« sans changement de fond ».

Le triage ne reçoit plus le raisonnement antérieur susceptible de l'ancrer. Il
reçoit plutôt :

- les textes avant et après;
- les différences exactes;
- un résumé factuel non arbitré;
- les preuves complètes lorsque leur taille le permet;
- les observations issues des preuves longues;
- les changements reliés et les indices mécaniques.

La décision « même divulgation » signifie donc seulement « même sujet
documentaire ». Elle ne démontre jamais l'équivalence métier.

### Levier 5 — Introduire une mémoire de précédents validés

Une mémoire autonome charge les décisions d'analystes présentes dans les
artefacts de revue. Elle exige une portée de décision « matérialité » et une
version de schéma reconnue. Elle exclut ainsi les anciennes validations qui
portaient sur un autre objet, les sorties brutes du modèle, les décisions
passées, les commentaires libres sans correction structurée et les décisions
qui exigent encore une revue.

Une approbation peut valider une décision structurée existante. Un rejet ne
devient un précédent que si l'analyste fournit explicitement un niveau final,
une nature de changement, l'équivalence métier, la confiance, la suffisance des
preuves, les thèmes requis et une justification structurée.

Pour chaque nouveau cas, la mémoire produit un petit paquet comprenant par
défaut :

- des précédents proches associés au niveau d'ancrage;
- des cas contrastifs proches mais classés différemment;
- le score de rapprochement;
- la provenance de la validation;
- une empreinte stable de la requête.

La recherche du pipeline principal combine maintenant la proximité sémantique
et la similarité lexicale. En cas d'indisponibilité du moteur de représentation
sémantique, elle revient automatiquement au mode lexical local. Dans les deux
cas, une simple correspondance de section ou de thème ne suffit pas : un
recouvrement textuel réel ou une proximité sémantique minimale est exigé. Des
décisions de même priorité portant le même identifiant mais se contredisant
sont mises en quarantaine.

Les précédents aident à comparer les ressemblances et les différences; ils ne
remplacent jamais les preuves du cas courant et leur niveau n'est pas copié
automatiquement.

### Levier 6 — Ajouter une seconde lecture aveugle

Une seconde évaluation est déclenchée pour :

- les verdicts MINEUR portant sur un thème sensible;
- les décisions à confirmer;
- les décisions ayant une confiance faible ou indéterminée;
- les preuves insuffisantes ou indéterminées;
- les cas déjà signalés comme nécessitant une revue;
- les verdicts « non pertinent » dont les textes contiennent néanmoins un
  signal sensible de capital, de gouvernance, de méthode, de contrôle,
  d'obligation réglementaire, de retrait de divulgation ou de risque émergent.

Cette lecture ne voit pas le verdict primaire. Elle reçoit les preuves, le
dossier relié, les thèmes candidats, les indices, les précédents et, lorsqu'ils
existent, les paquets de preuves exactes complets ou leurs observations. Elle
cherche aussi bien une sous-classification qu'un surclassement.

Lorsque le contradicteur démontre un niveau supérieur, ce niveau est retenu.
Tout désaccord de pertinence, de niveau, de thème ou d'équivalence est conservé
dans l'audit et force le statut « à confirmer » avec revue analyste. Si la
seconde lecture échoue techniquement, le système ne confirme pas silencieusement
le premier verdict : il exige également une revue.

Cette indépendance est une indépendance de contexte et de lecture, pas encore
une indépendance de fournisseur ou de modèle : les deux appels utilisent
actuellement le même modèle avec des consignes séparées.

### Levier 7 — Mesurer l'ancien et le nouveau système en parallèle

Un évaluateur séparé compare, sur un même corpus de référence :

- la décision validée par l'analyste;
- l'ancien niveau;
- le niveau candidat produit par la nouvelle architecture.

Il génère les matrices de confusion, les erreurs MAJEUR vers MINEUR, MAJEUR
vers MODÉRÉ et MODÉRÉ vers MINEUR, le rappel des changements non mineurs, le
rappel de MAJEUR, la précision de MAJEUR, la couverture automatique, le taux de
revue, l'accord exact et l'accord ordinal pondéré. Une décision
« A_CONFIRMER », quel que soit son niveau provisoire, est comptée comme une
abstention de revue plutôt que comme un verdict automatique. Les résultats sont
ventilés par banque, thème et nature de changement.

Le dispositif applique les seuils globalement et séparément par banque. Il
vérifie également que BMO, BNC, BNS, CIBC, RBC et TD disposent chacune du
volume minimal configuré, soit vingt cas par défaut. Le rapport contient une
empreinte du corpus de référence indépendante des prédictions et une empreinte
distincte de l'évaluation effectuée. Il est toujours produit; le blocage
automatique sur les seuils ou sur la couverture bancaire est activé seulement
lorsqu'il est explicitement demandé.

Cette infrastructure est implémentée et testée techniquement. Elle n'a pas
encore été exécutée sur un corpus complet de référence validé par les analystes;
aucun résultat de performance n'est donc affirmé dans ce document.

## 10. Traitement recommandé des exemples observés

Les recommandations suivantes sont provisoires : elles doivent être confirmées
à partir des passages complets et après vérification des déplacements.

| Famille de cas | Niveau recommandé |
|---|---|
| Terminologie des unités d'exploitation et de l'adéquation des fonds propres | MODÉRÉ; MAJEUR si changement de périmètre, de responsabilité ou de méthode |
| Objectif ou logique de répartition des fonds propres | MAJEUR si la modification substantielle est confirmée |
| Libellé des APR | MODÉRÉ; MINEUR seulement si l'équivalence est démontrée |
| Reclassement du risque d'assurance | MAJEUR si le rattachement change réellement |
| Loi sur l'impôt minimum mondial | MODÉRÉ; MAJEUR si une nouvelle obligation ou un nouvel impact est démontré |
| Retrait de facteurs de risque ou de jugements comptables | MODÉRÉ; MAJEUR si l'information structurante disparaît complètement |
| Refonte du cadre de gestion des risques | MAJEUR si le fond du cadre change; MODÉRÉ si la refonte est éditoriale |
| Remplacement d'une section de surveillance par les APR et les simulations de crise | MAJEUR si la structure métier change; MODÉRÉ si le contenu est seulement déplacé |
| Suppression de graphiques ou de divulgations d'exposition | MODÉRÉ; MAJEUR si l'information unique disparaît |
| Financement à levier financier | MODÉRÉ; MAJEUR si le périmètre ou le dispositif de surveillance est retiré |
| Modification d'une définition de risque | MAJEUR si le périmètre de la définition change |
| Sécurité physique remplacée par fraude interne ou externe | MAJEUR si le facteur de risque est reclassé ou réduit |
| Protection des consommateurs ou réglementation américaine | MODÉRÉ; MAJEUR si les obligations, contrôles ou expositions changent |
| Nouvelle structuration des risques climatiques | MODÉRÉ; MAJEUR si une nouvelle gouvernance, stratégie ou obligation est introduite |

## 11. Critères d'acceptation

Ces critères demeurent les conditions métier de sortie. Leur prise en charge
technique est en place, mais ils ne sont pas encore démontrés sur un corpus
complet validé par les analystes. La correction sera considérée comme
satisfaisante lorsque les conditions suivantes seront respectées :

1. un changement pertinent sans nouvelle idée peut recevoir MODÉRÉ ou MAJEUR;
2. tout classement MINEUR dans un domaine sensible contient une preuve
   d'équivalence, de déplacement ou de caractère rédactionnel;
3. un changement de périmètre, de responsabilité, de méthode ou de définition
   ne peut pas être MINEUR lorsque cet effet est démontré;
4. une variation numérique ne masque pas une suppression qualitative;
5. les changements liés sont évalués individuellement et collectivement;
6. l'incertitude dans un domaine sensible conduit à MODÉRÉ;
7. MAJEUR reste réservé aux effets substantiels démontrés;
8. la décision est accompagnée des textes sources et d'une justification
   vérifiable;
9. les cas corrigés par les analystes sont réutilisables comme précédents;
10. le jeu de validation historique ne contient aucun cas MAJEUR validé par un
    analyste qui soit automatiquement rétrogradé à MINEUR.

## 12. Mesure de la qualité

La qualité ne doit pas être évaluée uniquement par le pourcentage global de
bonnes classifications. Les erreurs n'ont pas toutes le même coût.

Les mesures prioritaires sont :

- taux de changements MAJEUR ou MODÉRÉ classés à tort MINEUR;
- taux de rappel des changements non mineurs;
- précision de la catégorie MAJEUR;
- proportion des verdicts MINEUR sensibles accompagnés d'une preuve
  d'équivalence;
- taux de désaccord entre le système et les analystes;
- répartition des corrections MINEUR vers MODÉRÉ et MINEUR vers MAJEUR;
- performance par thème : capital, gouvernance, réglementation, risques,
  conformité, climat, cyber, IA, données et tiers.

Dans ce contexte, le coût d'un faux négatif est supérieur au coût d'une revue
MODÉRÉ supplémentaire. Le système doit donc privilégier le rappel des
changements substantiels, sans transformer tous les changements sensibles en
MAJEUR.

## 13. Risques de la solution et mesures de contrôle

### Surclassement généralisé

Si tout changement contenant un terme sensible devient MAJEUR, la file de
priorité perd sa valeur. La classification doit reposer sur l'effet démontré, et
non sur un mot-clé.

### Instabilité du jugement

Une évaluation entièrement libre peut varier. Des définitions précises, des
exemples validés et une vérification indépendante sont nécessaires.

### Confusion entre divulgation et pratique réelle

La suppression d'un texte prouve un changement de divulgation, mais ne prouve
pas nécessairement que la pratique de gestion a été supprimée. La justification
doit distinguer clairement ces deux conclusions.

### Perte de contexte lors du découpage

Une décision locale peut être incorrecte si le passage a été déplacé ou si la
suite du paragraphe contient l'information manquante. La vérification doit
utiliser la sous-section complète et rechercher les reprises ailleurs dans le
rapport.

## 14. Décision de mise en service recommandée

La branche de correctif rend maintenant le niveau d'impact **indépendant de la
seule notion de nouvelle idée** et ajoute :

- une analyse sémantique directe de la matérialité;
- une consolidation des changements liés sans propagation automatique;
- un traitement explicite de l'incertitude;
- une contestation aveugle des verdicts MINEUR sensibles;
- une mémoire contrôlée des décisions analystes;
- une évaluation parallèle de l'ancien et du nouveau classement.

La règle opérationnelle centrale est :

> MINEUR exige une équivalence démontrée. MODÉRÉ protège les cas sensibles ou
> incertains. MAJEUR exige un effet métier substantiel démontré.

La mise en service ne doit toutefois pas reposer uniquement sur la réussite des
tests techniques. Elle doit être conditionnée à la constitution du corpus de
référence, à l'exécution du protocole parallèle sur les six banques, à une revue
des désaccords et à l'acceptation explicite des seuils par les responsables
métier.

## 15. Boucle contrôlée de précédents validés

La boucle d'adaptation fonctionne sans réentraînement automatique du modèle et
sans réécriture autonome des consignes.

1. Le pipeline parcourt le répertoire de résultats qui lui est fourni et
   recherche les comparaisons textuelles, les états de revue et les registres
   de précédents.
2. Le chargeur conserve seulement les décisions finales suffisamment
   structurées. Une correction explicite a priorité sur une simple approbation
   en cas de doublon. Deux décisions de même priorité qui se contredisent sous
   le même identifiant sont retirées de la mémoire et signalées comme conflit.
3. Chaque précédent contient les textes avant et après, la banque, la section,
   la nature, le niveau, l'équivalence, les thèmes, la justification, les
   preuves, les contre-arguments, l'auteur de la validation et sa provenance.
4. Pour le nouveau changement, le système recherche des cas proches et des cas
   contrastifs.
5. Le paquet compact est transmis au juge principal et, lorsque nécessaire, au
   contradicteur. Les scores et l'identité des précédents restent visibles dans
   l'audit du changement.
6. L'analyste approuve la décision ou la corrige en renseignant un niveau, une
   ou plusieurs natures, l'équivalence métier, les thèmes requis, la confiance,
   la suffisance des preuves et une justification.
7. Lors d'une exécution ultérieure, cette décision validée peut devenir un
   précédent pour des cas similaires, y compris dans une autre banque lorsque
   le même répertoire de résultats partagé est utilisé.

Cette boucle est volontairement prudente :

- une sortie automatique non revue ne s'auto-valide jamais;
- une décision passée ou à confirmer n'entre pas dans la mémoire;
- une validation historique sans portée « matérialité » ni schéma reconnu
  n'entre pas dans la mémoire;
- un rejet accompagné seulement d'un commentaire libre est exclu;
- les textes très longs sont limités dans le paquet remis au modèle;
- les preuves du changement courant priment toujours sur le précédent;
- aucune décision analyste ne modifie automatiquement le modèle de base.

La mémoire produit également un diagnostic de chargement : fichiers examinés,
enregistrements acceptés, rejetés, dupliqués et erreurs de lecture. Cela permet
de savoir si l'adaptation repose réellement sur un corpus exploitable.

## 16. Garde-fous implémentés et limites connues

### 16.1 Invariants de décision

Une décision directe doit comporter une nature, une confiance et une évaluation
de la suffisance des preuves. Une décision confirmée exige :

- un niveau explicite;
- une preuve suffisante;
- une confiance élevée ou moyenne;
- au moins un élément de preuve favorable.

Un statut « à confirmer », une preuve insuffisante, une confiance faible ou un
verdict MINEUR dont l'équivalence est probable, non démontrée, réfutée ou
indéterminée exige une revue. Seule une équivalence confirmée, appuyée par une
preuve suffisante et une confiance adéquate, permet de confirmer MINEUR.

### 16.2 Traçabilité et compatibilité

Le résultat conserve notamment :

- la base de décision directe ou de repli historique;
- le niveau produit par l'ancienne grille;
- les indices mécaniques;
- le statut et la suffisance de preuve;
- le dossier de changements liés;
- le verdict consolidé;
- les deux lectures et leur désaccord éventuel;
- le paquet de précédents utilisé;
- les compteurs de décisions directes, contestées et en désaccord.

Si un ancien artefact ne contient pas la nouvelle évaluation, les valeurs
neutres et la grille historique permettent encore sa lecture. Si un nouveau
résultat fournit simultanément un niveau direct et un niveau historique
explicites, leur incohérence est refusée.

### 16.3 Revue analyste structurée

L'interface permet de corriger le niveau, la pertinence, le statut de nouvelle
idée, une ou plusieurs natures de changement, l'équivalence, les thèmes, la
confiance et la suffisance des preuves. Elle exige également une justification
et au moins une preuve analyste; un contre-argument peut être ajouté. Une
correction incomplète ou contradictoire est refusée. En particulier, la
correction d'un ancien verdict non pertinent vers MODÉRÉ ou MAJEUR doit
comporter au moins un thème, tandis qu'une correction vers non pertinent doit
être MINEUR, sans thème et sans nouvelle idée.

La correction validée remplace la décision automatique effective dans les
badges, les filtres, le périmètre retenu, les synthèses et les exports. Les
dossiers consolidés touchés sont marqués comme à recalculer. La décision
automatique d'origine et l'historique des corrections demeurent conservés dans
l'audit. Une correction encore provisoire reste dans la file « à traiter ».
Seules les corrections suffisamment structurées et finales peuvent ensuite
alimenter la mémoire de précédents.

### 16.4 Limites restantes avant la mise en service

Les principales limites restantes sont opérationnelles plutôt que liées aux
sept mécanismes :

- l'indépendance du contradicteur porte sur ses consignes et son contexte, mais
  pas encore sur le fournisseur ou le modèle utilisé;
- le corpus de référence complet, équilibré et validé par les analystes n'est
  pas encore constitué;
- l'évaluation parallèle n'a donc pas encore démontré les seuils de qualité sur
  les six banques;
- le minimum technique actuel est défini par banque; le corpus métier doit
  encore garantir une représentation suffisante de chaque classe dans chaque
  banque afin d'éviter des taux statistiquement fragiles;
- l'arbitrage métier doit encore confirmer les seuils de passage et la procédure
  de traitement des désaccords persistants.

La branche est ainsi adaptée à une exécution parallèle contrôlée, mais ne doit
pas encore substituer automatiquement ses résultats aux résultats officiels
avant la validation sur corpus.

## 17. Protocole parallèle sur les six banques

### 17.1 Préparation du corpus

Le corpus de référence doit couvrir BMO, BNC, BNS, CIBC, RBC et TD et contenir,
pour chaque cas :

- un identifiant stable;
- la banque;
- le thème AMF;
- la nature du changement;
- le niveau validé par l'analyste;
- le niveau produit par l'ancienne grille;
- le niveau produit par la nouvelle architecture;
- les textes ou preuves de référence et leur provenance.

Le corpus doit être équilibré entre MAJEUR, MODÉRÉ et MINEUR et inclure les cas
difficiles : déplacements, reformulations réellement équivalentes, changements
terminologiques ambigus, gouvernance, capital, réglementation, méthodes,
contrôles, données, tiers, IA et risques émergents.

L'outil d'évaluation ne fabrique pas lui-même la référence analyste. La
constitution et l'arbitrage du corpus demeurent une activité métier.

### 17.2 Exécution et rapport

L'évaluation compare l'ancien et le nouveau système sur exactement les mêmes
cas. Le rapport présente :

- les résultats globaux;
- les résultats par banque, thème et nature;
- les matrices de confusion;
- le rappel des changements non mineurs;
- le rappel propre aux changements MAJEUR;
- les taux MAJEUR vers MINEUR et MODÉRÉ vers MINEUR;
- la précision de MAJEUR;
- la couverture des décisions automatiques et le taux de revue;
- l'accord exact;
- l'accord ordinal pondéré;
- les seuils globaux et les seuils par banque;
- la présence et le volume de chacune des six banques;
- les empreintes distinctes du corpus de référence et des prédictions évaluées.

Le rapport est séparé des résultats de production et sa création refuse
l'écrasement silencieux d'un rapport existant.

### 17.3 Seuils techniques par défaut

Les seuils actuellement configurés sont :

| Mesure | Seuil |
|---|---:|
| Rappel des changements MAJEUR ou MODÉRÉ | au moins 95 % |
| Rappel des changements MAJEUR | au moins 90 % |
| MAJEUR classés MINEUR | au plus 2 % |
| MODÉRÉ classés MINEUR | au plus 5 % |
| Précision de MAJEUR | au moins 75 % |
| Couverture par une décision automatique finale | au moins 80 % |
| Accord ordinal pondéré | au moins 70 % |

Ces seuils sont évalués et affichés globalement ainsi que par banque. Une banque
doit contenir au moins vingt cas par défaut pour être qualifiée. Les seuils ne
bloquent l'exécution que si la porte d'acceptation est explicitement activée.
De même, la couverture qualifiée des six banques ne devient bloquante que
lorsque ce contrôle est demandé.

Les contrôles dont le dénominateur est absent sont signalés comme non évaluables
plutôt que considérés artificiellement réussis ou échoués.

### 17.4 État de validation

L'évaluateur, sa commande d'exécution, les portes optionnelles et leurs tests
techniques sont en place. En revanche :

- aucun corpus complet des six banques n'a encore été arbitré;
- aucune campagne réelle n'a encore été exécutée;
- aucun seuil n'a encore été calibré à partir des résultats métier;
- aucune affirmation de gain de rappel ou de réduction des faux MINEUR ne peut
  encore être faite.

La prochaine décision de déploiement doit donc s'appuyer sur les résultats de
cette campagne, et non seulement sur les exemples ou sur la réussite des tests
unitaires.

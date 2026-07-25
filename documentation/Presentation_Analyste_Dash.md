# Présentation analyste des changements textuels

## Objectif

Cette évolution sépare la sortie technique du pipeline de sa présentation aux
analystes. Les artefacts conservent les identifiants internes nécessaires à
l'audit, tandis que Dash et Excel utilisent une phrase métier stable.

Le format cible est :

> Banque + verbe de changement + contenu métier précis.

Exemples :

- TD ajoute l'incapacité à atteindre les cibles financières parmi les facteurs
  pouvant créer un écart par rapport aux attentes des investisseurs et des
  analystes.
- TD précise que l'incidence de la résolution globale comprend celle de la
  limite imposée à l'actif de la Banque aux États-Unis.
- BMO ajoute la surveillance des risques liés à l'intelligence artificielle à
  ses objectifs de gestion des risques.
- BMO ajoute le renforcement de sa capacité à absorber les périodes de crise à
  son objectif relatif au capital et à la liquidité.

## Principes

### Les périodes restent des métadonnées

Les libellés comme T1, T2 ou T4 restent autorisés dans :

- le sélecteur de période;
- la bannière de comparaison;
- les en-têtes des extraits sources;
- les métadonnées des artefacts.

Ils ne sont plus utilisés comme sujet d'un résumé narratif. Une phrase comme
« Le T2 ajoute… » devient « BMO ajoute… ».

### Une phrase principale par changement

Le résumé visible conserve une seule phrase factuelle. Les explications de
pertinence et les détails produits par l'analyse automatisée restent
disponibles dans un volet secondaire.

### Les preuves ne sont jamais réécrites

Les extraits des rapports restent exacts. Ils sont présentés dans un volet
« Voir la preuve source », avec le rapport courant et le rapport précédent
côte à côte et les différences surlignées.

## Périmètres d'affichage

Dash propose trois périmètres :

1. **Changements qualitatifs**, sélectionné par défaut;
2. **Tous les changements**;
3. **Secondaires / bruit**.

Les raisons suivantes sont classées comme secondaires :

- variation chiffrée propre à la banque;
- reformulation mineure;
- mise à jour de calendrier;
- formatage visuel;
- déplacement de texte;
- opération interne propre à la banque.

Ces éléments ne sont pas supprimés. Ils restent accessibles pour contrôle
humain.

## Structure d'une carte

Une carte est organisée dans l'ordre suivant :

1. nature du changement et badges de priorité;
2. section et pages concernées;
3. phrase métier sous « Changement constaté »;
4. paragraphe distinct sous « Pertinence métier »;
5. preuve source repliée;
6. preuve de posture, si disponible;
7. détails de l'évaluation automatisée repliés;
8. décision et commentaire de l'analyste.

Ce séquencement permet de comprendre le changement avant d'examiner les
preuves et l'interprétation.

Le premier paragraphe décrit uniquement le fait observé. Le second contient
jusqu'à trois phrases complémentaires :

1. la signification métier du changement;
2. les dimensions concrètes qu'il permet de comparer entre les banques;
3. la limite d'interprétation, c'est-à-dire ce que le passage ne permet pas
   encore de conclure.

Pour les nouveaux rapports, le triage produit un constat factuel suivi de ces
trois phrases. La phrase factuelle répétée est retirée automatiquement de la
pertinence métier afin que les deux paragraphes restent complémentaires. Les
anciens artefacts restent compatibles et affichent les phrases utiles déjà
disponibles, sans inventer une limite absente de leur analyse.

Les introductions génériques sont retirées de l'affichage, notamment « Pour
la vigie », « Cette information est importante », « Il convient de noter que »
et « Dans le cadre de cette analyse ». Leur suppression ne retire pas le
contenu métier qui suit.

Exemple :

**Changement constaté**

CIBC ajoute la surveillance des risques liés à l'intelligence artificielle à
ses objectifs de gestion des risques.

**Pertinence métier**

Cet ajout fait passer l'intelligence artificielle d'un enjeu technologique
implicite à une catégorie de risque explicitement reconnue par CIBC. Il permet
de comparer la gouvernance, les responsabilités ainsi que les contrôles sur
les modèles et les données déclarés par les banques. Le passage ne permet
toutefois pas encore de conclure que ces mécanismes sont entièrement mis en
œuvre.

La pertinence métier visible est limitée aux changements qualitatifs. Les
variations chiffrées, reformulations et autres changements secondaires
n'affichent pas ce deuxième paragraphe dans leur vue de contrôle.

## Listes et unités atomiques

Lorsque plusieurs changements proviennent de la même liste, Dash les regroupe
sous leur contexte parent sans les fusionner.

Exemple :

**Bloc de liste analysé — 2 idées modifiées**

Contexte parent : Notre cadre d'appétit pour le risque s'articule autour de
cinq objectifs.

- BMO ajoute la surveillance des risques liés à l'intelligence artificielle.
- BMO ajoute le renforcement de sa capacité à absorber les périodes de crise.

Chaque idée conserve sa propre carte, sa preuve, son identifiant et sa décision
analyste.

## Contrôle de qualité

Le résumé est marqué « Résumé à valider » lorsqu'il ne peut pas satisfaire
automatiquement les règles suivantes :

- présence de la banque comme sujet;
- présence d'un verbe de changement explicite;
- absence des alias internes T1 et T2;
- absence d'une formulation générique décrivant uniquement la structure du
  document.

Dans ce cas, les extraits sources restent disponibles et aucune preuve n'est
inventée.

## Export Excel

La colonne « Ce qui change » utilise le même résumé canonique que Dash. La
colonne « Type de changement » porte déjà l'information Ajout, Suppression ou
Renommage; le résumé n'ajoute donc plus un préfixe comme « Ajout dans le texte
courant ».

Les textes exacts des deux rapports demeurent inchangés dans leurs colonnes de
preuve.

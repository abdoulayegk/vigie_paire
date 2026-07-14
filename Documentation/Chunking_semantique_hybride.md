# Architecture du chunking sémantique hybride

## Statut du document

| Champ | Valeur |
| --- | --- |
| Statut | Document de référence de l’implémentation actuelle |
| Périmètre | Création des chunks narratifs et préparation de leur comparaison |
| Exemple principal | BNC — Rapport annuel 2024 comparé au rapport annuel 2025 |
| Section auditée | `Gestion du capital > Accord de Bâle` |
| Branche d’implémentation | `refactor/semantic-chunking` |
| Commits principaux | `b2a8cb7`, `101a912` |
| Dernière validation | Juillet 2026 |

Ce document décrit le chunking sémantique hybride actuellement implémenté dans le pipeline texte. Il remplace, pour la création des chunks, les règles historiques fondées principalement sur les lignes vides et une taille maximale fixe.

Le document `Decoupage_et_matching_sous_sections.md` reste utile pour comprendre certains principes du matching local, mais ses anciennes règles de fusion par longueur ne décrivent plus le chunker actuel.

---

## 1. Objectif métier

Le chunker transforme le texte narratif d’un rapport bancaire en unités d’idées comparables entre deux périodes.

Une bonne unité doit être :

- complète sur le plan grammatical;
- cohérente sur le plan réglementaire;
- assez précise pour isoler une apparition ou une disparition;
- assez large pour éviter un résultat « une phrase = un changement »;
- fidèle au texte source, sans réécriture par le LLM;
- indépendante des dates, montants et pourcentages propres à une période.

Le résultat attendu n’est pas nécessairement un paragraphe du PDF. Un paragraphe peut contenir plusieurs idées réglementaires et devenir plusieurs chunks. Inversement, plusieurs phrases qui développent la même règle peuvent former un seul chunk.

### Exemple métier recherché

Dans le rapport BNC 2024, le passage suivant doit permettre d’isoler quatre idées :

1. le calcul général de l’actif pondéré;
2. les réformes de Bâle III mises en œuvre;
3. les approches de notation interne;
4. les paramètres de risque et l’approche standardisée.

Si la deuxième idée disparaît en 2025, la comparaison doit produire un retrait précis au lieu de signaler la modification d’un paragraphe de plusieurs centaines de mots.

---

## 2. Ce que le chunking fait et ne fait pas

### Le chunking fait

- conserver tout le texte narratif admis par l’extraction;
- reconnaître les paragraphes, les listes et certains micro-titres;
- découper les paragraphes complexes en phrases complètes;
- mesurer la continuité entre les phrases adjacentes;
- regrouper les phrases qui développent la même idée;
- demander un arbitrage LLM lorsque la frontière est ambiguë;
- appliquer une limite dure de taille à une frontière de phrase;
- produire des objets `TextChunk` minimaux.

### Le chunking ne fait pas

- décider si une différence est pertinente pour l’AMF;
- ignorer un chunk parce qu’il contient une date ou un montant;
- supprimer une acquisition ou une émission d’actions;
- comparer les deux années;
- attribuer un thème AMF final;
- réécrire ou résumer le texte source;
- couper une phrase au milieu pour respecter une taille cible.

La règle centrale est donc :

> Tous les chunks narratifs sont créés avant que la comparaison décide quels changements sont pertinents.

---

## 3. Position dans le pipeline

```text
PDF BNC
  ↓
Extraction Docling et filtrage géométrique
  ↓
Markdown canonique text_extraction_*.md
  ↓
Sections ##
  ↓
Sous-sections ###
  ↓
Chunking structurel
  ↓
Chunking sémantique hybride
  ↓
TextChunk[]
  ↓
Alignement 2024 ↔ 2025
  ↓
Comparaison et filtrage de pertinence
```

Les principales responsabilités sont réparties ainsi :

| Fichier | Responsabilité |
| --- | --- |
| `src/vigilance/text_analysis/chunking.py` | Structure Markdown, listes, micro-titres et création des `TextChunk` |
| `src/vigilance/text_analysis/semantic_chunking.py` | Phrases, embeddings, TF-IDF, scores, LLM et contrôle anti-fragmentation |
| `src/vigilance/text_analysis/comparison.py` | Appel du chunker dans les sous-sections appariées ou orphelines |
| `src/vigilance/text_analysis/chunk_alignment.py` | Alignement un-à-un et un-à-plusieurs entre les chunks des deux périodes |
| `scripts/export_semantic_chunks.py` | Export Markdown d’audit des chunks réels |

---

## 4. Contrat minimal de `TextChunk`

Le modèle reste volontairement minimal :

```text
TextChunk
├── chunk_id
├── kind
├── text
├── subsection_heading
├── hierarchy_path
└── order
```

Exemple :

```text
chunk_id: c03
kind: paragraph
subsection_heading: Accord de Bâle
hierarchy_path: Gestion du capital > Accord de Bâle
order: 3
text: La Banque utilise les approches de notation interne...
```

Les champs suivants ne sont pas ajoutés :

- micro-titre séparé;
- identifiants des blocs PDF;
- position absolue dans le document;
- nombre de mots persisté;
- dates ou montants détectés;
- références réglementaires détectées.

Ces informations ne sont pas nécessaires au contrat de comparaison et auraient alourdi chaque chunk.

### Identifiants locaux

Les identifiants `c00`, `c01`, `c02`, etc. recommencent dans chaque sous-section et chaque période.

Ils représentent l’ordre local, pas une identité métier permanente.

```text
2024 c03 ≠ automatiquement 2025 c03
```

L’alignement doit utiliser le contenu, pas seulement l’identifiant.

---

## 5. Étape structurelle

Avant les embeddings, le système applique la structure du Markdown.

### 5.1 Titres Markdown

Les lignes `##` et `###` ne deviennent pas le texte d’un chunk. Elles sont conservées dans le chemin hiérarchique.

Entrée :

```markdown
### Accord de Bâle

Comme l’exige l’Accord de Bâle, l’actif pondéré...
```

Sortie :

```text
text = Comme l’exige l’Accord de Bâle, l’actif pondéré...
hierarchy_path = Gestion du capital > Accord de Bâle
```

### 5.2 Lignes vides

Une ligne vide produit d’abord une frontière de bloc candidat. Cette frontière structurelle est respectée avant l’analyse sémantique.

Chaque bloc narratif complexe peut ensuite être découpé en plusieurs unités d’idée.

### 5.3 Listes

Les listes adjacentes restent une seule unité. Les éléments ne deviennent pas chacun un chunk.

Les marqueurs reconnus comprennent notamment :

```text
-
*
•
‰

1.
1)
[]
[x]
```

Exemple BNC :

```text
 examiner et approuver la politique de gestion du capital;
 examiner et approuver l’appétit pour le risque;
 examiner et approuver le plan de capital.
```

Résultat : un seul chunk de type `list`, sans les symboles de puce.

### 5.4 Micro-titres

Un libellé court comme `Dividendes` ne doit pas devenir un chunk isolé.

Entrée :

```text
Dividendes

La stratégie en matière de dividende sur les actions ordinaires de la Banque est de cibler un ratio de versement...
```

Résultat :

```text
Dividendes

La stratégie en matière de dividende sur les actions ordinaires de la Banque est de cibler un ratio de versement...
```

Le micro-titre reste dans le texte pour donner le contexte, mais aucun champ supplémentaire n’est ajouté à `TextChunk`.

La détection est prudente :

- cinq mots ou moins;
- présence de lettres;
- première lettre en majuscule;
- absence de chiffre;
- absence de ponctuation finale phrastique;
- paragraphe narratif suivant d’au moins huit mots.

### 5.5 Contenu exclu à ce niveau

Le filtrage principal des tableaux et notes de tableau se fait en amont, avec les métadonnées et les zones géométriques Docling.

Le chunker possède deux protections finales :

- exclusion du marqueur autonome `s.o.`;
- exclusion d’une table Markdown structurellement reconnaissable.

Il ne rejette jamais un paragraphe uniquement parce qu’il contient `%`, `$`, une année ou une référence à un tableau.

---

## 6. Détection des paragraphes complexes

Tous les paragraphes ne nécessitent pas d’embeddings.

Un paragraphe passe au chunking sémantique lorsqu’il contient :

- au moins quatre phrases; ou
- au moins 150 mots et au moins deux phrases.

Sinon, le paragraphe reste intact.

### Exemple simple

```text
Un comité de risque est créé.
```

Cette phrase forme directement un chunk. Aucun embedding ni appel LLM n’est effectué.

### Exemple complexe

```text
La Banque utilise les approches NI...
L’approche NI fondation vise certaines expositions...
L’approche NI avancée vise les autres expositions...
Les paramètres sont assujettis à des limites plancher...
```

Ce bloc possède quatre phrases et passe à l’analyse sémantique.

---

## 7. Segmentation en phrases

Le paragraphe complexe est d’abord séparé en phrases.

La frontière normale est :

```text
ponctuation . ! ?
+ espaces
+ prochaine phrase commençant par une majuscule
```

Exemple :

```text
Phrase 1. Phrase 2. Phrase 3.
```

devient :

```text
1. Phrase 1.
2. Phrase 2.
3. Phrase 3.
```

### Garantie de non-troncature

Toutes les partitions ultérieures utilisent des intervalles de phrases complètes.

Le système ne reçoit jamais l’autorisation de produire :

```text
La Banque utilise les approches de notation
```

à partir de :

```text
La Banque utilise les approches de notation interne pour le risque de crédit.
```

L’audit BNC réel a confirmé :

- 34 chunks en 2024, tous terminés par une ponctuation de fin;
- 32 chunks en 2025, tous terminés par une ponctuation de fin;
- aucun chunk coupé au milieu d’une phrase.

### Limite connue

Une abréviation suivie d’une majuscule pourrait exceptionnellement ressembler à une fin de phrase.

Exemple théorique :

```text
M. Dupont présente le cadre.
```

Ce cas n’a pas été observé dans l’audit BNC actuel. Une évolution future pourrait employer un segmenteur linguistique français spécialisé si le problème apparaît dans d’autres banques.

---

## 8. Normalisation utilisée pour les similarités

Le système conserve deux représentations :

1. le texte source exact, destiné au `TextChunk`;
2. une copie normalisée, destinée uniquement aux embeddings et à TF-IDF.

### Exemple

Texte source :

```text
Au 31 octobre 2025, la Banque maintient un ratio de 13,2 % et un capital de 525 M$.
```

Copie utilisée pour les similarités :

```text
au <nombre> la banque maintient un ratio de <nombre> et un capital de <nombre>
```

Le chunk final conserve toujours :

```text
31 octobre 2025
13,2 %
525 M$
```

### Pourquoi neutraliser les nombres

Le but est d’éviter qu’une variation propre à la banque devienne une frontière d’idée.

```text
2024 : ratio de 12,8 %
2025 : ratio de 13,2 %
```

Ces phrases restent proches sur le plan sémantique malgré la différence numérique.

### Ce qui n’est pas utilisé comme signal de frontière

- année;
- date;
- montant;
- pourcentage;
- acquisition;
- émission d’actions;
- rachat d’actions.

Ces éléments restent néanmoins présents dans le texte source. Leur pertinence est évaluée plus tard pendant la comparaison.

---

## 9. Embeddings

Le modèle par défaut est :

```text
text-embedding-3-small
```

### Stratégie d’appel

- aucun embedding pour les paragraphes simples;
- collecte de toutes les phrases des paragraphes complexes de la sous-section;
- normalisation des phrases;
- déduplication des phrases normalisées identiques;
- un lot logique d’embeddings par sous-section;
- reconstruction des vecteurs dans l’ordre d’origine.

Le client OpenAI découpe techniquement un lot supérieur à 96 entrées en plusieurs requêtes API. Pour l’exemple `Accord de Bâle`, les 75 phrases de 2024 et les 68 phrases de 2025 tiennent chacune dans une requête d’embeddings.

### Validation des embeddings

Le pipeline vérifie :

- le nombre de vecteurs retournés;
- la présence de dimensions;
- la compatibilité des dimensions;
- l’absence de vecteurs nuls.

Une réponse invalide provoque une erreur explicite.

---

## 10. Similarité lexicale TF-IDF

En complément des embeddings, le système calcule une similarité lexicale locale.

Configuration principale :

```text
TfidfVectorizer(ngram_range=(1, 2))
```

Les unigrammes captent les mots importants. Les bigrammes captent des expressions comme :

```text
approche standardisée
fonds propres
risque opérationnel
notation interne
limites plancher
```

La similarité cosine est calculée uniquement entre deux phrases adjacentes du même paragraphe.

TF-IDF est secondaire. Il ne remplace pas les embeddings et ne constitue pas un fallback lorsque l’API échoue.

---

## 11. Score de continuité

Pour chaque frontière potentielle entre deux phrases, le score est :

```text
continuité = 70 % similarité embeddings
            + 30 % similarité lexicale TF-IDF
            + ajustements discursifs
```

### Signaux de continuité

Les expressions suivantes augmentent la probabilité de regroupement :

```text
Cette approche...
Ces exigences...
Selon cette méthode...
Dans cette approche...
Elle permet...
En conséquence...
```

### Signaux de changement d’idée

Les expressions suivantes peuvent réduire la continuité :

```text
Toutefois...
Par ailleurs...
En revanche...
Néanmoins...
Une nouvelle méthode...
La gouvernance...
La politique...
```

Ces signaux ne suffisent pas seuls. Ils ajustent le score embeddings + TF-IDF.

### Seuils

| Score | Décision initiale |
| ---: | --- |
| `≤ 0,72` | frontière probable, donc découpage |
| `≥ 0,84` | continuité forte, donc regroupement |
| `> 0,72` et `< 0,84` | frontière ambiguë, donc arbitrage LLM |

Les seuils ne constituent pas une vérité métier absolue. Ils servent à distinguer les cas évidents des cas qui méritent un arbitrage.

---

## 12. Détection de sur-fragmentation

Le premier audit BNC a révélé que les scores absolus pouvaient produire presque un chunk par phrase.

Résultat initial non acceptable :

| Période | Chunks | Médiane | Chunks d’une phrase |
| --- | ---: | ---: | ---: |
| 2024 | 66 | 31,5 mots | 60 |
| 2025 | 58 | 32 mots | 51 |

Une partition déterministe est maintenant considérée ambiguë lorsqu’elle présente simultanément :

- au moins quatre phrases et quatre groupes;
- un nombre de groupes représentant au moins 75 % du nombre de phrases;
- au moins 65 % de groupes composés d’une seule phrase;
- au moins 65 % de groupes de moins de 50 mots.

Cette règle ne force pas artificiellement la fusion d’une série de longues divulgations autonomes. Elle cible le cas pathologique où le système revient pratiquement à « une phrase = un chunk ».

---

## 13. Arbitrage LLM

Le LLM est utilisé seulement lorsqu’un paragraphe est ambigu :

- un score se trouve entre les deux seuils; ou
- la partition déterministe est sur-fragmentée.

Le modèle par défaut est :

```text
gpt-4o
```

### Entrée du LLM

Le modèle reçoit :

- les phrases numérotées;
- les scores de continuité adjacents;
- les règles métier;
- la cible de 80 à 180 mots pour une idée développée;
- la limite dure de 240 mots;
- l’interdiction de créer une frontière uniquement à cause d’un chiffre ou d’une opération propre à la banque.

### Sortie du LLM

Le modèle ne retourne pas un nouveau texte. Il retourne uniquement des intervalles inclusifs :

```json
{
  "groups": [
    {"start": 1, "end": 1},
    {"start": 2, "end": 2},
    {"start": 3, "end": 5},
    {"start": 6, "end": 10}
  ]
}
```

Le texte final est reconstruit localement à partir des phrases sources.

### Validation de la partition

Les intervalles doivent :

- commencer à la phrase 1;
- être contigus;
- respecter l’ordre;
- ne contenir aucun trou;
- ne contenir aucun chevauchement;
- couvrir exactement la dernière phrase;
- ne jamais dépasser le nombre réel de phrases.

Une réponse invalide provoque une erreur explicite.

### Correction d’une réponse sur-fragmentée

Si le LLM retourne encore presque un groupe par phrase, le système effectue un seul appel correctif explicite.

La consigne corrective demande de :

- regrouper les variantes d’une même méthode;
- regrouper les paramètres et conditions d’un même cadre;
- viser 80 à 180 mots pour une idée développée;
- conserver une phrase courte seule uniquement lorsqu’elle constitue une divulgation réellement indépendante.

Si le deuxième résultat reste sur-fragmenté, le pipeline échoue. Il n’accepte pas silencieusement une mauvaise partition.

---

## 14. Limite dure de 240 mots

Après la partition sémantique, tout groupe de plus de 240 mots est réexaminé.

Le système choisit la frontière de phrase ayant le score de continuité le plus faible à l’intérieur du groupe.

```text
Groupe de 330 mots
  ↓
recherche de la frontière de phrase la plus faible
  ↓
chunk A + chunk B
```

Le système ne coupe pas une phrase longue en deux. Si une seule phrase dépasse 240 mots, elle reste indivisible.

La taille est donc un garde-fou, pas la règle principale. L’unité d’idée reste prioritaire.

---

## 15. Politique « aucun fallback »

L’absence de fallback signifie :

- si un paragraphe complexe exige des embeddings et que l’appel échoue, le pipeline échoue;
- si une frontière ambiguë exige le LLM et que l’appel échoue, le pipeline échoue;
- si la réponse LLM est invalide, le pipeline échoue après la correction autorisée;
- si la réponse reste sur-fragmentée, le pipeline échoue;
- l’ancien découpage par taille n’est jamais utilisé pour masquer une panne.

### Ce qui n’est pas un fallback

Un paragraphe simple ne nécessite légitimement aucun service externe.

```text
Un comité de risque est créé.
```

Le conserver tel quel est le comportement normal, pas un résultat de repli.

TF-IDF reste également une composante normale du score hybride. Il ne remplace jamais les embeddings en cas d’indisponibilité.

---

## 16. Exemple réel BNC — début de `Accord de Bâle`

### 16.1 BNC 2024

#### `c00` — présentation des approches — 69 mots

```text
L’Accord de Bâle propose un éventail d’approches comportant différents degrés de complexité...
Une approche moins complexe, telle que la méthode standardisée, utilise des pondérations réglementaires...
```

#### `c01` — calcul de l’actif pondéré — 23 mots

```text
Comme l’exige l’Accord de Bâle, l’actif pondéré en fonction des risques est calculé pour chacun des risques de crédit, de marché et opérationnel.
```

Le chunk reste court parce qu’il constitue une divulgation complète et autonome.

#### `c02` — réformes du BSIF — 86 mots

```text
Certaines révisions apportées par le BSIF à ses règles de fonds propres, de levier, de liquidité et de communication de renseignements dans le cadre des réformes de Bâle III ont pris effet...
```

#### `c03` — approches NI et paramètres — 154 mots

Ce chunk regroupe :

- l’utilisation des approches NI;
- l’approche NI fondation;
- l’approche NI avancée;
- les paramètres PD, PCD et ECD;
- les limites plancher.

#### `c04` — approche standardisée — 43 mots

```text
Le risque de crédit de certains portefeuilles... est pondéré conformément à l’approche standardisée révisée...
L’exposition aux titres de participation... est également pondérée selon cette approche.
```

### 16.2 BNC 2025

Le contenu commun est réparti ainsi :

| Chunk | Idée | Taille |
| --- | --- | ---: |
| `c00` | Présentation des approches | 69 mots |
| `c01` | Calcul de l’actif pondéré | 23 mots |
| `c02` | Approches NI | 77 mots |
| `c03` | Paramètres et limites plancher | 77 mots |
| `c04` | Approche standardisée | 43 mots |

Le chunk 2024 sur les réformes du BSIF n’existe plus.

### 16.3 Différence de frontière acceptable

La frontière n’est pas exactement identique :

```text
2024 c03 = approches NI + paramètres
2025 c02 = approches NI
2025 c03 = paramètres
```

Ce n’est pas une troncature. Toutes les phrases sont complètes. C’est une différence de granularité sémantique.

L’aligneur la résout par un appariement un-à-plusieurs :

```text
2024 c03 ↔ 2025 c02+c03    matched_grouped
2024 c04 ↔ 2025 c04        matched_strong
2024 c02 ↔ aucun chunk     possible_removed
```

Le retrait des réformes est ainsi isolé sans créer un faux gros changement sur les approches NI.

---

## 17. Résultats de l’audit réel BNC

Après la correction anti-fragmentation :

| Mesure | 2024 | 2025 |
| --- | ---: | ---: |
| Nombre de phrases source | 75 | 68 |
| Nombre de chunks | 34 | 32 |
| Taille minimale | 12 mots | 12 mots |
| Taille médiane | 71 mots | 71,5 mots |
| Taille moyenne | 75,4 mots | 71,6 mots |
| Taille maximale | 156 mots | 156 mots |
| Chunks de moins de 40 mots | 9 | 9 |
| Chunks d’une phrase | 12 | 9 |
| Chunks sans ponctuation finale | 0 | 0 |

Les chunks courts restants correspondent principalement à des divulgations autonomes : définition, responsabilité réglementaire, règle de calcul ou état de conformité.

---

## 18. Interaction avec l’alignement

Le chunking est exécuté indépendamment pour chaque période. Les identifiants et certaines frontières peuvent donc varier.

L’alignement utilise ensuite :

- similarité lexicale;
- embeddings;
- proximité d’ordre;
- candidats forts et faibles;
- regroupement de chunks adjacents lorsque le contenu complet est quasi équivalent.

Types de résultats principaux :

| Type | Signification |
| --- | --- |
| `matched_strong` | correspondance forte un-à-un |
| `matched_weak` | correspondance probable nécessitant plus d’analyse |
| `matched_grouped` | un chunk correspond à plusieurs chunks adjacents |
| `possible_added` | aucun équivalent précédent suffisamment solide |
| `possible_removed` | aucun équivalent courant suffisamment solide |

L’alignement groupé est essentiel lorsqu’une même idée est répartie différemment entre deux années.

---

## 19. Ce qui sera ignoré plus tard pendant la comparaison

Le chunking conserve tous les contenus narratifs. La comparaison peut ensuite décider qu’une différence n’est pas pertinente si elle porte uniquement sur :

- une année;
- une date;
- un chiffre;
- un montant;
- un ratio ou pourcentage;
- une acquisition propre à la banque;
- une émission ou un rachat d’actions;
- une opération sans changement de règle, méthode, risque, politique ou gouvernance.

Exemple :

```text
2024 : la valeur des passifs admissibles s’élève à 23,5 G$.
2025 : la valeur des passifs admissibles s’élève à 26,1 G$.
```

Les deux phrases deviennent des chunks. C’est seulement pendant la comparaison que la variation purement quantitative pourra être ignorée.

---

## 20. Export et audit des chunks

L’exporteur ne modifie pas `TextChunk`. Il écrit une vue Markdown lisible.

### Audit d’une sous-section BNC 2024

```bash
UV_CACHE_DIR=/tmp/vigie-uv-cache uv run python scripts/export_semantic_chunks.py \
  --markdown outputs/resultats/bnc/2025_t4_vs_2024_t4/text_extraction_2024_t4.md \
  --output outputs/resultats/bnc/2025_t4_vs_2024_t4/chunks_2024_t4_accord_bale.md \
  --section-key gestion_capital \
  --subsection "Accord de Bâle"
```

### Audit d’une sous-section BNC 2025

```bash
UV_CACHE_DIR=/tmp/vigie-uv-cache uv run python scripts/export_semantic_chunks.py \
  --markdown outputs/resultats/bnc/2025_t4_vs_2024_t4/text_extraction_2025_t4.md \
  --output outputs/resultats/bnc/2025_t4_vs_2024_t4/chunks_2025_t4_accord_bale.md \
  --section-key gestion_capital \
  --subsection "Accord de Bâle"
```

### Audit de toute la section `Gestion du capital`

Il suffit d’omettre `--subsection` :

```bash
UV_CACHE_DIR=/tmp/vigie-uv-cache uv run python scripts/export_semantic_chunks.py \
  --markdown outputs/resultats/bnc/2025_t4_vs_2024_t4/text_extraction_2025_t4.md \
  --output outputs/resultats/bnc/2025_t4_vs_2024_t4/chunks_2025_t4_gestion_capital.md \
  --section-key gestion_capital
```

### Format produit

```markdown
[c03 | paragraph | Gestion du capital > Accord de Bâle]

La Banque utilise les approches de notation interne...
```

Les audits sont des artefacts de contrôle. Ils ne sont pas nécessaires au contrat de production et ne doivent pas être confondus avec le Markdown canonique d’extraction.

---

## 21. Checklist d’acceptation

Pour accepter un audit de chunking :

- [ ] aucun tableau ou note de tableau ne réapparaît;
- [ ] aucun chunk autonome `s.o.` n’est présent;
- [ ] aucun micro-titre narratif ne devient un chunk d’un mot;
- [ ] aucune phrase n’est coupée au milieu;
- [ ] chaque phrase source apparaît exactement une fois;
- [ ] aucun chunk ne dépasse 240 mots, sauf phrase indivisible;
- [ ] les méthodes et variantes d’une même règle sont regroupées;
- [ ] les changements réglementaires autonomes peuvent être isolés;
- [ ] les frontières communes restent raisonnablement stables entre périodes;
- [ ] les différences de granularité sont absorbables par `matched_grouped`;
- [ ] les dates, montants et pourcentages sont toujours présents dans le texte source;
- [ ] un échec embeddings ou LLM provoque une erreur explicite.

---

## 22. Tests automatisés

Les tests couvrent notamment :

- absence d’embeddings pour un paragraphe simple;
- erreur explicite lorsque les embeddings échouent;
- erreur explicite lorsque le LLM échoue;
- rejet d’une partition incomplète;
- un seul lot logique d’embeddings par sous-section;
- déduplication des phrases identiques;
- conservation des nombres dans le chunk;
- neutralisation des nombres pour les similarités;
- reconnaissance des puces Docling ``;
- limite dure de 240 mots;
- détection de sur-fragmentation déterministe;
- correction unique d’une réponse LLM sur-fragmentée;
- erreur si la correction reste sur-fragmentée;
- scénario BNC où le chunk de réforme existe seulement en 2024.

État lors de la validation :

```text
890 tests réussis
20 tests ignorés
Ruff réussi
pre-commit réussi
```

---

## 23. Erreurs et diagnostic

### Client absent pour un paragraphe complexe

```text
Un client OpenAI est requis pour découper les paragraphes complexes; aucun fallback n’est autorisé.
```

Cause : appel direct au chunker sans client alors que le passage exige les embeddings.

### Échec embeddings

```text
Échec des embeddings sans fallback: ...
```

Le pipeline s’arrête. Il ne revient pas à l’ancien découpage.

### Échec LLM

```text
Échec du partitionnement LLM sans fallback: ...
```

Le passage était ambigu et aucun résultat valide n’a pu être obtenu.

### Partition invalide

```text
Partition LLM invalide: les groupes doivent couvrir toutes les phrases, dans l’ordre, sans trou ni chevauchement.
```

Le modèle a produit des intervalles incohérents.

### Sur-fragmentation persistante

```text
Partition LLM toujours sur-fragmentée après correction; aucun fallback n’est autorisé.
```

Le résultat est volontairement rejeté plutôt que transmis à la comparaison.

---

## 24. Limites connues et évolutions possibles

### Frontières variables entre deux périodes

Deux textes presque identiques peuvent parfois être regroupés différemment. L’alignement un-à-plusieurs limite l’impact, mais la stabilité des frontières reste un axe d’amélioration.

Évolution possible : ajouter une validation de cohérence inter-périodes après création indépendante des chunks, sans modifier leur texte.

### Segmentation française

La séparation actuelle par ponctuation est simple et auditable. Un tokenizer français spécialisé pourrait mieux gérer les abréviations, au prix d’une dépendance supplémentaire.

### Coût des grandes sous-sections

Une sous-section comme `Accord de Bâle` contient plusieurs dizaines de phrases. L’embedding reste économique grâce au lot unique, mais une ambiguïté peut déclencher plusieurs appels LLM, un par paragraphe complexe.

### Données transmises aux services externes

L’utilisation des modèles OpenAI envoie les phrases normalisées aux embeddings et les phrases sources des paragraphes ambigus au LLM. L’exploitation doit respecter les règles de confidentialité et les autorisations de l’organisation.

### Qualité de l’extraction en amont

Le chunker ne peut pas reconstruire parfaitement un texte déjà mélangé à un tableau ou mal ordonné par l’extracteur. La qualité du Markdown canonique reste une condition d’entrée essentielle.

---

## 25. Résumé opérationnel

```text
1. Lire une sous-section Markdown.
2. Retirer les titres du texte et conserver leur hiérarchie.
3. Regrouper les listes et rattacher les micro-titres.
4. Conserver directement les paragraphes simples.
5. Séparer les paragraphes complexes en phrases complètes.
6. Neutraliser les nombres uniquement pour les similarités.
7. Encoder les phrases uniques en un lot d’embeddings.
8. Calculer embeddings 70 % + TF-IDF 30 % + signaux discursifs.
9. Détecter les frontières évidentes et la sur-fragmentation.
10. Demander au LLM des intervalles pour les cas ambigus.
11. Valider couverture, ordre, absence de trous et chevauchements.
12. Corriger une seule fois une partition LLM sur-fragmentée.
13. Appliquer la limite de 240 mots aux frontières de phrases.
14. Construire les TextChunk sans modifier le texte source.
15. Aligner ensuite les chunks 2024 et 2025.
16. Filtrer seulement plus tard les changements non pertinents.
```

Le principe final est :

> La structure propose les blocs, les embeddings mesurent la continuité, TF-IDF renforce les indices lexicaux, le LLM arbitre seulement l’ambiguïté, et aucune panne n’est masquée par un fallback.

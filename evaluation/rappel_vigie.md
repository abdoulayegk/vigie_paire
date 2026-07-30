# Etage de recherche lexical (candidats)

Paire: `2025_t4_vs_2024_t4` — seuil de rapprochement lexical: 0.45

> Ce rapport n'est pas la mesure de rappel. Le rapprochement purement lexical
> accepte des candidats qui partagent du vocabulaire bancaire generique sans
> decrire le meme changement, ce qui surestime le rappel d'une dizaine de
> points. Son role est de proposer les candidats a l'adjudication.
> La mesure de reference est `evaluation/rappel_vigie_juge.md`
> (`scripts/eval_vigie_judge.py`).

| Banque | Items manuels | Evalues | Retrouves | Rappel | via texte retenu | via texte filtre | via tableaux |
|---|---:|---:|---:|---:|---:|---:|---:|
| BMO | 57 | 57 | 56 | 98% | 44 | 12 | 0 |
| CIBC | 62 | 58 | 58 | 100% | 36 | 14 | 8 |
| TD | 27 | 26 | 23 | 88% | 16 | 6 | 1 |
| BNS / Scotia | 28 | 28 | 24 | 86% | 17 | 6 | 1 |
| RBC | 41 | 38 | 36 | 95% | 21 | 13 | 2 |

## Volume produit par le pipeline

| Banque | Changements texte retenus | Changements texte filtres | Paires de tables |
|---|---:|---:|---:|
| BMO | 335 | 245 | 37 |
| CIBC | 269 | 215 | 41 |
| TD | 346 | 154 | 27 |
| BNS / Scotia | 327 | 202 | 36 |
| RBC | 252 | 184 | 44 |

## BMO — items manuels non retrouves (1)

### bmo-001 — p.151 — Cadre

> De maniere general, les ternes "groupes d'exploitation" et suffisance du capital" sont respectivement remplaces par "unites d'exploitation" et "adequation des fonds propres

Meilleur score: 0.436

- `0.436` [texte_filtre] Activités de gestion du capital (p.[66, 67], modified) — Activités de gestion du capital Le secteur des services financiers est fortement réglementé et BMO fait face à des exigences et à des attentes réglementaires de plus en plus complexes, les pouvoirs pu
- `0.417` [texte_retenu] Activités de gestion du capital (p.[66], added) — Activités de gestion du capital Le cadre aide la haute direction et le Conseil d'administration à évaluer le profil de risque de la Banque par rapport à notre appétit pour le risque. Le cadre et l'éno
- `0.403` [texte_filtre] Exigences en matière de fonds propres réglementaires (p.[61, 62], modified) — Exigences en matière de fonds propres réglementaires Les réserves au titre du deuxième pilier couvrent les risques associés aux vulnérabilités systémiques et comprennent la réserve pour stabilité inté
- `0.384` [texte_filtre] Demande de capital (p.[60, 61], modified) — Demande de capital Les principes et les éléments clés de notre cadre de gestion du capital sont exposés dans notre politique générale de gestion du capital et dans le plan de capital annuel, qui intèg
- `0.384` [texte_filtre] Exigences en matière de fonds propres réglementaires (p.[61, 62], modified) — Exigences en matière de fonds propres réglementaires La TLAC comprend le total des instruments de fonds propres et des autres instruments de TLAC pouvant être convertis, en tout ou en partie, en actio


## CIBC — items manuels non retrouves (0)

Aucun.

## TD — items manuels non retrouves (3)

### td-022 — p.121 — Risque de liquidite

> Retrait des textes explicatifs du ratio LCR et les ratios reglementaires

Meilleur score: 0.631

- `0.631` [texte_retenu] MODE DE GESTION DU RISQUE DE LIQUIDITÉ DE LA TD (p.[114, 115], modified) — MODE DE GESTION DU RISQUE DE LIQUIDITÉ DE LA TD Les mesures internes servent de complément aux exigences de liquidité réglementaires, elles comprennent le ratio de liquidité à court terme (LCR), le ra
- `0.626` [texte_filtre] RATIO DE LIQUIDITÉ À COURT TERME (p.[86, 105], modified) — RATIO DE LIQUIDITÉ À COURT TERME Le LCR moyen de la Banque de 138 % pour le trimestre clos le 31 octobre 2024 continue à satisfaire aux exigences réglementaires. Le LCR moyen de la Banque était de 130
- `0.619` [texte_retenu] APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD (p.[86], removed) — APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD D'après la ligne directrice Normes de liquidité du BSIF, les banques canadiennes doivent maintenir un ratio de liquidité à court terme (LCR) de 100 % ou pl
- `0.57` [texte_retenu] RATIO DE LIQUIDITÉ À COURT TERME (p.[118], added) — RATIO DE LIQUIDITÉ À COURT TERME Le LCR a diminué tout au long du trimestre alors que la Banque a continué de se concentrer à dégager des excédents importants de la vente de sa participation en action
- `0.499` [texte_retenu] RATIO DE LIQUIDITÉ À LONG TERME → RATIO DE LIQUIDITÉ À COURT TERME (p.[86], removed) — RATIO DE LIQUIDITÉ À LONG TERME → RATIO DE LIQUIDITÉ À COURT TERME Les HQLA comme présentés de la Banque ne tiennent pas compte des HQLA excédentaires des Services de détail aux États-Unis, conforméme

### td-024 — p.122 — Risque de liquidite

> Tableau Ratio LCR: Ajout d'une note de bas de tableau pour expliquer que la cellule est sans donnee selon le gabarit de divulgation

Meilleur score: 0.556

- `0.556` [texte_filtre] RATIO DE LIQUIDITÉ À COURT TERME (p.[86, 105], modified) — RATIO DE LIQUIDITÉ À COURT TERME 1 Le LCR est calculé conformément à la ligne directrice Normes de liquidité du BSIF, qui tient compte des exigences en matière de liquidité publiées par le CBCB. Le LC
- `0.534` [texte_retenu] APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD (p.[86], removed) — APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD D'après la ligne directrice Normes de liquidité du BSIF, les banques canadiennes doivent maintenir un ratio de liquidité à court terme (LCR) de 100 % ou pl
- `0.514` [texte_filtre] Gestion des données (p.[111, 112], modified) — Gestion des données La Banque gère le risque lié aux données au moyen du cadre de gestion des risques liés aux données, lequel décrit la gouvernance, les politiques et les processus auxquels les secte
- `0.494` [texte_retenu] RATIO DE LIQUIDITÉ À COURT TERME (p.[118], added) — RATIO DE LIQUIDITÉ À COURT TERME Le LCR a diminué tout au long du trimestre alors que la Banque a continué de se concentrer à dégager des excédents importants de la vente de sa participation en action
- `0.493` [texte_retenu] Gestion des données (p.[111, 112], modified) — Gestion des données Les actifs informationnels de la Banque sont traités et gérés de façon à conserver leur valeur et à appuyer les objectifs d'affaires. Des pratiques irrégulières ou inadéquates en m

### td-025 — p.122 — Risque de liquidite

> Ajout d'un texte pour enumerer les consultations lancees en 2025 par le BSIF.

Meilleur score: 0.466

- `0.466` [texte_retenu] APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD (p.[86], removed) — APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD D'après la ligne directrice Normes de liquidité du BSIF, les banques canadiennes doivent maintenir un ratio de liquidité à court terme (LCR) de 100 % ou pl
- `0.423` [texte_retenu] MODE DE GESTION DU RISQUE DE LIQUIDITÉ DE LA TD (p.[114, 115], modified) — MODE DE GESTION DU RISQUE DE LIQUIDITÉ DE LA TD Les mesures internes servent de complément aux exigences de liquidité réglementaires, elles comprennent le ratio de liquidité à court terme (LCR), le ra
- `0.398` [texte_filtre] APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD (p.[114], modified) — APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD La TD met en œuvre un programme de gestion de la liquidité rigoureux, lequel est assujetti à la gouvernance et à la surveillance des risques et est conçu a
- `0.379` [texte_filtre] Faits nouveaux des organismes de réglementation et des instances de normalisation concernant le risque environnemental et social → FAITS NOUVEAUX RÉGLEMENTAIRES CONCERNANT LA LIQUIDITÉ ET LE FINANCEMENT (p.[125], added) — Faits nouveaux des organismes de réglementation et des instances de normalisation concernant le risque environnemental et social → FAITS NOUVEAUX RÉGLEMENTAIRES CONCERNANT LA LIQUIDITÉ ET LE FINANCEME
- `0.371` [texte_retenu] FAITS NOUVEAUX RÉGLEMENTAIRES CONCERNANT LA LIQUIDITÉ ET LE FINANCEMENT (p.[125], added) — FAITS NOUVEAUX RÉGLEMENTAIRES CONCERNANT LA LIQUIDITÉ ET LE FINANCEMENT Également en mai, le BSIF a lancé une consultation publique auprès des institutions sur le processus d'examen de surveillance au


## BNS / Scotia — items manuels non retrouves (4)

### bns-001 — p.99 — Reserve pour stabilite interieure

> Mise a jour du texte sur la reserve pour stabilite interieure (RSI) de 3,5% pour mentionner que le taux a ete maintenu en juin 2024.

Meilleur score: 0.317

- `0.317` [texte_filtre] Fonds propres réglementaires (p.[58, 63], modified) — Fonds propres réglementaires En juin 2023, le BSIF a annoncé que la réserve pour stabilité intérieure (« RSI ») serait portée à 3,5 % de la valeur totale des actifs pondérés en fonction des risques à 
- `0.117` [tableaux] T33 Actions et autres instruments (p.[64, 69], modified) — T32 Actions et autres instruments T33 Actions et autres instruments Actions et autres instruments de capitaux propres Actions et autres instruments de capitaux propres Billets avec remboursement de ca
- `0.106` [tableaux] T50 Sensibilité aux taux d’intérêt structurels* (p.[96, 101], modified) — T50 Sensibilité aux taux d’intérêt structurels T50 Sensibilité aux taux d’intérêt structurels* Sensibilité aux taux d’intérêt structurels Sensibilité aux taux d’intérêt structurels inchange Aucun chan
- `0.104` [texte_filtre] Information sur les actions et les autres instruments de capitaux propres (p.[64, 69], modified) — Information sur les actions et les autres instruments de capitaux propres 2) Les dividendes sur les actions ordinaires sont versés sur une base trimestrielle, lorsqu'ils seront déclarés. Au 22 novembr
- `0.093` [texte_retenu] Fonds propres réglementaires (p.[63], added) — Fonds propres réglementaires Les exigences du BSIF en matière de ratios de fonds propres réglementaires minimaux, y compris le supplément de 1,0 % s'appliquant aux BIS i  et la RSI, s'établissent à 11

### bns-020 — p.107 — Risque de marche

> Ajout que depuis le 1er novembre la VaR comprend la volatilite de l'ecart de credit propre aux emetteurs, qui etait auparavant dans la VaR propre a la dette.

Meilleur score: 0.549

- `0.549` [texte_retenu] Valeur à risque (VàR) → Sommaire des mesures du risque (p.[87], modified) — Valeur à risque (VàR) → Sommaire des mesures du risque La VàR a deux composantes, à savoir le risque de marché général et le risque propre à la dette. Pour ce qui est des instruments de créance et de 
- `0.426` [texte_filtre] Gouvernance du risque de marché (p.[94, 99], modified) — Gouvernance du risque de marché La Banque fait appel à un certain nombre de mesures et de modèles pour mesurer et contrôler le risque de marché. Ces mesures sont choisies en fonction d'une évaluation 
- `0.425` [tableaux] T51 Mesure du risque de marché* (p.[97, 102], modified) — T51 Mesure du risque de marché T51 Mesure du risque de marché* Mesure du risque de marché Mesure du risque de marché 1 Depuis le 1er novembre 2024, la VaR au titre de l’écart de crédit comprend égalem
- `0.398` [texte_retenu] Gouvernance du risque de marché (p.[94, 99], modified) — Gouvernance du risque de marché Le groupe Gestion du risque global supervise indépendamment tout risque de marché important, soutenant le CGRMAP et le CGAP à l'aide d'analyses, d'évaluations du risque
- `0.303` [texte_retenu] Qualité du crédit (p.[87, 93], modified) — Qualité du crédit politiques ayant une incidence sur le marché jusqu'à la date des états financiers. La Banque a recours au jugement d'experts en matière de crédit dans l'appréciation de la détériorat

### bns-023 — p.107 — Risque de marche

> Retrait de la note de bas de page en lien avec l'adoption de l'IFRS 17.

Meilleur score: 0.447

- `0.447` [texte_retenu] Gouvernance du risque de marché (p.[94, 99], modified) — Gouvernance du risque de marché Le groupe Gestion du risque global supervise indépendamment tout risque de marché important, soutenant le CGRMAP et le CGAP à l'aide d'analyses, d'évaluations du risque
- `0.447` [texte_filtre] Gouvernance du risque de marché (p.[94, 99], modified) — Gouvernance du risque de marché La Banque fait appel à un certain nombre de mesures et de modèles pour mesurer et contrôler le risque de marché. Ces mesures sont choisies en fonction d'une évaluation 
- `0.401` [texte_retenu] Adoption rapide de l'IA (p.[89], added) — Adoption rapide de l'IA L'adoption de l'IA peut donner lieu à des risques tels que des perturbations opérationnelles, à des failles de sécurité, notamment un risque accru de fraude, à des défis réglem
- `0.401` [texte_retenu] Adoption rapide de l'IA (p.[89], added) — Adoption rapide de l'IA Afin de répondre à la vigilance réglementaire croissante liée à l'utilisation de l'IA et aux règles potentiellement incohérentes en matière d'IA entre les différents territoire
- `0.342` [tableaux] T52 Interdépendance du risque de marché et de l’état consolidé de la situation financière de la Banque (p.[99, 104], modified) — T52 Interdépendance du risque de marché et de l’état consolidé de la situation financière de la Banque T52 Interdépendance du risque de marché et de l’état consolidé de la situation financière de la B

### bns-028 — p.108 — Principaux risques non financiers

> Dans la section des risques ESG, retrait de la mention que la banque est membre de l'alliance Net Zero (NZBA)

Meilleur score: 0.448

- `0.448` [texte_filtre] Principaux types de risques (p.[74, 79], modified) — Principaux types de risques Les risques que la direction estime d'importance primordiale i) qui ont une incidence ou une influence importante sur les principales activités de la Banque et sur ses acti
- `0.438` [texte_filtre] Principaux types de risques (p.[74, 79], modified) — Principaux types de risques Les principaux risques sont classés dans l'un ou l'autre des deux grands groupes suivants : Les principaux risques sont classés dans l'un ou l'autre des deux grands groupes
- `0.301` [texte_retenu] Principaux types de risques → Exigences en matière de surveillance du risque (p.[74, 82], modified) — Principaux types de risques → Exigences en matière de surveillance du risque l'établissement des structures de gouvernance des comités pour gérer le risque;
l'affectation de ressources dédiées à la de
- `0.255` [texte_filtre] Rôle de la direction (p.[87], removed) — Rôle de la direction La Banque a pris, à l'échelle de son entreprise, divers engagements afin de faire face aux risques et aux possibilités liés au climat à court, à moyen et à long termes. La Banque 
- `0.232` [texte_filtre] Limites → Limites de risque (p.[78, 85], modified) — Limites → Limites de risque Les limites régissent et circonscrivent les activités impliquant une prise de risques en fonction du seuil d'appétence établi par le conseil d'administration et les membres


## RBC — items manuels non retrouves (2)

### rbc-028 — p.74 — Architecture des documents de gestion du risque d'entreprise

> Modifications du titre du graphique, passant de «Architecture de gestion du risque d'entreprise» a «Architecture des documents de gestion du risque d'entreprise»

Meilleur score: 0.437

- `0.437` [texte_retenu] Stratégie de gestion des risques (p.[118], added) — Stratégie de gestion des risques Notre programme axé sur les crimes financiers de l'entreprise a pour mission d'élaborer et de tenir à jour des politiques, des lignes directrices, des formations ainsi
- `0.437` [texte_filtre] Gestion du risque (p.[122, 126], modified) — Gestion du risque Nous avons à gérer les risques inhérents au secteur des services financiers, notre objectif à cet égard étant de générer une valeur maximale pour nos actionnaires, nos clients, nos e
- `0.367` [texte_retenu] Stratégie de gestion des risques (p.[118], added) — Stratégie de gestion des risques Risque lié aux technologies de l'information et à la cybersécurité Passage de sous-section ajouté: Stratégie de gestion des risques
- `0.366` [texte_retenu] Principes de gestion des risques (p.[75], removed) — Principes de gestion des risques Les réformes financières et autres qui sont ou seront adoptées dans plusieurs territoires, telles que la réglementation sur le numérique, sur les données et la technol
- `0.366` [texte_retenu] Principes de gestion des risques (p.[75, 76], modified) — Principes de gestion des risques Parvenir à un juste équilibre risque-rendement de sorte à permettre une croissance durable;
Partager collectivement la responsabilité de la gestion des risques;
Ne pre

### rbc-029 — p.74 — Architecture des documents de gestion du risque d'entreprise

> Retrait des exemples de politiques de gestion des risques specifiques a l'echelle de l'entreprise

Meilleur score: 0.417

- `0.417` [texte_retenu] Stratégie de gestion des risques (p.[118], added) — Stratégie de gestion des risques Notre programme axé sur les crimes financiers de l'entreprise a pour mission d'élaborer et de tenir à jour des politiques, des lignes directrices, des formations ainsi
- `0.404` [texte_filtre] Gestion du risque (p.[122, 126], modified) — Gestion du risque Nous avons à gérer les risques inhérents au secteur des services financiers, notre objectif à cet égard étant de générer une valeur maximale pour nos actionnaires, nos clients, nos e
- `0.393` [texte_retenu] Principes de gestion des risques (p.[75, 76], modified) — Principes de gestion des risques La nature dynamique du secteur des services financiers ainsi que l'innovation technologique exigent de perfectionner sans cesse nos processus, outils et pratiques et d
- `0.383` [texte_retenu] Principes de gestion des risques (p.[75, 76], modified) — Principes de gestion des risques Notre organigramme et nos processus de gouvernance sont structurés de manière à assurer que la Gestion des risques du Groupe et la Conformité à la réglementation demeu
- `0.381` [texte_retenu] Contrôle des risques (p.[79], added) — Contrôle des risques Le cadre de gestion du risque d'entreprise, le cadre d'appétit pour le risque d'entreprise et le cadre de gestion du risque lié à la conduite et à culture d'entreprise, conjointem

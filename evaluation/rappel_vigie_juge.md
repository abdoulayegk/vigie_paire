# Rappel de la vigie automatique — adjudication LLM

Paire: `2025_t4_vs_2024_t4` · juge: `gpt-4o` · 12 candidats proposes par item

Lecture: « couvert » = le pipeline a produit un changement qui porte sur le meme
passage et le meme sens que l'observation de l'analyste. « texte filtre » = le
changement a bien ete detecte mais ecarte avant l'export analyste.

| Banque | Items | Couverts | Rappel | dont haute conf. | via texte retenu | via texte filtre | via tableaux |
|---|---:|---:|---:|---:|---:|---:|---:|
| BMO | 57 | 52 | 91% | 46 | 38 | 14 | 0 |
| CIBC | 58 | 51 | 88% | 33 | 29 | 6 | 16 |
| TD | 26 | 24 | 92% | 21 | 20 | 0 | 4 |
| BNS / Scotia | 28 | 26 | 93% | 22 | 20 | 3 | 3 |
| RBC | 38 | 31 | 82% | 26 | 22 | 5 | 4 |
| BNC | 39 | 37 | 95% | 26 | 32 | 5 | 0 |

**Rappel de detection: 221/246 = 90%**

**Rappel effectivement livre a l'analyste: 188/246 = 76%** — l'ecart de 33 items correspond a des changements detectes puis ecartes par le triage de pertinence (`genai_triage.is_relevant = false`), donc absents de l'export.

## Volume produit par le pipeline

| Banque | Texte retenu | Texte filtre | Paires de tables |
|---|---:|---:|---:|
| BMO | 335 | 245 | 37 |
| CIBC | 269 | 215 | 41 |
| TD | 346 | 154 | 27 |
| BNS / Scotia | 327 | 202 | 36 |
| RBC | 252 | 184 | 44 |
| BNC | 126 | 81 | 28 |

## BMO — couverts (52)

| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |
|---|---:|---|---|---|---|---|---|
| bmo-001 | 151 | Cadre | De maniere general, les ternes "groupes d'exploitation" et suffisance du capital" sont respectivement remplaces par "unites d'exploitation" et "adequa | texte_filtre | Demande de capital | modified | haute |
| bmo-003 | 151 | Exigences en matiere de fonds propres regleme | Ajout d'un texte sur l'entree en vigueur du cadre de capacite totale d'absorption des pertes par etablissement des societes meres de BISN | texte_filtre | Évolution des exigences en matière de fonds p | modified | haute |
| bmo-005 | 152 | Ratios de fonds propres reglementaires et de  | Ajout des termes "moins les deductions reglementaires". Auparavant etait "net des deductions" | texte_filtre | Ratios de fonds propres réglementaires et de  | modified | haute |
| bmo-006 | 152 | Evolution des exigences en matiere de fonds p | Publication le 29 octobre 2025 de la mise a jour (ancienne publication du 20 fevrier 2025) de la ligne directrice Regime au regard des normes de fonds | texte_retenu | Évolution des exigences en matière de fonds p | added | haute |
| bmo-007 | 152 | Evolution des exigences en matiere de fonds p | Publication de la version revisee de le ligne directrice NFP le 11 septembre 2025 | texte_retenu | Évolution des exigences en matière de fonds p | added | haute |
| bmo-008 | 153 | Evolution des exigences en matiere de fonds p | Capital economique et actifs ponderes en fonction des risques par unite d'exploitation et type de risque | texte_filtre | Capital économique et actifs pondérés en fonc | renamed | haute |
| bmo-009 | 153 | Evolution des exigences en matiere de fonds p | Modification de libelles dans le tableau "Actifs ponderes en fonction des risques" devient "Actifs ponderes par type de risque" et "Services bancaires | texte_filtre | Capital économique et actifs pondérés en fonc | modified | haute |
| bmo-010 | 155 | Introduction | Modification du texte sur l'objectif du cadre de gestion des risques. Le "Cadre de gestion globale des risques" devient le "Cadre de gestion des risqu | texte_retenu | Cadre de gestion globale des risques → Cadre  | removed | haute |
| bmo-011 | 155 | Principaux risques et risques emergents pouva | Mise a jour de la Situation economique generale avec un focus sur les tarifs douaniers americains et l'inflation | texte_retenu | Situation économique générale | added | moyenne |
| bmo-012 | 156 | Principaux risques et risques emergents pouva | Le theme de la Montee des differends commerciaux et evolution des risques geopolitiques est replace en seconde position des principaux risques. Mise a | texte_retenu | Risques géopolitiques et montée des différend | added | haute |
| bmo-013 | 157 | Principaux risques et risques emergents pouva | Mise a jour du texte sur le Risque lie a la cybersecurite et a la securite de l'information avec l'introduction de l'IA dans les cyberattaques et le r | texte_filtre | Risque lié à la cybersécurité et à la sécurit | modified | haute |
| bmo-014 | 158 | Principaux risques et risques emergents pouva | Mise a jour du texte sur le Resilience de la technologie et innovation avec notamment l'introduction de l'IA dans les innovations technologiques | texte_retenu | Risque lié à la résilience de la technologie  | modified | haute |
| bmo-015 | 159 | Principaux risques et risques emergents pouva | Mise a jour du texte sur le Risque environnemental et social (changement du titre de la sous-section avec suppression des termes ", y compris les chan | texte_retenu | Risque environnemental et social, y compris l | modified | haute |
| bmo-016 | 160 | Principaux risques et risques emergents pouva | Mise a jour du texte sur le Marche canadien de l'habitation et endettement des particuliers avec le risque de pertes sur les prets a la consommation n | texte_retenu | Marché canadien de l'habitation et endettemen | modified | haute |
| bmo-017 | 160 | Autres facteurs pouvant influer sur les resul | Suppression du texte sur la Politiques budgetaires et monetaires et autres conditions economiques dans les pays ou BMO est present | texte_retenu | Politiques budgétaires et monétaires et autre | removed | haute |
| bmo-018 | 160 | Autres facteurs pouvant influer sur les resul | Mise a jour du texte sur la Legislation fiscale et interpretations connexes avec promulgation de la Loi sur l'impot minimum mondial afin de mettre en  | texte_filtre | Législation fiscale et interprétations connex | modified | haute |
| bmo-019 | 161 | Autres facteurs pouvant influer sur les resul | Modification du texte sur la Modification du portefeuille d'activites avec le retrait de phrases sur les difficultes que peut rencontrer la banque dan | texte_filtre | Modification du portefeuille d'activités | modified | haute |
| bmo-020 | 161 | Autres facteurs pouvant influer sur les resul | Modification du texte sur les Estimations et jugements comptables critiques et normes comptables avec le retrait de phrases sur les impacts possibles  | texte_filtre | Estimations et jugements comptables critiques | modified | haute |
| bmo-021 | 161 | Cadre de gestion des risques | Refonte majeure de l'introduction avec davantage de details sur l'objectif et les composantes du cadre de gestion pour la banque et refonte du graphiq | texte_retenu | Cadre de gestion globale des risques → Cadre  | added | moyenne |
| bmo-022 | 163 | Cadre de gestion des risques | Modification du texte Gouvernance des risques avec davantage de details sur les roles et l'arbre decisionnel | texte_retenu | Gouvernance des risques | added | haute |
| bmo-023 | 164 | Cadre de gestion des risques | Modification du texte dans le role des comites avec notamment l'introduction de la notion de changement climatique et controle interne au sein du Comi | texte_filtre | Cadre de gestion globale des risques → Cadre  | modified | moyenne |
| bmo-024 | 166 | Cadre de gestion des risques | Suppression dans Limite des risques du texte sur l'utilite de la delegation de pouvoir faite aux dirigeants responsables des risques sur l'etablisseme | texte_retenu | Limites de risque | removed | haute |
| bmo-025 | 166 | Cadre de gestion des risques | Modification du texte dans Gestion des politiques globales (anciennement : Politique globale de gestion des risques) avec l'introduction des objectifs | texte_retenu | Gestion des politiques globales | added | haute |
| bmo-026 | 167 | (introduction d'un nouveau graphique) | Refonte majeure du texte dans Cycle de vie de la gestion des risques scinde en trois sous partie Detection et evaluation des risques (qui inclut maint | texte_retenu | Taxinomie des risques → Détection et évaluati | modified | moyenne |
| bmo-028 | 168 | (introduction d'un nouveau graphique) | La section Culture de gestion des risques est modifiee avec l'introduction des notions de Prises de decisions et de Formation | texte_retenu | Culture de gestion des risques | added | haute |
| bmo-030 | 170 | Financement a levier financier | Suppression du texte relatif au Financement a levier financier | texte_retenu | Financement à levier financier 1 | removed | haute |
| bmo-031 | 170 | Introduction | Modification de la definition du risque de marche | texte_retenu | Risque de marché | modified | haute |
| bmo-032 | 171 | Risque de marche lie a l'assurance | Suppression de la sous-section risque de marche lie a l'assurance | texte_retenu | Risque de marché lié à l'assurance | removed | haute |
| bmo-033 | 171 | Risque de change lie aux activites autres que | Suppression d'une phrase traitant l'incidence de la variation des cours de change sur le risque de transaction | texte_filtre | Risque de change lié aux activités autres que | modified | haute |
| bmo-034 | 172 | Risque d'assurance | Le risque d'assurance est deplace dans le risque de strategie et les donnees quantitatives qui etaient publiees en 2024 ne sont plus publiees en 2025  | texte_retenu | Risque de stratégie | added | haute |
| bmo-035 | 173 | Faits nouveaux en matiere de reglementation | Mise a jour du texte avec Consultation BSIF sur LD Normes de liquidite et publication d'un document sur le | texte_retenu | Faits nouveaux en matière de réglementation | added | haute |
| bmo-037 | 175 | Gestion du risque operationnel non financier | Suppression d'un texte sur la mise en oeuvre des principes directeurs du cadre de gestion du risque operationnel non financier | texte_retenu | Gestion du risque opérationnel non financier  | removed | haute |
| bmo-038 | 175 | Risque lie au blanchiment d'argent, au financ | Texte mis a jour pour preciser que le programme de lutte contre le blanchiment d'argent de BMO applique des mesures de diligence et qu'elle surveille  | texte_retenu | Risque lié au blanchiment d'argent, au financ | modified | haute |
| bmo-039 | 176 | Risque lie a l'intelligence artificielle | Mise a jour du texte | texte_retenu | Risque lié à l'intelligence artificielle | modified | haute |
| bmo-040 | 177 | Risque lie aux donnees et a l'analyse | Mise a jour du texte | texte_filtre | Risque lié aux données et à l'analyse | modified | haute |
| bmo-041 | 177 | Risque lie a la fraude interne et externe | Le risque lie a la securite physique n'est plus traite dans ce facteur de risque et est remplace par la fraude externe | texte_retenu | Risque lié à la fraude et à la sécurité physi | added | haute |
| bmo-042 | 177 | Risque lie a la securite physique et a la pro | Nouveau facteur de risque qui reprend le texte du risque lie a la securite physique publie precedemment mais introduit le risque lie a la propriete | texte_filtre | Risque lié à la fraude et à la sécurité physi | modified | haute |
| bmo-043 | 178 | Risque lie a la gestion de projets et de chan | Suppression de ce facteur de risque | texte_retenu | Risque lié à la gestion de projets et de chan | removed | haute |
| bmo-044 | 178 | Risque lie a la culture et au comportement et | Ajout de deux facteurs de risque "Risque lie a la culture et au comportement" et "Risque de paiement" dans un nouveau paragraphe intitule "Les facteur | texte_retenu | Risque opérationnel non financier → Risque li | added | moyenne |
| bmo-045 | 178 | Risque de modele | Mise a jour majeure de la sous-section | texte_retenu | Risque de modèle | modified | haute |
| bmo-046 | 180 | Risque lie a la conformite juridique et regle | Suppression du texte traitant des Questions relatives a la durabilite et aux changements climatiques | texte_retenu | Risque juridique et réglementaire → Risque li | modified | haute |
| bmo-047 | 181 | Protection des consommateurs | Mise a jour du texte sur la Protection des consommateurs | texte_filtre | Protection des consommateurs | modified | haute |
| bmo-048 | 181 | Protection de la vie privee | Mise a jour du texte sur la Protection de la vie privee avec suppression de la mention relative a la Loi C-27 et ajout de la notion d'IA | texte_retenu | Protection de la vie privée | removed | haute |
| bmo-049 | 182 | Faits nouveaux en matiere de reglementation a | Mise a jour du texte sur la reglementation aux Etats-Unis | texte_retenu | Faits nouveaux en matière de réglementation a | modified | haute |
| bmo-050 | 183 | Risque de strategie | Modification de la definition du risque de strategie et integration du risque d'assurance dans le risque de strategie | texte_retenu | Risque de stratégie | added | haute |
| bmo-051 | 184 | Risque environnemental et social | Refonte de la section, notamment avec introduction d'une sous-section Risque climatique et des ambitions de la banque en la matiere | texte_retenu | L'ambition climatique de BMO | added | moyenne |
| bmo-052 | 184 | Risque environnemental et social | Gouvernance, davantage de details sur le role des differents comites/instances | texte_retenu | Risque environnemental et social → Risque d'a | modified | haute |
| bmo-053 | 185 | Risque environnemental et social | Suppression du texte sur la Strategie | texte_retenu | Risque environnemental et social | removed | haute |
| bmo-054 | 185 | Risque environnemental et social | Gestion des risques, Modification du texte. Le texte auparavant intitule "Gestion du risque environnemental et social dans la chaine d'approvisionneme | texte_retenu | Gestion du risque environnemental et social d | added | haute |
| bmo-055 | 188 | Risque environnemental et social | Retrait du texte Mesure et cibles | texte_retenu | Mesures et cibles | removed | haute |
| bmo-056 | 188 | Risque environnemental et social | Risques climatiques, ajout de cette sous-section scindee en Gestion du risque et L'ambition climatique de BMO | texte_retenu | L'ambition climatique de BMO | added | haute |
| bmo-057 | 189 | Risque de reputation | Ajout du terme Reputation dans la definition, ajout du classement de ce risque dans la taxonomie en tant que risque transversal et suppression de la p | texte_retenu | Risque de réputation | modified | haute |

## CIBC — couverts (51)

| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |
|---|---:|---|---|---|---|---|---|
| cibc-001 | 125 | Actif pondere en fonction du risque | Retrait de la mention sur l'utilisation de l'approche NI au lieu de l'approche standard dans le tableau des APR - section risque de credit | tableaux | Composantes de l’actif pondéré en fonction du | modified | haute |
| cibc-002 | 125 | Actif pondere en fonction du risque | Retrait de la mention sur la mise en oeuvre des reformes de Bale III relatives au REC dans le tableau des APR - section Risque lie au rajustement de l | texte_filtre | Réformes de Bâle III et exigences de communic | removed | moyenne |
| cibc-003 | 125 | Actif pondere en fonction du risque | Retrait de la mention des reformes de Bale III relatives au risque de marche dans le tableau des APR - section | tableaux | Composantes de l’actif pondéré en fonction du | modified | moyenne |
| cibc-005 | 126 | Risque de marche | Ajout d'une note de bas de page sur le calcul de plancher de fonds propres en appliquant un facteur d'ajustement du plancher au total de l'APR Reforme | texte_retenu | Réformes de Bâle III et exigences de communic | modified | moyenne |
| cibc-006 | 126 | Cadre de capacite totale d'absorption des per | Retrait des sections sur les Reformes de Bale III et exigences de communication financiere au titre du troisieme pilier revisees et le Cadre de capaci | texte_filtre | Cadre de capacité totale d'absorption des per | renamed | haute |
| cibc-007 | 126 | Ligne directrice Normes de fonds propres du B | Ajout d'une nouvelle section Ligne directrice Normes de fonds propres du BSIF publie le 11 septembre 2025 par le BSIF Exclusion d'un evenement generat | texte_retenu | Réformes de Bâle III et exigences de communic | added | moyenne |
| cibc-008 | 127 | Ligne directrice Normes de fonds propres du B | Ajout d'une nouvelle section Exclusion d'un evenement generateur de pertes operationnelles de l'APR refletant le risque operationnel | texte_retenu | Réformes de Bâle III et exigences de communic | added | haute |
| cibc-009 | 127 | Variations du total des fonds propres regleme | Modification d'un libelle, passage de «Solde des fonds propres de premiere categorie sous forme d'actions ordinaires a la fin de l'exercice», « Solde  | tableaux | Variations du total des fonds propres régleme | modified | haute |
| cibc-010 | 127 | Variations du total des fonds propres regleme | Retrait d'une note de bas de tableau sur l'incidence des reformes de Bale III dans le tableau des variations du total des fonds propres reglementaires | tableaux | Variations du total des fonds propres régleme | modified | haute |
| cibc-011 | 127 | Composantes de l'actif pondere en fonction du | Retrait de plusieurs lignes VAR, VAR en situation de crise, Exigences supplementaires liees aux risques, | tableaux | Composantes de l’actif pondéré en fonction du | modified | moyenne |
| cibc-012 | 128 | Titrisation et autres dans les composantes de | Retrait de 3 notes de bas de tableau sur l'approche NI appliquee a la majeure partie des portefeuilles de credit et l'examen fondamental du portefeuil | tableaux | Composantes de l’actif pondéré en fonction du | modified | haute |
| cibc-013 | 128 | Regime d'achat d'actions par les employes | Retrait de la section Regime d'achat d'actions par les employes | texte_filtre | Régime d'achat d'actions par les employés → D | removed | haute |
| cibc-014 | 128 | Structure de la gouvernance du risque | Mise a jour de la structure de la gouvernance du risque: ajout d'un nouveau comite, comite de la technologique. Le comite de direction a ete renomme p | texte_retenu | Structure de la gouvernance du risque | added | moyenne |
| cibc-015 | 129 | Structure de gestion du risque | Changement de position de la gestion du risque en Europe et Asie-Pacifique qui depend maintenant de la gestion du risque lie aux marches des capitaux | texte_retenu | Structure de gestion du risque | added | moyenne |
| cibc-016 | 130 | Structure de gestion du risque | Mise a jour des definitions des divisions de la structure de gestion des risques | texte_retenu | Structure de gestion du risque | added | haute |
| cibc-017 | 131 | Enonce sur l'interet a l'egard du risque et c | Mise a jour de tous les enonces du cadre de gestion des risques: ajout de nouveaux cadres et limites connexes et surveillance par la direction dans le | texte_retenu | Énoncé sur l'intérêt à l'égard du risque | added | moyenne |
| cibc-018 | 133 | Recensement et mesure des risques | Retrait de la mention du capital economique dans le calcul du risque dans la section Recensement et mesure des risques | texte_retenu | Recensement et mesure des risques | modified | haute |
| cibc-020 | 134 | Gestion du risque lie aux modeles | Ajout de la mention d'un examen de la qualite des donnees dans les politiques d'attenuation du risque lie aux modeles - Recensement et mesure des risq | texte_retenu | Politiques d'atténuation du risque lié aux mo | added | haute |
| cibc-021 | 134 | Principaux risques et nouveaux risques | Incertitude concernant la politique commerciale: Retrait du risque Inflation, taux d'interet et croissance economique pour le nouveau risque Incertitu | texte_retenu | Risque géopolitique → Incertitude concernant  | added | moyenne |
| cibc-022 | 134 | Principaux risques et nouveaux risques | Endettement des consommateurs canadiens et marche du logement canadien: Retrait d'un paragraphe sur la revision par le BSIF des lignes directrices Nor | texte_retenu | Endettement des consommateurs canadiens et ma | removed | haute |
| cibc-023 | 135 | Principaux risques et nouveaux risques | Risque geopolitique: Retrait des mentions des relations entre les Etats-Unis et l'Iran, les enjeux commerciaux persistants comme facteurs preoccupants | texte_retenu | Risque géopolitique | removed | haute |
| cibc-024 | 135 | Principaux risques et nouveaux risques | Risque lie aux changements climatiques: Mise a jour de la section en retirant les mentions de Normes canadiennes d'information sur la durabilite (NCID | texte_retenu | Risque lié aux changements climatiques | added | moyenne |
| cibc-025 | 136 | Principaux risques et nouveaux risques | Risque lie a la securite de l'information et a la cybersecurite: Retrait du segment de la technologie pour en faire un risque a part entiere. Mise a j | texte_retenu | Risque lié à la sécurité de l'information (y  | modified | haute |
| cibc-026 | 136 | Principaux risques et nouveaux risques | Risque lie a la technologie: Nouveau risque sur la technologie qui etait avant regroupe avec le risque de securite de l'information et la cybersecurit | texte_retenu | Risque lié à la technologie | added | haute |
| cibc-028 | 136 | Principaux risques et nouveaux risques | Risque lie aux donnees et a l'intelligence artificielle: Mention que le BSIF a publie sa version definitive de la ligne directrice E-23 sur la gestion | texte_filtre | Risque lié aux données et à l'intelligence ar | modified | haute |
| cibc-030 | 137 | Principaux risques et nouveaux risques | Sanctions et lutte contre le blanchiment d'argent et le financement des activites terroristes: Allegement de la section en retirant plusieurs mentions | texte_retenu | Sanctions et lutte contre le blanchiment d'ar | removed | haute |
| cibc-031 | 137 | Principaux risques et nouveaux risques | Reglementation bancaire aux Etats-Unis: Allegement de la section en retirant plusieurs paragraphes apportant des details sur l'exposition de la filial | texte_retenu | Réglementation bancaire aux États-Unis | removed | haute |
| cibc-032 | 137 | Principaux risques et nouveaux risques | Transition liee a la reforme des taux interbancaires offerts: retrait de la section | texte_filtre | Transition liée à la réforme des taux interba | removed | haute |
| cibc-033 | 137 | Principaux risques et nouveaux risques | Reforme fiscale: Mise a jour de la section pour retirer les anciennes lois (projet de loi C-69, regles du Pilier 2) et mention de la regle relative au | texte_retenu | Réforme fiscale | added | moyenne |
| cibc-036 | 138 | Risques decoulant des activites commerciales | Mention du risque de culture et de securite dans le profil de risque | texte_retenu | Risques liés au comportement et à la culture | added | moyenne |
| cibc-037 | 138 | Risque de credit | Changements sur plusieurs tableaux: Portefeuilles de detail, Qualite du credit des portefeuilles | tableaux | Exposition au risque de crédit | modified | moyenne |
| cibc-038 | 138 | Risque de credit | Modification de libelle dans le tableau des niveaux de risque, passage de la ligne Bas a Faible, passage de Haut a Eleve | tableaux |  | modified | haute |
| cibc-039 | 139 | Risque de credit | Changements sur plusieurs tableaux: Expositions au risque de credit, Expositions assujetties a l'approche standard, Repartition geographique, Expositi | tableaux | Répartition géographique | modified | moyenne |
| cibc-040 | 139 | Risque de credit | Retrait d'une note de bas de tableau sur l'application de l'approche NI a la majeure partie des portefeuilles de credit de CIBC Bank USA pour lesquell | tableaux | Expositions assujetties à l’approche standard | modified | haute |
| cibc-041 | 139 | Expositions au risque de credit | Retrait d'une note de bas de tableau sur l'application de l'approche NI a la majeure partie des portefeuilles de credit de CIBC Bank USA pour lesquell | tableaux | Expositions assujetties à l’approche standard | modified | haute |
| cibc-042 | 139 | Expositions au risque de credit | Modification de bas de tableau pour retirer la mention de la ponderation des titres detenus a des fins autre que de negociation dans le tableau Exposi | tableaux | Exposition au risque de crédit | modified | moyenne |
| cibc-044 | 139 | Expositions liees aux entreprises et aux gouv | Retrait du paragraphe sur la strategie d'attenuation du risque et la protection de credit souscrite dans le tableau | texte_retenu | Expositions liées aux entreprises et aux gouv | removed | haute |
| cibc-045 | 140 | En fonction des paiements reels des clients | Retrait d'un paragraphe sur les precisions sur deux type de prets lies aux coproprietes au Canada: les prets hypothecaires et les prets octroyes aux p | texte_filtre | __intro__ | removed | haute |
| cibc-046 | 140 | Exposition au secteur de l'immobilier de bure | Retrait de la section Exposition au secteur de l'immobilier de bureaux aux Etats-Unis | texte_retenu | Exposition au secteur de l'immobilier de bure | removed | haute |
| cibc-047 | 140 | Activites de titrisation | Ajout de precisions concernant les activites de titrisation, notamment de la titrisation classique et de la notion de risque transfere a des tiers | texte_retenu | Activités de titrisation | modified | haute |
| cibc-048 | 140 | Mesure de risque | Retrait des lignes Engagements de clients en vertu d'acceptations et Acceptations dans le tableau Mesure de risque | tableaux | Obligations contractuelles
Actifs et passifs | modified | haute |
| cibc-050 | 141 | Risque lie aux actions | Modification de libelle, passage de Titres de participation evalues a la JVAERG a Titres de participation designes a la JVAERG dans le tableau de risq | tableaux | Risque lié aux actions | added | haute |
| cibc-051 | 141 | Actifs assortis d'une charge | Retrait d'une note de bas de tableau sur le retraitement de donnees suite a l'adoption de l'IFRS 17 dans le tableau Actifs assortis d'une charge | tableaux | Actifs assortis d’une charge | modified | moyenne |
| cibc-052 | 141 | Obligations contractuelles | Retraits de la ligne Engagements de clients en vertu d'acceptations et Acceptations dans le tableau Obligations contractuelles | tableaux | Obligations contractuelles
Actifs et passifs | modified | haute |
| cibc-054 | 142 | Risque strategique | Ajout de mentions sur les effets de la resilience et la croissance interne ou externe. Retrait des exemples ou les strategies pourraient etre applique | texte_retenu | Autres risques Risque stratégique | modified | haute |
| cibc-055 | 142 | Risque operationnel | Ajout de precisions sur la definition et l'objectif du Cadre d'integrite et de securite de la CIBC | texte_retenu | Risque opérationnel | added | haute |
| cibc-056 | 142 | Risque liee aux donnees | Modification de la definition | texte_retenu | Risque lié aux données | modified | haute |
| cibc-057 | 142 | Risque de fraude | Modification de la definition | texte_retenu | Risque de fraude | modified | haute |
| cibc-058 | 143 | Risque lie a la securite de l'information (y  | Modification de la definition | texte_retenu | Risque lié à la sécurité de l'information (y  | modified | haute |
| cibc-060 | 143 | Gouvernance | Mise a jour de la section: restructuration et ajout de precisions sur la gouvernance et certains roles de supervision | texte_retenu | Gouvernance | modified | moyenne |
| cibc-061 | 144 | Gestion du risque | Ajout d'un paragraphe pour apporter des precisions sur les procedures et controles en place pour evaluer le risque lie aux fournisseurs contractuels. | texte_retenu | Risque lié aux tiers | added | moyenne |

## TD — couverts (24)

| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |
|---|---:|---|---|---|---|---|---|
| td-001 | 114 | Actifs ponderes en fonction des risques par s | Retrait de l'information sur le capital economique du tableau sur les actifs ponderes en fonction des risques par secteur. | texte_retenu | CAPITAL ÉCONOMIQUE ET ACTIFS PONDÉRÉS EN FONC | modified | haute |
| td-002 | 115 | Evolution future des fonds propres reglementa | Ajout d'un texte pour informer de la publication le 20 fevrier 2025 de la ligne directrice du BSIF sur les crypto-actifs et les changements aux lignes | texte_retenu | Évolution future des fonds propres réglementa | added | haute |
| td-003 | 115 | Evolution future des fonds propres reglementa | Ajout de la publication le 11 septembre 2025 de la LD NFP par le BSIF. | texte_retenu | Exigences en matière de fonds propres du BSIF | added | haute |
| td-004 | 115 | Facteurs de risque et gestion des risques | Mise a jour de la section Risques geopolitiques en mettant l'emphase sur les negociations ACEUM, les tarifs douaniers et les tensions Ukraine-Russie e | texte_retenu | Risques géopolitiques | modified | haute |
| td-005 | 115 | Facteurs de risque et gestion des risques | Nouveau risque presente: Risque de catastrophe | texte_retenu | Risque de catastrophe | added | haute |
| td-006 | 116 | Facteurs de risque et gestion des risques | Ajout d'un texte dans "Concurrence, changements de comportements des consommateurs et perturbations liees a la technologie" pour parler des investisse | texte_retenu | États-Unis → Concurrence, changements de comp | added | moyenne |
| td-007 | 116 | Risque environnemental et social (y compris l | Retrait du texte qui parle des pressions externes pour regler les inegalites sociales et financieres | texte_retenu | Risque environnemental et social (y compris l | modified | haute |
| td-008 | 116 | Risque environnemental et social (y compris l | Ajout d'information sur les regles et sanctions de l'ecoblanchiment | texte_retenu | Risque environnemental et social (y compris l | modified | haute |
| td-009 | 117 | Structure de gouvernance pour la gestion des  | Mise a jour du paragraphe du comite de redressement pour preciser le role du conseil d'administration; | texte_retenu | Le comité de redressement | modified | haute |
| td-011 | 118 | Structure de gouvernance pour la gestion des  | Retrait du paragraphe sur la "Gestion des risques lies a la conduite de l'entreprise" | texte_retenu | Gestion des risques liés à la conduite de l'e | removed | haute |
| td-012 | 118 | Structure de gouvernance pour la gestion des  | Retrait de la section sous la section des trois lignes de defense qui detaille les principes de gestion des risques que la banque applique. | texte_retenu | Trois lignes de défense | removed | moyenne |
| td-013 | 118 | Risque de credit | Changement Tableau 42 (Expositions brutes au risque de credit): "Total - Expositions de detail" pour "Total - risque de credit de detail" | tableaux | EXPOSITIONS BRUTES AU RISQUE DE CRÉDIT – Appr | modified | haute |
| td-014 | 118 | Risque de marche | Tableau 44: Mesures du risque de portefeuille. Ajout d'un texte pour expliquer que les categories d'actifs ont ete touchees par la volatilite du aux f | texte_retenu | RESPONSABLES DE LA GESTION DU RISQUE DE MARCH | added | moyenne |
| td-015 | 119 | Risque de marche | Presentation des donnees dans le tableau Sensibilite au risque de taux d'interet structurel par dollar canadien et dollar americain et ajout d'une not | tableaux | SENSIBILITÉ AU RISQUE DE TAUX D’INTÉRÊT STRUC | modified | haute |
| td-016 | 119 | Risque de marche | Retrait de la note de bas de page en lien avec l'adoption de l'IFRS 17. | tableaux | LIENS ENTRE LE RISQUE DE MARCHÉ ET LE BILAN | modified | haute |
| td-017 | 119 | Risque operationnel | Changement de la definition d'un tiers | texte_retenu | Gestion des tiers | modified | haute |
| td-018 | 120 | Risque operationnel | Mise a jour du texte sur le "Risque en matiere de conduite". | texte_retenu | Gestion du risque interne | modified | haute |
| td-019 | 120 | Risque operationnel | Ajout d'un texte sur le "Risque de conformite a la reglementation". | texte_retenu | Gestion du risque interne → Risque de conform | added | haute |
| td-020 | 120 | Risque de modele | Ajout d'un texte sur le "Risque lie a l'intelligence artificielle". | texte_retenu | Risque lié à l'intelligence artificielle | added | haute |
| td-021 | 120 | Risque de liquidite | Mise a jour des comites et sous-comites responsables de la gestion du risque de liquidite | texte_retenu | RESPONSABILITÉ EN MATIÈRE DE GESTION DU RISQU | added | haute |
| td-022 | 121 | Risque de liquidite | Retrait des textes explicatifs du ratio LCR et les ratios reglementaires | texte_retenu | APPÉTIT POUR LE RISQUE DE LIQUIDITÉ DE LA TD | removed | haute |
| td-023 | 121 | Risque de liquidite | Tableau 48 (Sommaire des actifs liquides moyens par type et par monnaie): Fusion des postes "Obligations du gouvernement du Canada" et "Titres adosses | tableaux | SOMMAIRE DES ACTIFS LIQUIDES MOYENS PAR TYPE  | modified | haute |
| td-025 | 122 | Risque de liquidite | Ajout d'un texte pour enumerer les consultations lancees en 2025 par le BSIF. | texte_retenu | FAITS NOUVEAUX RÉGLEMENTAIRES CONCERNANT LA L | added | haute |
| td-026 | 123 | Risque lie aux crimes financiers | Le risque lie aux crimes financiers a ete retire de la section "Mode de gestion du risque juridique et de conformite a la reglementation (y compris le | texte_retenu | MODE DE GESTION DU RISQUE JURIDIQUE ET DE CON | modified | haute |

## BNS / Scotia — couverts (26)

| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |
|---|---:|---|---|---|---|---|---|
| bns-001 | 99 | Reserve pour stabilite interieure | Mise a jour du texte sur la reserve pour stabilite interieure (RSI) de 3,5% pour mentionner que le taux a ete maintenu en juin 2024. | texte_filtre | Fonds propres réglementaires | modified | haute |
| bns-002 | 99 | Modifications a la reglementation liee aux fo | Ajout d'un texte portant sur le report d'augmentations du plancher des fonds propres par le BSIF. | texte_retenu | Le BSIF reporte d'autres augmentations du niv | added | haute |
| bns-003 | 100 | Modifications a la reglementation liee aux fo | Ajout d'un texte sur la publication en fevrier 2025 de la ligne directrice du BSIF sur le regime au regard des normes de fonds propres et de liquidite | texte_retenu | Modifications à la réglementation liée aux fo | added | haute |
| bns-004 | 100 | Notations de credit | Retrait du paragraphe sur les notations de credit | texte_retenu | Notations de crédit | removed | haute |
| bns-005 | 100 | Tableau Variations des fonds propres reglemen | Modification au poste "Autres modifications, y compris les ajustements reglementaires et le retrait graduel des instruments non admissibles" pour gard | tableaux | T31 Variation des fonds propres réglementaire | modified | moyenne |
| bns-006 | 100 | Actifs ponderes en fonction des risques | Ajout d'une mention par rapport au report de l'augmentation du plancher des APR que le BSIF s'est engage a aviser les banques concernes au moins deux  | texte_filtre | Actifs pondérés en fonction des risques | modified | haute |
| bns-007 | 100 | Risque de marche - actifs ponderes en fonctio | Mention du BSIF qui a mis en place durant le premier trimestre de 2024 le cadre de gestion revise du risque de marche a la suite de la revision comple | texte_retenu | Risque de marché - actifs pondérés en fonctio | modified | haute |
| bns-008 | 102 | Cadre de gestion du risque | Refonte de la section et ajout de plusieurs autres sections (voir plus bas) | texte_retenu | Cadre de gestion du risque | added | moyenne |
| bns-010 | 103 | Cadre de gestion du risque | Ajout du cycle de gestion du risque et d'un graphique l'accompagnant. | texte_retenu | Cycle de gestion du risque | added | haute |
| bns-011 | 104 | Cadre de gestion du risque | Retrait des sections Cadres, politiques et limites | texte_retenu | Cadres et politiques | removed | haute |
| bns-012 | 104 | Cadre de gestion du risque | Retrait de la section Surveillance et presentation de l'information | texte_retenu | Surveillance et présentation de l'information | removed | haute |
| bns-013 | 104 | Gouvernance et surveillance du risque | Section Structure de gouvernance: Ajout du paragraphe sur leur comite responsable des technologies du conseil | texte_retenu | Structure de gouvernance de la gestion du ris | added | haute |
| bns-014 | 105 | Gouvernance et surveillance du risque | Ajout d'une section sur les exigences en matiere de surveillance du risque | texte_retenu | La gestion efficace du risque repose sur une  | added | haute |
| bns-015 | 105 | Risques importants et emergents | Ajout d'un paragraphe sur l'evolution des politiques gouvernementales | texte_retenu | Évolution des politiques gouvernementales | added | haute |
| bns-016 | 105 | Risques importants et emergents | Ajout d'un paragraphe sur l'adoption de l'IA | texte_retenu | Adoption rapide de l'IA | added | haute |
| bns-017 | 106 | Risques importants et emergents | Ajout d'un paragraphe sur la gestion du changement | texte_retenu | Risques importants et risques émergents | added | moyenne |
| bns-018 | 106 | Risques importants et emergents | Ajout d'un paragraphe sur les risques par contagion | texte_retenu | Risques par contagion | added | haute |
| bns-019 | 106 | Risque de marche | Dans tableau Sensibilite aux taux d'interet structurels, les donnees auparavant presentees uniquement en dollar canadien sont maintenant aussi present | tableaux | T50 Sensibilité aux taux d’intérêt structurel | modified | haute |
| bns-020 | 107 | Risque de marche | Ajout que depuis le 1er novembre la VaR comprend la volatilite de l'ecart de credit propre aux emetteurs, qui etait auparavant dans la VaR propre a la | texte_retenu | Valeur à risque (VàR) → Sommaire des mesures  | modified | haute |
| bns-021 | 107 | Risque de marche | Ajout du Risque sur actions dans la section du risque de marche | texte_retenu | Risque de marché lié aux activités de transac | added | moyenne |
| bns-022 | 107 | Risque de marche | Retrait de la mention que la banque utilise l'approche standard pour calculer les fonds propres | texte_retenu | Interdépendance du risque de marché et de l'é | removed | haute |
| bns-023 | 107 | Risque de marche | Retrait de la note de bas de page en lien avec l'adoption de l'IFRS 17. | tableaux |  | modified | haute |
| bns-024 | 108 | Risque de liquidite | Ajout d'un paragraphe sur les instruments derives | texte_retenu | Instruments dérivés | added | haute |
| bns-025 | 108 | Risque de liquidite | Actifs greves: Ajout d'un paragraphe sur les notations de credit | texte_retenu | Notations de crédit | added | haute |
| bns-027 | 108 | Principaux risques non financiers | Ajout d'un paragraphe sur les risques de pots-de-vin et de corruption dans la section Risque de conformite. | texte_retenu | Risque de conformité | added | haute |
| bns-028 | 108 | Principaux risques non financiers | Dans la section des risques ESG, retrait de la mention que la banque est membre de l'alliance Net Zero (NZBA) | texte_filtre | Rôle de la direction | removed | haute |

## RBC — couverts (31)

| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |
|---|---:|---|---|---|---|---|---|
| rbc-001 | 66 | Accord de Bale III | Retrait du paragraphe sur l'adoption des reformes definitives de Bale III du CBCB du BSIF Fonds propres reglementaires, TLAC disponible, actif pondere | texte_retenu | Accord de Bâle III | removed | haute |
| rbc-002 | 66 | Accord de Bale III | Retrait d'une note de bas de tableau sur le retraitement des chiffres des periodes precedentes et l'adoption de l'IFRS 17 | tableaux | Fonds propres réglementaires et TLAC disponib | modified | moyenne |
| rbc-004 | 67 | Fonds propres reglementaires et TLAC disponib | Modification de libelle: remplacement du terme provisions par dotations | tableaux | Fonds propres réglementaires et TLAC disponib | modified | haute |
| rbc-005 | 67 | Fonds propres reglementaires et TLAC disponib | Retrait d'une note de bas de tableau sur le retraitement des chiffres des periodes precedentes et l'adoption de l'IFRS 17 | tableaux | Fonds propres réglementaires et TLAC disponib | modified | moyenne |
| rbc-006 | 67 | Continuite du ratio CET1 (Bale III) | Ajout d'une note de bas de tableau pour mentionner l'exclusion de l'indice de change dans l'augmentation de l'APR | texte_retenu | Actif pondéré en fonction des risques au titr | added | haute |
| rbc-007 | 67 | Actif pondere en fonction des risques au titr | Ajout d'un paragraphe sur l'annonce du BSIF de reporter de facon indefini les augmentations du plancher de fonds propres exige par sa ligne directrice | texte_filtre | Accord de Bâle III → Actif pondéré en fonctio | modified | haute |
| rbc-008 | 68 | Total de l'actif pondere en fonction des risq | Retrait d'une note de bas de tableau sur l'impact du cadre revise du risque de marche dans le cadre des reformes de Bale III | tableaux | Total de l’actif pondéré en fonction des risq | modified | haute |
| rbc-009 | 68 | Fonds propres attribues | Ajout d'une phrase mentionnant la mise a jour de la methode d'attribution des fonds propres afin de refleter plus fidelement les exigences en matiere  | texte_retenu | Fonds propres attribués | added | haute |
| rbc-010 | 68 | Faits nouveaux en matiere de reglementation | Retrait de l'ancien paragraphe sur l'adoption du Cadre de capacite totale d'absorption des pertes pour un nouveau paragraphe sur les revisions aux lig | texte_retenu | Faits nouveaux en matière de réglementation → | removed | haute |
| rbc-011 | 69 | Vue d'ensemble - Principes de gestion des ris | Mise a jour de la vue d'ensemble et des principes de gestion des risques: ajout de l'evaluation de l'incidence decoulant du choix et de l'execution d' | texte_retenu | Principes de gestion des risques | modified | haute |
| rbc-012 | 70 | Vue d'ensemble - Principes de gestion des ris | Ajout d'une mention sur le cycle de vie de la gestion des risques, y compris la definition et l'habitation, l'identification et l'evaluation, la gesti | texte_retenu | Principes de gestion des risques | modified | haute |
| rbc-014 | 71 | Gouvernance des risques | Ajout d'une phrase dans la gouvernance des risques pour mentionner l'harmonisation de l'appetit pour le risque, les strategies commerciales et les act | texte_retenu | Gouvernance des risques | modified | haute |
| rbc-015 | 71 | Conseil d'administration | Modification de libelle: passage de «code de conduite» a «Code de deontologie» | texte_filtre | CONSEIL D'ADMINISTRATION | modified | haute |
| rbc-018 | 71 | Conseil d'administration | Comite de gouvernance: modification de libelles, passage «coordonner les questions ESG» a «coordonner les questions liees a la durabilite» | texte_retenu | Gouvernance | added | haute |
| rbc-020 | 73 | Surveillance des risques | Retrait de trois sous sections de la deuxieme ligne de defense - surveillance des risques: gestion des risques, conformite mondiale, et lutte au blanc | texte_retenu | Surveillance de la gestion des risques et ges | removed | moyenne |
| rbc-021 | 73 | Surveillance des risques | Changements generaux: Appetit pour le risque, Enonces relatifs a l'appetit pour le risque | texte_retenu | Appétit pour le risque | modified | haute |
| rbc-022 | 73 | Surveillance des risques | Modification de libelle: passage de code de conduite a Code de deontologie | texte_retenu | Risques liés à la culture et à la conduite | modified | haute |
| rbc-023 | 73 | Enonces qualitatifs | Ajout d'une mention sur l'evaluation des risques decoulant du choix et l'execution d'une strategie | texte_retenu | Principes de gestion des risques | modified | haute |
| rbc-024 | 73 | Simulation de crise | Retrait des precisions sur le fonctionnement et l'utilisation des resultats de simulation de crise | texte_retenu | Simulations de crise | modified | haute |
| rbc-025 | 73 | Simulation de crise | Ajout d'une phrase pour specifier que les simulations sont effectuees en continu et de maniere independante | texte_retenu | Simulations de crise | modified | haute |
| rbc-026 | 74 | Gouvernance et validation des modeles | Retrait de la section Gouvernance et validation des modeles | texte_retenu | Gouvernance et validation des modèles | removed | haute |
| rbc-027 | 74 | Controle des risques | Retrait de precisions sur le Cadre mettant en oeuvre de processus officiels et independants dans le controle des risques | texte_retenu | Contrôle des risques | modified | haute |
| rbc-029 | 74 | Architecture des documents de gestion du risq | Retrait des exemples de politiques de gestion des risques specifiques a l'echelle de l'entreprise | texte_retenu | Contrôle des risques | removed | haute |
| rbc-030 | 75 | Architecture des documents de gestion du risq | Mise a jour de certains de cadre de gestion des risques: passage de a Cadre de gestion du risque lie a la conduite et a la culture d'entreprise et pas | texte_filtre | Risques liés à la culture et à la conduite | modified | haute |
| rbc-031 | 76 | Architecture des documents de gestion du risq | Retrait du mot delegues dans le titre Appetit pour le risque, pouvoirs d'approbation des risques et limites de risque | texte_retenu | Appétit pour le risque, pouvoirs d'approbatio | added | haute |
| rbc-032 | 76 | Architecture des documents de gestion du risq | Mise a jour de la section, retrait de la definition de l'appetit pour le risque, retrait des precisions sur les pouvoirs et limites | texte_retenu | Appétit pour le risque | modified | haute |
| rbc-033 | 76 | Processus d'analyse et d'approbation des risq | Retraits des precisions sur les pouvoirs et limites | texte_retenu | Processus d'analyse et d'approbation des risq | removed | haute |
| rbc-035 | 76 | Gestion des controles internes de gestion des | Retrait des precisions et des mentions sur les enjeux et mention du modele des trois lignes de defenses | texte_retenu | Surveillance de la gestion des risques et ges | modified | moyenne |
| rbc-036 | 77 | Communication aux echelons superieurs des ris | Modification du titre, passage de Communication aux echelons superieurs des risques et des enjeux lies aux evenements a Communication aux echelons sup | texte_filtre | Communication aux échelons supérieurs des ris | modified | haute |
| rbc-037 | 77 | Communication aux echelons superieurs des ris | Retrait de la mention des enjeux | texte_filtre | Communication aux échelons supérieurs des ris | modified | haute |
| rbc-038 | 77 | Risques connus et risques emergents | Risques lies au contexte commercial et a la conjoncture economique: ajout de notions sur les politiques commerciales protectionnistes, pression budget | texte_retenu | Politiques budgétaires, monétaires et autres | modified | moyenne |

## BNC — couverts (37)

| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |
|---|---:|---|---|---|---|---|---|
| bnc-001 |  |  | Dans le texte sur la mesure du rendement du capital, remplacement du terme « périodiquement » par « trimestriellement ». | texte_retenu | Cadre de gestion du capital | modified | haute |
| bnc-002 |  |  | Précision du rôle du CGIR : supervision trimestrielle de la gestion du capital, délégation de pouvoirs reçue du Comité des risques globaux et surveill | texte_retenu | Structure et gouvernance | modified | haute |
| bnc-003 |  |  | Retrait du texte concernant l’entrée en vigueur, au deuxième trimestre de 2023, de certaines révisions apportées par le BSIF à ses règles sur les fond | texte_retenu | Accord de Bale | modified | haute |
| bnc-004 |  |  | Modification du texte sur le Cadre de capacité totale d’absorption des pertes applicable aux sociétés mères des banques d’importance systémique intéri | texte_filtre | Accord de Bale | modified | haute |
| bnc-005 |  |  | Ajout de la notion de « comportements attendus » relativement aux programmes de rémunération incitative. | texte_retenu | Cadre de la gestion des risques | modified | haute |
| bnc-006 |  |  | Ajout d’une précision sur le soutien apporté par la troisième ligne de défense à la promotion de la solidité financière à long terme de la BNC. | texte_filtre | Integration de la gestion des risques a la cu | modified | haute |
| bnc-007 |  |  | Remplacement de « Comité de risque des marchés financiers » par « Comité de risque des marchés des capitaux ». | texte_retenu | Risque de marche | modified | haute |
| bnc-008 |  |  | Ajout du terme « accessibilité » à la mission du Comité de ressources humaines. | texte_filtre | Le comite de ressources humaines | modified | haute |
| bnc-009 |  |  | Remplacement de « Expérience employé » par « Expérience et performance humaine ». | texte_filtre | Le groupe de travail sur la surveillance des  | modified | haute |
| bnc-010 |  |  | Reformulation du texte concernant le Comité ESG. | texte_retenu | Le comite ESG | modified | haute |
| bnc-011 |  |  | Le « Bureau de la protection des renseignements personnels » devient « l’Équipe de la protection des renseignements personnels et de l’intelligence ar | texte_retenu | L'equipe de la protection des renseignements  | added | haute |
| bnc-012 |  |  | Le « Comité du risque de réputation » devient le « Comité des risques de réputation, d’intégrité et de sécurité ». Le texte explicatif est également m | texte_retenu | Le comite des risques de reputation, de cultu | added | haute |
| bnc-013 |  |  | Reformulation du texte sur la gouvernance de la gestion du risque de modèle. | texte_retenu | Gouvernance de la gestion du risque de modele | modified | haute |
| bnc-014 |  |  | Précision de la fréquence trimestrielle des rencontres de l’Audit interne avec le Comité d’audit et de l’assurance fournie dans le cadre de ses travau | texte_retenu | Evaluation independante par le service de l'A | modified | haute |
| bnc-015 |  |  | Ajout de textes sur le risque de fraude ainsi que sur l’intégrité et la sécurité, et retrait du texte sur les changements climatiques. | texte_retenu | Risque de fraude | added | moyenne |
| bnc-016 |  |  | Ajout d’un texte sur la ligne directrice E-23 du BSIF relative à la gestion du risque de modélisation, dont la version révisée entrera en vigueur le 1 | texte_retenu | Dependance envers les tiers et les modeles | added | moyenne |
| bnc-017 |  |  | Mise à jour des autres thèmes liés aux risques principaux et émergents. | texte_filtre | Risques principaux et risques emergents | modified | faible |
| bnc-018 |  |  | Retrait du texte « Procédures judiciaires et réglementaires ». | texte_retenu | Procedures judiciaires et reglementaires | removed | haute |
| bnc-019 |  |  | Ajout d’une référence aux modifications apportées à la réglementation touchant les activités de la Banque. | texte_retenu | Risque de non-conformite a la reglementation | modified | moyenne |
| bnc-020 |  |  | Ajout de mentions concernant l’impact des risques émergents sur le risque de crédit et toute autre analyse pertinente. | texte_retenu | Reddition de comptes | modified | haute |
| bnc-021 |  |  | Mise à jour du texte sur les faits nouveaux en matière de réglementation suivis depuis le 1er novembre 2024. | texte_retenu | Contexte reglementaire | modified | haute |
| bnc-022 |  |  | Introduction d’un exercice visant à évaluer les effets des tarifs douaniers sur la situation financière des titulaires d’emprunts, en remplacement de  | texte_retenu | Simulations de crises | modified | haute |
| bnc-023 |  |  | Ajout d’une précision sur la périodicité de la présentation des comptes sous surveillance à la direction des groupes de gestion du risque. | texte_retenu | Suivi des comptes sous surveillance et recouv | modified | haute |
| bnc-025 |  |  | Remplacement général de « risque de marché financier » par « risque de marché des capitaux », retrait des notes liées à IFRS 17 et retrait du terme «  | texte_retenu | Risque de marche | modified | moyenne |
| bnc-026 |  |  | Modification du texte sur le risque de taux d’intérêt dans le portefeuille bancaire afin de mieux préciser la nature du risque de taux d’intérêt. | texte_retenu | Risque de taux d'interet dans le portefeuille | modified | moyenne |
| bnc-027 |  |  | Suppression du texte sur la ligne directrice « Normes de liquidité » et ajout d’un texte sur le PIEAL, les flux de liquidités transfrontaliers et l’as | texte_retenu | Contexte reglementaire | modified | moyenne |
| bnc-028 |  |  | Ajout d’une référence à l’Audit interne dans le texte consacré à l’ALCO. | texte_retenu | Contexte reglementaire | modified | moyenne |
| bnc-029 |  |  | Suppression du mot « structurel » dans le texte sur le ratio de liquidité à long terme, ou NSFR. | texte_retenu | Ratio structurel de liquidite a long terme | removed | haute |
| bnc-030 |  |  | Retrait d’une phrase décrivant la mission du CGRO relativement à la mise en place de cadres adéquats et au suivi de leur application. | texte_retenu | Cadre de gestion du risque operationnel | modified | haute |
| bnc-031 |  |  | Retrait d’un paragraphe sur l’analyse et les leçons tirées des événements opérationnels observés dans d’autres grandes entreprises. | texte_retenu | Analyse et lecons apprises des evenements ope | removed | haute |
| bnc-032 |  |  | Ajout d’un texte précisant les risques et les conséquences associés à la non-conformité. | texte_retenu | Risque de non-conformite a la reglementation | modified | haute |
| bnc-033 |  |  | Ajout d’un texte sur la création d’une vice-présidence consacrée à la lutte contre le blanchiment d’argent. | texte_retenu | Structure organisationnelle de la Conformite | added | haute |
| bnc-034 |  |  | Suppression des textes portant sur la Charte de la langue française, les prêts hypothécaires dans des circonstances exceptionnelles, les organismes ex | texte_retenu | Ligne directrice sur les prets hypothecaires  | removed | moyenne |
| bnc-035 |  |  | Ajout de textes portant sur les pratiques commerciales abusives, les comptes à frais modiques ou sans frais, la protection des consommateurs, la compe | texte_retenu | Engagement a fournir des comptes a frais modi | added | haute |
| bnc-036 |  |  | Mise à jour des textes sur le blanchiment d’argent, la protection des renseignements personnels, la FATCA, la norme commune de déclaration, les actifs | texte_retenu | Protection des renseignements personnels | modified | moyenne |
| bnc-038 |  |  | Modification du texte sur la structure de gouvernance et ajout d’un passage sur le caractère collaboratif de la gouvernance au sein de la Banque. | texte_retenu | Structure et gouvernance | modified | moyenne |
| bnc-039 |  |  | Suppression des textes relatifs aux engagements PCAF, NZBA et PRB Biodiversity Community, et ajout de textes sur la quantification des émissions de GE | texte_retenu | Gestion du risque | modified | haute |

## BMO — non couverts (5)

- **bmo-002** p.151 · Cadre — Suppression d'un texte sur l'objectif de repartition des fonds propres
  - juge: Aucun candidat ne mentionne la suppression d'un texte sur l'objectif de répartition des fonds propres dans la section 'Cadre'.
- **bmo-004** p.151 · Exigences en matiere de fonds propres reglementaires — Modification de libelles dans le tableau des exigences
  - juge: Aucun candidat ne mentionne une modification de libellés dans le tableau des exigences en matière de fonds propres réglementaires.
- **bmo-027** p.168 · (introduction d'un nouveau graphique) — La section Surveillance et signalement des risques devient Evaluation du capital pondere en fonction des risques et simulation de crise, sans modification importante dans les textes
  - juge: Aucun candidat ne traite du changement de nom de la section 'Surveillance et signalement des risques' en 'Évaluation du capital pondéré en fonction des risques et simulation de crise'.
- **bmo-029** p.170 · Analyse des portefeuilles — Suppression des graphiques sur les engagements de credit en cours ventilant Canada et autres pays versus
  - juge: Aucun candidat ne mentionne la suppression des graphiques sur les engagements de crédit ventilant Canada et autres pays.
- **bmo-036** p.174 · Definition et introduction — Modification de la definition du risque operationnel (suppression de l'enumeration des risques sous-jacents) et simplification de l'introduction en focalisant sur le cadre de gestion
  - juge: Aucun candidat ne traite spécifiquement de la modification de la définition du risque opérationnel et de la simplification de l'introduction en focalisant sur le cadre de gestion.

### BMO — detectes mais ecartes avant l'export (14)

- **bmo-001** p.151 · Cadre — De maniere general, les ternes "groupes d'exploitation" et suffisance du capital" sont respectivement remplaces par "unites d'exploitation" et "adequa
  - candidat ecarte: Demande de capital (modified, p.[60, 61])
- **bmo-003** p.151 · Exigences en matiere de fonds propres reglementaires — Ajout d'un texte sur l'entree en vigueur du cadre de capacite totale d'absorption des pertes par etablissement des societes meres de BISN
  - candidat ecarte: Évolution des exigences en matière de fonds propres réglementaires → Exigences en matière de fonds propres réglementaires (modified, p.[61, 64])
- **bmo-005** p.152 · Ratios de fonds propres reglementaires et de la capacite totale d'absorption des pertes — Ajout des termes "moins les deductions reglementaires". Auparavant etait "net des deductions"
  - candidat ecarte: Ratios de fonds propres réglementaires et de la capacité totale d'absorption des pertes (modified, p.[61, 63])
- **bmo-008** p.153 · Evolution des exigences en matiere de fonds propres reglementaires — Capital economique et actifs ponderes en fonction des risques par unite d'exploitation et type de risque
  - candidat ecarte: Capital économique et actifs pondérés en fonction des risques par groupe d'exploitation et type de risque → Capital économique et actifs pondérés en fonction des risques par unité d'exploitation et type de risque (renamed, p.[61, 63])
- **bmo-009** p.153 · Evolution des exigences en matiere de fonds propres reglementaires — Modification de libelles dans le tableau "Actifs ponderes en fonction des risques" devient "Actifs ponderes par type de risque" et "Services bancaires
  - candidat ecarte: Capital économique et actifs pondérés en fonction des risques par groupe d'exploitation et type de risque → Capital économique et actifs pondérés en fonction des risques par unité d'exploitation et type de risque (modified, p.[65, 66])
- **bmo-013** p.157 · Principaux risques et risques emergents pouvant nuire aux resultats futurs — Mise a jour du texte sur le Risque lie a la cybersecurite et a la securite de l'information avec l'introduction de l'IA dans les cyberattaques et le r
  - candidat ecarte: Risque lié à la cybersécurité et à la sécurité de l'information (modified, p.[70, 71])
- **bmo-018** p.160 · Autres facteurs pouvant influer sur les resultats futurs — Mise a jour du texte sur la Legislation fiscale et interpretations connexes avec promulgation de la Loi sur l'impot minimum mondial afin de mettre en 
  - candidat ecarte: Législation fiscale et interprétations connexes (modified, p.[71, 73])
- **bmo-019** p.161 · Autres facteurs pouvant influer sur les resultats futurs — Modification du texte sur la Modification du portefeuille d'activites avec le retrait de phrases sur les difficultes que peut rencontrer la banque dan
  - candidat ecarte: Modification du portefeuille d'activités (modified, p.[73])
- **bmo-020** p.161 · Autres facteurs pouvant influer sur les resultats futurs — Modification du texte sur les Estimations et jugements comptables critiques et normes comptables avec le retrait de phrases sur les impacts possibles 
  - candidat ecarte: Estimations et jugements comptables critiques et normes comptables (modified, p.[71, 73])
- **bmo-023** p.164 · Cadre de gestion des risques — Modification du texte dans le role des comites avec notamment l'introduction de la notion de changement climatique et controle interne au sein du Comi
  - candidat ecarte: Cadre de gestion globale des risques → Cadre de gestion des risques (modified, p.[72, 74])
- **bmo-033** p.171 · Risque de change lie aux activites autres que de negociation — Suppression d'une phrase traitant l'incidence de la variation des cours de change sur le risque de transaction
  - candidat ecarte: Risque de change lié aux activités autres que de négociation (modified, p.[90, 91])
- **bmo-040** p.177 · Risque lie aux donnees et a l'analyse — Mise a jour du texte
  - candidat ecarte: Risque lié aux données et à l'analyse (modified, p.[102, 105])
- **bmo-042** p.177 · Risque lie a la securite physique et a la propriete — Nouveau facteur de risque qui reprend le texte du risque lie a la securite physique publie precedemment mais introduit le risque lie a la propriete
  - candidat ecarte: Risque lié à la fraude et à la sécurité physique → Risque lié à la sécurité physique et à la propriété (modified, p.[102, 105])
- **bmo-047** p.181 · Protection des consommateurs — Mise a jour du texte sur la Protection des consommateurs
  - candidat ecarte: Protection des consommateurs (modified, p.[105, 108])

## CIBC — non couverts (7)

- **cibc-004** p.126 · Risque de marche — Retrait de la mention de la revision de l'approche standard au deuxieme trimestre de 2023 relative au risque operationnel dans le tableau des APR - section Risque operationnel
  - juge: Aucun candidat ne traite du retrait de la mention de la révision de l'approche standard au deuxième trimestre de 2023 relative au risque opérationnel dans le tableau des APR.
- **cibc-019** p.134 · Gestion du risque lie aux modeles — Ajout de la mention des modeles d'IA dans la gestion du risque lie aux modeles - Recensement et mesure des risques
  - juge: Aucun candidat ne mentionne l'ajout des modèles d'IA dans la gestion du risque lié aux modèles.
- **cibc-035** p.138 · Risques decoulant des activites commerciales — Mise a jour d'une unite d'exploitation strategique et activites commerciales pour retirer les mentions sur les services financiers directs
  - juge: Aucun candidat ne mentionne le retrait des services financiers directs dans les unités d'exploitation stratégiques.
- **cibc-043** p.139 · Expositions liees aux entreprises et aux gouvernements par secteur d'activite — Changement de l'ordre du libelle, passage de Diffusion, edition et impression a Edition, impression et diffusion dans le tableau Expositions liees aux entreprises et aux gouvernements par secteur d'ac
  - juge: Aucun candidat ne traite du changement de l'ordre du libellé dans le tableau mentionné par l'observation.
- **cibc-049** p.141 · Mesure de risque — Retrait de deux notes de bas de tableau pour preciser que les donnees refletent les reformes de Bale III et l'adoption de l'IFRS 17 dans le tableau Mesure de risque
  - juge: Aucun candidat ne mentionne le retrait des notes de bas de tableau concernant les réformes de Bâle III et l'IFRS 17 dans le tableau Mesure de risque.
- **cibc-053** p.141 · Obligations contractuelles — Modification d'une note de bas de tableau en retirant la mention sur le retraitement de donnees la suite de l'adoption de l'IFRS 17 dans le tableau Obligations contractuelles
  - juge: Aucun candidat ne mentionne la suppression d'une note de bas de tableau concernant l'IFRS 17 dans le tableau des obligations contractuelles.
- **cibc-059** p.143 · Risque lie a la technologie — Modification de la definition
  - juge: Aucun candidat ne traite spécifiquement de la modification de la définition du risque lié à la technologie.

### CIBC — detectes mais ecartes avant l'export (6)

- **cibc-002** p.125 · Actif pondere en fonction du risque — Retrait de la mention sur la mise en oeuvre des reformes de Bale III relatives au REC dans le tableau des APR - section Risque lie au rajustement de l
  - candidat ecarte: Réformes de Bâle III et exigences de communication financière au titre du troisième pilier révisées (removed, p.[55])
- **cibc-006** p.126 · Cadre de capacite totale d'absorption des pertes par etablissement des societes mere — Retrait des sections sur les Reformes de Bale III et exigences de communication financiere au titre du troisieme pilier revisees et le Cadre de capaci
  - candidat ecarte: Cadre de capacité totale d'absorption des pertes par établissement des sociétés mères → Ligne directrice Normes de fonds propres du BSIF (renamed, p.[44, 55])
- **cibc-013** p.128 · Regime d'achat d'actions par les employes — Retrait de la section Regime d'achat d'actions par les employes
  - candidat ecarte: Régime d'achat d'actions par les employés → Données sur les actions en circulation (removed, p.[57])
- **cibc-028** p.136 · Principaux risques et nouveaux risques — Risque lie aux donnees et a l'intelligence artificielle: Mention que le BSIF a publie sa version definitive de la ligne directrice E-23 sur la gestion
  - candidat ecarte: Risque lié aux données et à l'intelligence artificielle (modified, p.[64, 70])
- **cibc-032** p.137 · Principaux risques et nouveaux risques — Transition liee a la reforme des taux interbancaires offerts: retrait de la section
  - candidat ecarte: Transition liée à la réforme des taux interbancaires offerts (removed, p.[71])
- **cibc-045** p.140 · En fonction des paiements reels des clients — Retrait d'un paragraphe sur les precisions sur deux type de prets lies aux coproprietes au Canada: les prets hypothecaires et les prets octroyes aux p
  - candidat ecarte: __intro__ (removed, p.[61])

## TD — non couverts (2)

- **td-010** p.117 · Structure de gouvernance pour la gestion des risques — Ajout d'un nouveau comite nomme "Sous-comite de redressement".
  - juge: Aucun candidat ne mentionne l'ajout d'un 'Sous-comite de redressement' dans la structure de gouvernance pour la gestion des risques.
- **td-024** p.122 · Risque de liquidite — Tableau Ratio LCR: Ajout d'une note de bas de tableau pour expliquer que la cellule est sans donnee selon le gabarit de divulgation
  - juge: Aucun candidat ne mentionne l'ajout d'une note de bas de tableau pour expliquer l'absence de données dans le tableau Ratio LCR.

## BNS / Scotia — non couverts (2)

- **bns-009** p.102 · Cadre de gestion du risque — Ajout de la section "Taxonomie du risque d'entreprise".
  - juge: Aucun candidat ne mentionne l'ajout d'une section 'Taxonomie du risque d'entreprise' dans le cadre de gestion du risque.
- **bns-026** p.108 · Principaux risques non financiers — Mention de la ligne directrice E-21 du BSIF dans le risque d'exploitation
  - juge: Aucun candidat ne mentionne la ligne directrice E-21 du BSIF dans le risque d'exploitation.

### BNS / Scotia — detectes mais ecartes avant l'export (3)

- **bns-001** p.99 · Reserve pour stabilite interieure — Mise a jour du texte sur la reserve pour stabilite interieure (RSI) de 3,5% pour mentionner que le taux a ete maintenu en juin 2024.
  - candidat ecarte: Fonds propres réglementaires (modified, p.[58, 63])
- **bns-006** p.100 · Actifs ponderes en fonction des risques — Ajout d'une mention par rapport au report de l'augmentation du plancher des APR que le BSIF s'est engage a aviser les banques concernes au moins deux 
  - candidat ecarte: Actifs pondérés en fonction des risques (modified, p.[65, 70])
- **bns-028** p.108 · Principaux risques non financiers — Dans la section des risques ESG, retrait de la mention que la banque est membre de l'alliance Net Zero (NZBA)
  - candidat ecarte: Rôle de la direction (removed, p.[87])

## RBC — non couverts (7)

- **rbc-003** p.66 · Accord de Bale III — Modification d'une note de bas de tableau pour retirer la mention de l'impact de l'adoption des lignes directrices revisees sur les normes de fonds propres et le ratio de levier
  - juge: Aucun candidat ne mentionne la modification d'une note de bas de tableau concernant l'impact des lignes directrices révisées sur les normes de fonds propres et le ratio de levier.
- **rbc-013** p.70 · Principaux risques — Retrait du tableau presentant un sommaire des principaux risques comme: les facteurs d'ordre economiques, strategiques, operationnel et transactionnel
  - juge: Aucun candidat ne mentionne le retrait d'un tableau présentant un sommaire des principaux risques.
- **rbc-016** p.71 · Conseil d'administration — Comite de gestion de risques: Ajout du role de supervision des groupes de conformites de lutte contre le financement des activites terroristes, de sanctions economiques mondiales et la corruption. Ret
  - juge: Aucun candidat ne traite spécifiquement du changement de rôle du comité de gestion des risques concernant la supervision des groupes de conformité et le retrait de la supervision de leur budget, mandats, ressources et structure organisationnelle.
- **rbc-017** p.71 · Conseil d'administration — Comite d'audit: Retrait du role sur la surveillance des autres documents d'informations, y compris sur la durabilite
  - juge: Aucun candidat ne mentionne le retrait du rôle du comité d'audit sur la surveillance des autres documents d'informations, y compris sur la durabilité.
- **rbc-019** p.71 · Conseil d'administration — Comite des ressources humaines: retrait des phrases liees a l'engagement des employes, la diversite et l'inclusion, et la sante et le bien-etre
  - juge: Aucun candidat ne traite du retrait des phrases concernant l'engagement des employés, la diversité et l'inclusion, et la santé et le bien-être dans le comité des ressources humaines.
- **rbc-028** p.74 · Architecture des documents de gestion du risque d'entreprise — Modifications du titre du graphique, passant de «Architecture de gestion du risque d'entreprise» a «Architecture des documents de gestion du risque d'entreprise»
  - juge: Aucun candidat ne traite du changement de titre du graphique dans la section 'Architecture des documents de gestion du risque d'entreprise'.
- **rbc-034** p.76 · Gestion des controles internes de gestion des risques — Modification du titre, passage de Surveillance de la gestion des risques et gestion des enjeux a Surveillance de la gestion des risques et gestion des enjeux
  - juge: Aucun candidat ne traite de la modification du titre de la sous-section mentionnée dans l'observation.

### RBC — detectes mais ecartes avant l'export (5)

- **rbc-007** p.67 · Actif pondere en fonction des risques au titre de Bale III — Ajout d'un paragraphe sur l'annonce du BSIF de reporter de facon indefini les augmentations du plancher de fonds propres exige par sa ligne directrice
  - candidat ecarte: Accord de Bâle III → Actif pondéré en fonction des risques au titre de Bâle III (modified, p.[125])
- **rbc-015** p.71 · Conseil d'administration — Modification de libelle: passage de «code de conduite» a «Code de deontologie»
  - candidat ecarte: CONSEIL D'ADMINISTRATION (modified, p.[77])
- **rbc-030** p.75 · Architecture des documents de gestion du risque d'entreprise — Mise a jour de certains de cadre de gestion des risques: passage de a Cadre de gestion du risque lie a la conduite et a la culture d'entreprise et pas
  - candidat ecarte: Risques liés à la culture et à la conduite (modified, p.[74, 84])
- **rbc-036** p.77 · Communication aux echelons superieurs des risques et des evenements — Modification du titre, passage de Communication aux echelons superieurs des risques et des enjeux lies aux evenements a Communication aux echelons sup
  - candidat ecarte: Communication aux échelons supérieurs des risques et des enjeux liés aux événements → Communication aux échelons supérieurs des risques et des événements (modified, p.[81])
- **rbc-037** p.77 · Communication aux echelons superieurs des risques et des evenements — Retrait de la mention des enjeux
  - candidat ecarte: Communication aux échelons supérieurs des risques et des enjeux liés aux événements → Communication aux échelons supérieurs des risques et des événements (modified, p.[81])

## BNC — non couverts (3)

- **bnc-024** Précisions sur l’incidence des normes environnementales sur l’octroi du crédit et sur ce que les analyses effectuées permettent de mesurer.
  - juge: Aucun bloc ne porte sur l'incidence des normes environnementales sur l'octroi du credit. La formulation «normes environnementales» est absente des deux extractions markdown: le libelle de l'analyste est un paraphrase non localisable.
- **bnc-037** Modification de la définition des risques environnementaux et sociaux et intégration de leurs effets sur les risques de crédit, de marché, de liquidité, de financement et opérationnels, ainsi que sur 
  - juge: Aucun bloc ne porte sur la definition des risques environnementaux et sociaux ni sur leurs effets sur les risques de credit, marche, liquidite, financement et operationnel.
- **bnc-040** Mise à jour des textes sur les faits nouveaux en matière de réglementation et introduction d’un nouveau texte. La description originale est incomplète.
  - juge: Description originale incomplete signalee par l'analyste; aucun candidat plausible, item non evaluable.

### BNC — detectes mais ecartes avant l'export (5)

- **bnc-004** Modification du texte sur le Cadre de capacité totale d’absorption des pertes applicable aux sociétés mères des banques d’importance systémique intéri
  - candidat ecarte: Accord de Bale (modified, p.[])
- **bnc-006** Ajout d’une précision sur le soutien apporté par la troisième ligne de défense à la promotion de la solidité financière à long terme de la BNC.
  - candidat ecarte: Integration de la gestion des risques a la culture (modified, p.[])
- **bnc-008** Ajout du terme « accessibilité » à la mission du Comité de ressources humaines.
  - candidat ecarte: Le comite de ressources humaines (modified, p.[])
- **bnc-009** Remplacement de « Expérience employé » par « Expérience et performance humaine ».
  - candidat ecarte: Le groupe de travail sur la surveillance des risques lies a la remuneration (modified, p.[])
- **bnc-017** Mise à jour des autres thèmes liés aux risques principaux et émergents.
  - candidat ecarte: Risques principaux et risques emergents (modified, p.[])
# Logique RAG/LLM - Changements communs entre banques

## Objectif

L'objectif est d'aider les equipes de divulgation et de gestion integree des risques a identifier les changements de divulgation adoptes par plusieurs banques, sans utiliser une logique deterministe de mots-cles.

La question cible est:

> Quels changements semantiquement similaires sont observes dans plusieurs banques, et lesquels sont adoptes par au moins 3 banques?

Cette analyse sert surtout a reperer les pratiques de marche: par exemple, plusieurs banques qui renforcent leur divulgation sur les tarifs douaniers, le risque climatique, le capital, la liquidite, le credit, la cybersecurite ou les exigences reglementaires.

## Position dans le pipeline

Cette logique est une etape globale post-batch.

Elle doit etre lancee apres les comparaisons individuelles des 6 banques pour une meme periode, lorsque les fichiers suivants existent deja:

- `outputs/resultats/<banque>/<periode>/text_comparison.json`
- eventuellement `outputs/resultats/<banque>/<periode>/comparison.json` pour les tableaux

Elle ne remplace pas les deux pipelines existants et ne change pas leur fonctionnement. Elle lit leurs resultats, produit un artefact global, puis l'interface validateur lit cet artefact sans appeler de LLM.

La periode est obligatoire pour l'analyse principale. Par exemple, pour `2025_t2_vs_2025_t1`, le systeme compare seulement les changements de cette fenetre:

- `outputs/resultats/bmo/2025_t2_vs_2025_t1/text_comparison.json`
- `outputs/resultats/bnc/2025_t2_vs_2025_t1/text_comparison.json`
- `outputs/resultats/bns/2025_t2_vs_2025_t1/text_comparison.json`
- `outputs/resultats/cibc/2025_t2_vs_2025_t1/text_comparison.json`
- `outputs/resultats/rbc/2025_t2_vs_2025_t1/text_comparison.json`
- `outputs/resultats/td/2025_t2_vs_2025_t1/text_comparison.json`

Le systeme ne doit pas melanger `2025_t2_vs_2025_t1` avec `2025_t3_vs_2025_t2`, sauf dans une analyse historique separee.

## Emplacement des resultats

Le resultat final doit etre conserve sous `outputs/resultats`, car il fait partie de la couche resultats analytiques.

Chemin recommande:

```text
outputs/resultats/changements_communs_banques/<periode>/changements_communs_banques.json
```

Exemple:

```text
outputs/resultats/changements_communs_banques/2025_t2_vs_2025_t1/changements_communs_banques.json
```

Nom interface:

```text
Changements communs entre banques
```

## Unite analysee

Le RAG ne doit pas indexer les rapports complets.

Chaque changement extrait par les comparaisons banque par banque devient une fiche atomique:

- banque
- periode comparee
- chemin du fichier source
- identifiant du changement
- section
- sous-section
- type de changement
- texte avant
- texte apres
- resume du changement
- niveau d'impact
- preuves textuelles disponibles

Cette granularite est importante: l'equipe ne veut pas seulement savoir qu'un rapport parle du climat ou du credit, elle veut savoir quel paragraphe a change, dans quelle section, et chez quelles banques.

## Logique LLM/RAG

Le traitement global suit ce flux:

1. Collecter tous les changements produits pour les 6 banques.
2. Transformer chaque changement en document RAG.
3. Utiliser des embeddings pour recuperer les changements candidats selon plusieurs angles metier.
4. Envoyer les candidats au LLM juge.
5. Demander au LLM de regrouper les changements semantiquement similaires.
6. Sauvegarder le resultat dans un JSON lisible par le validateur.

La decouverte ne doit pas dependre d'une seule requete. Pour une periode comme `2025_t2_vs_2025_t1`, le systeme doit explorer plusieurs themes:

- tarifs douaniers et incertitude commerciale;
- risque de credit et provisions;
- risque de marche, VAR et volatilite;
- fonds propres, levier, TLAC et actifs ponderes;
- liquidite et financement stable;
- climat, durabilite et exigences ESG;
- faits nouveaux reglementaires;
- risques emergents et gouvernance.

Le LLM juge doit evaluer:

- si le sujet de risque est le meme;
- si l'intention de divulgation est comparable;
- si le changement est de meme nature;
- si les sections sont fonctionnellement comparables meme avec des titres differents;
- si les preuves textuelles suffisent;
- si le seuil de banques distinctes est atteint.

Le seuil recommande est 3 banques distinctes ou plus.

## Garde-fou

Le LLM peut proposer un regroupement, mais le systeme ne doit pas lui faire confiance aveuglement pour le seuil.

Le statut final doit etre recalcule a partir des preuves, jamais accepte tel quel depuis le LLM:

- si 3, 4, 5 ou 6 banques distinctes sont presentes: `consensus_3_plus`;
- si exactement 2 banques distinctes sont presentes: `signal_mineur_2_banques`;
- si une seule banque est presente: exclu du rapport principal.

Cela evite qu'un signal interessant observe chez 2 banques soit presente comme une pratique de marche.

## Exemple concret attendu

Signal possible:

```text
Renforcement des divulgations sur les tarifs douaniers, l'incertitude commerciale et leurs effets sur le risque economique, le risque de credit et la volatilite des marches.
```

Banques regroupees:

- BNS
- TD
- CIBC
- RBC

Interpretation:

```text
Plusieurs banques canadiennes renforcent leur divulgation sur les tarifs douaniers et l'incertitude commerciale. Le sujet n'est pas toujours place au meme endroit: BNS l'isole dans une sous-section specifique, TD le rattache aux risques geopolitiques, CIBC l'encadre comme incertitude de politique commerciale, et RBC le relie aux pertes de credit. La pratique de marche semble evoluer vers une divulgation plus explicite des impacts financiers et operationnels des tensions commerciales.
```

Exemple de sortie JSON simplifiee:

```json
{
  "artifact_type": "changements_communs_banques",
  "schema_version": 1,
  "theme": "changements communs entre banques",
  "period": "2025_t2_vs_2025_t1",
  "analysis_scope": "single_period",
  "min_banks": 3,
  "source_stats": {
    "total_changes": 468,
    "bank_count": 6,
    "banks": ["bmo", "bnc", "bns", "cibc", "rbc", "td"]
  },
  "signal_counts": {
    "total": 4,
    "consensus": 3,
    "minor": 1
  },
  "signals": [
    {
      "theme": "Tarifs douaniers et incertitude commerciale",
      "status": "consensus",
      "bank_count": 4,
      "min_banks_met": true,
      "banks": ["bns", "td", "cibc", "rbc"],
      "business_summary": "Plusieurs banques renforcent leur divulgation sur les tensions commerciales et les effets possibles sur le credit, l'economie et les marches.",
      "why_grouped": "Les changements portent sur le meme choc externe et sur la meme intention de divulgation, meme si les sections different.",
      "evidence": [
        {
          "bank": "bns",
          "period": "2025_t2_vs_2025_t1",
          "section": "Gestion des risques",
          "subsection": "Incidence des tarifs douaniers",
          "change_type": "added",
          "source_path": "outputs/resultats/bns/2025_t2_vs_2025_t1/text_comparison.json",
          "change_id": "..."
        }
      ]
    }
  ]
}
```

## Integration interface

L'interface validateur ne doit pas executer le LLM.

Elle doit seulement:

- afficher un onglet `Changements communs entre banques`;
- lire `outputs/resultats/changements_communs_banques/<periode>/changements_communs_banques.json`;
- afficher une vue executive des signaux;
- separer les consensus des signaux mineurs a 2 banques;
- afficher une vue preuve avec banque, periode, section, sous-section, resume, preuve et fichier source.

Si le JSON n'existe pas encore pour la periode chargee, l'interface doit afficher un etat vide indiquant que l'analyse doit etre generee apres le batch des 6 banques pour cette periode.

## Benefice metier

Pour l'equipe divulgation, cette logique permet de voir rapidement si une banque est alignee avec les pairs ou si elle manque une pratique emergente.

Pour l'equipe gestion integree des risques, elle permet de distinguer:

- un changement isole;
- une tendance emergente;
- un consensus de marche;
- un sujet qui merite une revue de divulgation ou une discussion avec les equipes de risque.

# Exemples de texte détecté ou ignoré

| Cas | Exemple | Doit être détecté ? | Pourquoi |
|---|---|---:|---|
| Nouveau risque explicite | “La banque identifie désormais le risque de crime financier comme catégorie distincte.” | Oui | Nouvelle idée risque, potentiellement majeure |
| Nouvelle contrainte | “La banque devra maintenir un cadre plus strict de contrôle du levier.” | Oui | Nouvelle contrainte prudentielle ou de gestion |
| Nouvelle méthode de gestion | “La banque adopte une nouvelle approche pour mesurer certains risques.” | Oui | Changement méthodologique pertinent |
| Nouvelle nuance de risque | “Les risques futurs incluent désormais des facteurs politiques et réglementaires.” | Oui | Élargissement du périmètre de risque |
| Suppression d’un thème important | Un paragraphe clé sur une catégorie de risque disparaît | Oui | Retrait d’une idée potentiellement significative |
| Reformulation sans changement de fond | Même idée T1/T2, texte légèrement réécrit | Non | Ce n’est pas un vrai changement métier |
| Chiffres seuls qui bougent | Variation d’un ratio, d’un montant ou d’un % sans nouvelle idée | Non | Bruit quantitatif, pas nouveauté sémantique |
| Note de bas de page | Renvoi ou précision en bas de page | Non | Hors périmètre métier final |
| Note ou texte de tableau | Mention de détail issue d’un tableau ou d’une note tabulaire | Non | Le système doit l’exclure |
| Référence réglementaire brute | “Selon OSFI / Bâle / TLAC…” sans idée métier nouvelle | Non | Ce n’est qu’un indice, pas un changement final à remonter |

## Règle simple

Le système doit garder :

- une idée nouvelle ou vraiment modifiée
- qui change la lecture du risque, du capital, de la contrainte ou de la méthode

Le système doit exclure :

- rédactionnel
- notes
- tableaux
- chiffres seuls
- références réglementaires brutes
- texte générique sans nouveauté métier

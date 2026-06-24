Oui. Pour la vigie, je réduirais à ces deux sections par banque.

Pages ci-dessous = pages imprimées du rapport, plages inclusives.

**T4 2025**

| Banque | Priorité 1: Gestion des risques | Priorité 2: Capital / fonds propres |
|---|---:|---:|
| RBC | Gestion du risque: 74–124 | Gestion des fonds propres: 125–137 |
| BMO | Gestion globale des risques: 67–107 | Gestion globale du capital: 58–64 |
| BNC | Gestion des risques: 72–118 | Gestion du capital: 62–71 |
| BNS | Gestion du risque: 76–115 | Gestion du capital: 60–72 |
| CIBC | Gestion du risque: 42–81 | Gestion des fonds propres: 31–39 |
| TD | Facteurs de risque et gestion des risques: 82–126 | Situation des fonds propres: 73–78 |

**T4 2024**

| Banque | Priorité 1: Gestion des risques | Priorité 2: Capital / fonds propres |
|---|---:|---:|
| RBC | Gestion du risque: 72–123 | Gestion des fonds propres: 124–135 |
| BMO | Gestion globale des risques: 68–109 | Gestion globale du capital: 59–65 |
| BNC | Gestion des risques: 65–112 | Gestion du capital: 55–64 |
| BNS | Gestion du risque: 72–110 | Gestion du capital: 55–67 |
| CIBC | Gestion du risque: 45–84 | Gestion des fonds propres: 35–42 |
| TD | Facteurs de risque et gestion des risques: 84–127 | Situation des fonds propres: 75–80 |

Note importante: ton exemple RBC `72–123` correspond au T4 2024. En T4 2025, RBC décale à `74–124`.

**Vocabulaire par banque**

Les banques ne utilisent pas exactement le même vocabulaire. Elles couvrent les mêmes grands concepts, mais les titres changent selon la banque.

| Concept vigie | RBC | BMO | BNC | BNS | CIBC | TD |
|---|---|---|---|---|---|---|
| Risques | Gestion du risque | Gestion globale des risques | Gestion des risques | Gestion du risque | Gestion du risque | Facteurs de risque et gestion des risques |
| Capital | Gestion des fonds propres | Gestion globale du capital | Gestion du capital | Gestion du capital | Gestion des fonds propres | Situation des fonds propres |

Pour la vigie, il ne faut donc pas matcher uniquement un titre exact comme `Gestion du risque`. Il faut plutôt normaliser par concept.

| Concept normalisé | Titres possibles |
|---|---|
| `gestion_risques` | Gestion du risque; Gestion des risques; Gestion globale des risques; Facteurs de risque et gestion des risques |
| `capital` | Gestion des fonds propres; Gestion globale du capital; Gestion du capital; Situation des fonds propres |

La différence la plus importante est TD: elle mélange souvent les facteurs de risque et la gestion des risques dans une seule grande section. BMO ajoute souvent `globale`. RBC/CIBC préfèrent `fonds propres`, alors que BMO/BNC/BNS utilisent plutôt `capital`.

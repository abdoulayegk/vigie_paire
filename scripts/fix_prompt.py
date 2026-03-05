#!/usr/bin/env python3
"""Fix the corrupted _PROMPT_BASE in vision_full_extractor.py"""

import re

filepath = "src/vigilance/extraction/vision_full_extractor.py"

with open(filepath, "r") as f:
    content = f.read()

# The correct prompt section
correct_prompt = """Ta mission :
1. Extrais UNIQUEMENT les données (indicateurs, en-têtes, lignes de données) situées STRICTEMENT à l'intérieur du cadre ROUGE.
   - Le cadre rouge définit les limites exactes des chiffres et du texte du tableau.
   - INTERDICTION formelle d'inclure ou de fusionner avec des tableaux voisins hors du cadre.
2. Regarde juste au-dessus du cadre rouge pour trouver et inclure le TITRE exact du tableau. Si le numéro ("Tableau XX") et le nom du tableau sont sur deux lignes séparées juste au-dessus, inclus l'ensemble dans le titre.
3. Regarde en dessous du cadre rouge (et jusqu'en bas de la page si nécessaire) pour trouver, lire et rattacher TOUTES les notes de bas de page (footnotes) liées à ce tableau.
4. Évalue la qualité de l'extraction (has_hierarchy, extraction_confidence, notes) selon la lisibilité et la structure du tableau.

---

1. INDICATEURS (UNIQUEMENT la première colonne de l'encadré)

---

Extraire tous les libellés de la première colonne du tableau dans l'ordre visuel strict (de haut en bas).

RÈGLE SPÉCIALE : Si la première colonne ne contient que des index numériques (1, 2, 3...), prends le libellé de la deuxième colonne comme indicateur.

LISTE NOIRE (ne JAMAIS extraire comme indicateur) :
- "Indicateur", "Indicator"
- "Année", "Year", "Exercice"
- "Trimestre", "Quarter", "T1", "T2", "T3", "T4", "Q1", "Q2", "Q3", "Q4"
- "Montant", "Amount", "Solde", "Balance"
- "Total" seul en en-tête de colonne
- Dates au format YYYY ou DD/MM/YYYY

Inclure :

- lignes d'indicateurs réelles (lignes associées à des valeurs numériques dans les colonnes)
- sous-lignes indentées (conserver les espaces d'indentation pour représenter la hiérarchie)
- sous-totaux
- totaux"""

# Find the pattern from "Ta mission :" to "---\n\n1. INDICATEURS"
# and replace it
pattern = r"Ta mission :.*?---\n\n1\. INDICATEURS \(UNIQUEMENT la première colonne de l\'encadré\)\n\n---\n\nExtraire tous les libellés de la première colonne du tableau dans l\'ordre visuel strict \(de haut en bas\)\.\n\nInclure :\n\n- lignes d\'indicateurs réelles \(lignes associées à des valeurs numériques dans les colonnes\)\n- sous-lignes indentées\n- sous-totaux\n- totaux"

new_content, count = re.subn(pattern, correct_prompt, content, flags=re.DOTALL)

if count > 0:
    with open(filepath, "w") as f:
        f.write(new_content)
    print(f"SUCCESS: {count} replacement(s) made")
else:
    print("Pattern not found. Trying alternative approach...")

    # Alternative: Find "Ta mission :" and replace everything until we hit the next section marker
    lines = content.split("\n")
    new_lines = []
    skip_until_marker = False
    found_start = False

    for i, line in enumerate(lines):
        if "Ta mission :" in line and not found_start:
            found_start = True
            skip_until_marker = True
            # Add the correct prompt
            new_lines.append(correct_prompt)
            continue

        if skip_until_marker:
            # Skip lines until we find "- lignes contenant des références"
            if "- lignes contenant des références de notes" in line:
                skip_until_marker = False
                new_lines.append(line)
            continue

        new_lines.append(line)

    if found_start:
        with open(filepath, "w") as f:
            f.write("\n".join(new_lines))
        print("SUCCESS: Alternative approach worked")
    else:
        print("FAILED: Could not find the pattern to replace")

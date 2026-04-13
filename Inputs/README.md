# 📂 Dossier `Inputs/` — Convention de Nomenclature

Ce dossier contient les rapports PDF des banques, organisés par **banque** puis par **année**.

## Structure Obligatoire

```
Inputs/
├── BNC/
│   └── 2025/
│       ├── BNC_2025_T1.pdf
│       └── BNC_2025_T2.pdf
├── RBC/
│   └── 2025/
│       ├── RBC_2025_T1.pdf
│       └── RBC_2025_T2.pdf
├── TD/
│   └── 2025/
│       ├── TD_2025_T1.pdf
│       └── TD_2025_T2.pdf
├── BMO/
├── BNS/
└── CIBC/
```

## Convention de Nommage des PDFs

**Format strict : `{BANQUE}_{ANNÉE}_{TRIMESTRE}.pdf`**

| Champ | Valeurs | Exemple |
|---|---|---|
| `{BANQUE}` | BNC, RBC, TD, BMO, BNS, CIBC | `BNC` |
| `{ANNÉE}` | 2024, 2025, … | `2025` |
| `{TRIMESTRE}` | T1, T2, T3, T4 | `T2` |

**Exemple complet :** `BNC_2025_T2.pdf`

## Utilisation

```bash
uv run python run_pipeline.py --bank BNC --year 2025 --quarter T2
```

```bash
uv run python run_pipeline.py --bank TD --year 2026 --quarter T1 --skip-extraction
```

Le pipeline trouvera automatiquement :
- **Courant :**  `Inputs/BNC/2025/BNC_2025_T2.pdf`
- **Précédent :** `Inputs/BNC/2025/BNC_2025_T1.pdf` (déduit automatiquement)

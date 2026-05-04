# Guide de contribution — Vigie de Paire

Conventions à respecter pour contribuer au projet.

---

## Stratégie de branches

| Branche      | Usage                                            |
| ------------ | ------------------------------------------------ |
| `main`       | Code stable, déployable en production            |
| `feat/*`     | Nouvelle fonctionnalité                          |
| `fix/*`      | Correction de bug                                |
| `refactor/*` | Refactorisation sans changement de comportement  |
| `chore/*`    | Mise à jour de dépendances, configuration        |

Créer une branche depuis `main` avant tout travail :

```bash
git checkout main && git pull
git checkout -b feat/nom-de-la-fonctionnalite
```

---

## Hooks pre-commit

Installer les hooks après `uv sync --group dev` :

```bash
uv run pre-commit install
```

Lancer tous les hooks localement avant une PR :

```bash
uv run pre-commit run --all-files
```

Les hooks couvrent les contrôles de fichiers de base, `ruff check src/` et le scan sécurité statique `bandit`.

---

## Convention de commits

Format : `type: description courte en français`

```text
feat: ajouter le tri par priorité dans la file d'attente
fix: corriger le mismatch tableau/preuves visuelles
refactor: extraire _resolve_selection dans un module dédié
chore: mettre à jour pydantic vers 2.6
docs: compléter le guide d'installation Windows
```

Types valides : `feat` · `fix` · `refactor` · `chore` · `docs` · `test`

---

## Pull Requests

1. Ouvrir la PR vers `main`
2. Titre court en français au format `type: description`
3. Description incluant :
   - Le problème résolu ou la fonctionnalité ajoutée
   - Les fichiers principaux modifiés
   - Comment tester manuellement
4. Assigner un relecteur avant de merger

---

## Ce qu'il ne faut pas faire

- Ne jamais commiter le fichier `.env` (il est dans `.gitignore`)
- Ne pas modifier `configs/bank_profiles.yaml` sans documenter le changement dans la PR
- Ne pas ajouter de dépendances sans les justifier dans la PR
- Ne pas bypasser les hooks pre-commit (`--no-verify`)

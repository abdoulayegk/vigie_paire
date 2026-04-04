# Convention docstrings -- Vigie Paire

Style **Google** avec descriptions en **francais**.

Les mots-cles de section (`Args`, `Returns`, `Raises`, `Attributes`) restent en
anglais (compatibilite ruff, IDE, Sphinx).

## Fonction

```python
def ma_fonction(param1: str, param2: int = 0) -> dict[str, Any]:
    """Description concise terminee par un point.

    Contexte metier ou technique additionnel si necessaire.

    Args:
        param1: Description du parametre.
        param2: Description. Par defaut 0.

    Returns:
        Description du retour.

    Raises:
        ValueError: Condition de l'erreur.
    """
```

## Classe

```python
class MaClasse:
    """Description concise de la classe.

    Role et contexte metier.

    Attributes:
        attr1: Description.
    """
```

## Module (en-tete de fichier)

```python
"""Description du module en une ligne.

Description plus longue du role du module, de son positionnement
dans l'architecture et de ses responsabilites principales.
"""
```

## Regles specifiques

- **Fonctions privees** (`_prefix`) : docstring d'une ligne minimum.
  `Args`/`Returns` seulement si la signature n'est pas evidente.
- **Pydantic pour OpenAI** : docstring de classe en francais.
  Ne PAS toucher les `Field(description=...)` (envoyes a l'API).
- **Dataclasses / Pydantic internes** : `Attributes:` pour les champs non
  evidents uniquement.
- **Enums** : docstring d'une ligne en francais.
- **Callbacks Dash** : documenter les inputs/outputs Dash et l'effet metier.
- **Prompts LLM** : ne pas documenter les chaines de prompt.
- **Pas de docstring triviale** : ne pas ecrire `"""Retourne True."""` quand
  le nom de la fonction est deja explicite.

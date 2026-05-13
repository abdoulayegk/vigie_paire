# Convention des docstrings -- Vigie de paire

Format compatible Sphinx, avec descriptions en **français**.

Les mots-clés techniques de section (`Args`, `Returns`, `Raises`, `Attributes`) restent en
anglais pour conserver la compatibilité avec Ruff, les IDE et Sphinx. Le contenu
des descriptions doit rester en français.

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

## Règles spécifiques

- **Fonctions privées** (`_prefix`) : docstring d'une ligne minimum.
  `Args`/`Returns` seulement si la signature n'est pas évidente.
- **Pydantic pour OpenAI** : docstring de classe en français.
  Ne PAS toucher les `Field(description=...)` (envoyés à l'API).
- **Dataclasses / Pydantic internes** : `Attributes:` pour les champs non
  évidents uniquement.
- **Enums** : docstring d'une ligne en français.
- **Rappels Dash** : documenter les entrées, les sorties et l'effet métier.
- **Consignes LLM** : ne pas documenter les chaînes de consigne.
- **Pas de docstring triviale** : ne pas écrire `"""Retourne True."""` quand
  le nom de la fonction est déjà explicite.

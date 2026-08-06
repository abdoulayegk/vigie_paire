"""Utilitaires de normalisation du JSON Schema Pydantic pour les Structured Outputs OpenAI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel


def build_strict_openai_response_format(
    model: type[BaseModel],
    *,
    name: str,
    error_cls: type[Exception] = RuntimeError,
) -> dict[str, Any]:
    """Construit un payload ``response_format`` OpenAI a partir d'un modele Pydantic.

    Le mode strict des Structured Outputs OpenAI exige que chaque noeud objet :
    - declare ``additionalProperties: false``
    - liste chaque propriete dans ``required`` meme si Pydantic les marque optionnelles

    La sortie brute de ``model_json_schema()`` de Pydantic ne satisfait pas toujours
    ce contrat ; on normalise donc ici une seule fois et on reutilise la meme logique partout.

    Args:
        model: Classe Pydantic a convertir en schema JSON.
        name: Nom du schema pour le payload OpenAI.
        error_cls: Classe d'exception a lever en cas d'erreur de validation.

    Returns:
        Dictionnaire ``response_format`` pret a etre envoye a l'API OpenAI.

    Raises:
        error_cls: Si le schema contient des objets map-like interdits en mode strict.
    """
    schema = deepcopy(model.model_json_schema())
    _normalize_object_nodes_for_openai(schema, path="$", error_cls=error_cls)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def validate_strict_openai_response_format(
    response_format: dict[str, Any],
    *,
    error_cls: type[Exception] = RuntimeError,
) -> None:
    """Valide un payload ``response_format`` OpenAI strict avant les appels API.

    Args:
        response_format: Payload ``response_format`` a valider.
        error_cls: Classe d'exception a lever en cas d'erreur.

    Raises:
        error_cls: Si le schema est malformed ou si required/properties ne correspondent pas.
    """
    try:
        if response_format.get("type") != "json_schema":
            raise error_cls("schema.type must be 'json_schema'")
        json_schema = response_format["json_schema"]
        schema = json_schema["schema"]
        properties = schema["properties"]
        required = schema["required"]
    except Exception as exc:
        if isinstance(exc, error_cls):
            raise
        raise error_cls(f"schema malformed for Structured Outputs: {exc}") from exc

    if not isinstance(properties, dict):
        raise error_cls("schema.properties must be a dict")
    if not isinstance(required, list):
        raise error_cls("schema.required must be a list")

    property_keys = set(properties.keys())
    required_keys = {str(key) for key in required}
    if property_keys != required_keys:
        missing = sorted(property_keys - required_keys)
        extra = sorted(required_keys - property_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing_in_required={missing}")
        if extra:
            details.append(f"unknown_in_required={extra}")
        joined = ", ".join(details) if details else "required/properties mismatch"
        raise error_cls(
            f"Structured Outputs strict contract invalid: required must exactly match properties ({joined})"
        )

    _validate_no_map_like_objects(schema, path="$", error_cls=error_cls)


def _normalize_object_nodes_for_openai(
    node: Any,
    *,
    path: str,
    error_cls: type[Exception],
) -> None:
    """Normalise recursivement les noeuds objet pour le mode strict OpenAI."""
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type == "object":
        additional_properties = node.get("additionalProperties")
        if additional_properties not in (None, False):
            raise error_cls(f"Structured Outputs strict contract invalid: map-like object not allowed at {path}")
        node["additionalProperties"] = False

        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            for key, sub_schema in properties.items():
                _normalize_object_nodes_for_openai(
                    sub_schema,
                    path=f"{path}.properties.{key}",
                    error_cls=error_cls,
                )
        else:
            node["required"] = []

    if node_type == "array":
        _normalize_object_nodes_for_openai(
            node.get("items"),
            path=f"{path}.items",
            error_cls=error_cls,
        )

    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for index, sub_schema in enumerate(variants):
                _normalize_object_nodes_for_openai(
                    sub_schema,
                    path=f"{path}.{key}[{index}]",
                    error_cls=error_cls,
                )

    defs = node.get("$defs")
    if isinstance(defs, dict):
        for key, sub_schema in defs.items():
            _normalize_object_nodes_for_openai(
                sub_schema,
                path=f"{path}.$defs.{key}",
                error_cls=error_cls,
            )


def _validate_no_map_like_objects(
    node: Any,
    *,
    path: str,
    error_cls: type[Exception],
) -> None:
    """Valide recursivement qu'aucun noeud objet n'est de type map-like."""
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type == "object":
        if node.get("additionalProperties") not in (False, None):
            raise error_cls(f"Structured Outputs strict contract invalid: map-like object not allowed at {path}")
        properties = node.get("properties")
        if isinstance(properties, dict):
            for key, sub_schema in properties.items():
                _validate_no_map_like_objects(
                    sub_schema,
                    path=f"{path}.properties.{key}",
                    error_cls=error_cls,
                )

    if node_type == "array":
        _validate_no_map_like_objects(
            node.get("items"),
            path=f"{path}.items",
            error_cls=error_cls,
        )

    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for index, sub_schema in enumerate(variants):
                _validate_no_map_like_objects(
                    sub_schema,
                    path=f"{path}.{key}[{index}]",
                    error_cls=error_cls,
                )

    defs = node.get("$defs")
    if isinstance(defs, dict):
        for key, sub_schema in defs.items():
            _validate_no_map_like_objects(
                sub_schema,
                path=f"{path}.$defs.{key}",
                error_cls=error_cls,
            )

"""Helpers to normalize Pydantic JSON Schema for OpenAI Structured Outputs."""

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
    """Return an OpenAI ``response_format`` payload from a Pydantic model.

    OpenAI Structured Outputs strict mode requires every object node to:
    - declare ``additionalProperties: false``
    - list every property in ``required`` even when Pydantic marks defaults optional

    Pydantic's raw ``model_json_schema()`` output does not always satisfy that
    contract, so we normalize it once here and reuse the same logic everywhere.
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
    """Validate a strict OpenAI ``response_format`` payload before API calls."""
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
        raise error_cls(
            f"schema malformed for Structured Outputs: {exc}"
        ) from exc

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
            "Structured Outputs strict contract invalid: "
            f"required must exactly match properties ({joined})"
        )

    _validate_no_map_like_objects(schema, path="$", error_cls=error_cls)


def _normalize_object_nodes_for_openai(
    node: Any,
    *,
    path: str,
    error_cls: type[Exception],
) -> None:
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type == "object":
        additional_properties = node.get("additionalProperties")
        if additional_properties not in (None, False):
            raise error_cls(
                "Structured Outputs strict contract invalid: "
                f"map-like object not allowed at {path}"
            )
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
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type == "object":
        if node.get("additionalProperties") not in (False, None):
            raise error_cls(
                "Structured Outputs strict contract invalid: "
                f"map-like object not allowed at {path}"
            )
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

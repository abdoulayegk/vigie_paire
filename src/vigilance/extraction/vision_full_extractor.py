"""Vision-based full extraction: indicators (first column) + footnotes in one GPT-4o call.

Quality-first: supports multi-pass (re-crop with extended bottom if confidence low),
retry on invalid JSON. Used as primary content source when vision_extraction.enabled.

Uses Pydantic validation and OpenAI Structured Outputs (json_schema) when available.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.genai import get_openai_api_key
from .vision_cache import (
    cache_get,
    cache_put,
    get_vision_cache_dir,
    make_cache_key,
)

logger = logging.getLogger(__name__)

_EXTRACTION_METHOD = "vision_full_gpt4o"

_CONFIDENCE_RETRY_THRESHOLD = 0.85
_RECROP_EXTENSION_INCREMENT = 0.06
_DEFAULT_MAX_COMPLETION_TOKENS_FULL = 16384
_DEFAULT_MAX_COMPLETION_TOKENS_LEAN = 8192
_DEFAULT_REFERENCE_TEXT_MAX_CHARS = 4000

_PROMPT_BASE = """
Tu es un expert en extraction de données financières à partir de rapports bancaires canadiens.

TÂCHE
On te fournit l'image RECADRÉE d'un tableau financier extrait d'un rapport bancaire canadien.
L'image montre le tableau ciblé et peut inclure un petit contexte au-dessus (titre) et en dessous (notes de bas de page).

Ta mission :
1. Extrais TOUTES les données (indicateurs, en-têtes, lignes de données) visibles dans l'image.
   - INTERDICTION formelle d'inventer des lignes ou des données non visibles.
2. Si le TITRE du tableau est visible en haut de l'image (numéro "Tableau XX" et/ou nom), inclus-le.
3. Si des notes de bas de page sont visibles en bas de l'image, lis-les et rattache-les au tableau.

---

1. INDICATEURS (première colonne du tableau)

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
- totaux
- lignes contenant des références de notes comme (1), (2),(1)(2), *, †

Exclure :

- marqueurs de notes isolés (ex: (1), *, 1, 2,¹, ², ³) s'ils ne sont pas rattachés à un libellé textuel
- symboles monétaires seuls (ex: $)
- titres du tableau
- en-têtes de colonnes
- titres de groupes ou sections (lignes sans valeurs associées)
- années ou périodes (ex : 2024, 2025, T1)
- unités (ex : %, en millions, en milliards)
- valeurs numériques
- notes de bas de page

Règles importantes :

- conserver EXACTEMENT le texte visible
- conserver l'indentation si visible
- ne jamais modifier ou interpréter le texte
- respecter strictement l'ordre visuel
- ne jamais inventer d'indicateur

RÈGLE ANTI-FUSION (CRITIQUE) :
- Chaque ligne visuelle distincte du tableau doit produire UN indicateur séparé.
- Si deux lignes de texte sont proches verticalement mais clairement séparées, crée DEUX objets indicateurs distincts.
- Ne JAMAIS concaténer ou fusionner plusieurs lignes en un seul indicateur.
- En cas de doute sur la séparation, privilégie la création d'indicateurs séparés.

EXCLUSION DES PARAGRAPHES NARRATIFS :
- Ne pas extraire les blocs de texte explicatif (phrases complètes qui traversent plusieurs colonnes).
- Les paragraphes descriptifs ou les notes intégrées au milieu du tableau ne sont PAS des indicateurs.
- Un indicateur est typiquement un libellé court (1 à 10 mots) aligné dans la première colonne.

---

2. NOTES DE BAS DE TABLEAU (FOOTNOTES)

---

Extraire toutes les notes situées en bas de l'image recadrée du tableau (jusqu'au bas de l'image).

Formats possibles des marqueurs :

- (1) (2) (3)(4)
- ¹ ² ³
- 1 2 3
- * † ‡
- a) b)

IMPORTANT :

Les notes doivent être retournées dans leur ordre visuel exact (de haut en bas).
Ne jamais trier les notes par identifiant.

Pour chaque note retourner :

- id : identifiant normalisé (ex : "1", "2", "*")
- text : texte complet de la note

Règles :

- conserver le texte EXACT
- ne pas fusionner plusieurs notes
- ne pas inventer de notes
- respecter l'ordre visuel
- si aucune note n'est visible retourner une liste vide

---

REGLES GENERALES

- Transcrire uniquement ce qui est visible dans l'image
- Ne jamais inventer d'information
- Respecter l'ordre visuel

Retourner également :

- confidence : score global entre 0.0 et 1.0 basé sur la lisibilité
"""

_PROMPT_JSON_STRICT = """
REPONSE JSON STRICTE.
Retourner uniquement du JSON valide.
Aucun texte avant ou après.

L'objet JSON doit rigoureusement suivre cette structure:

{
"table_title": "Tableau 1 - Titre complet ou chaine vide si absent",
"headers": ["Colonne 1", "Colonne 2", "Colonne 3"],
"indicators": ["Libelle 1", " Sous-libelle", "Total"],
"rows": [
  ["Libelle 1", "100", "200"],
  [" Sous-libelle", "50", "150"]
],
"footnotes_content": [
  {"id": "1", "text": "texte note 1"},
  {"id": "2", "text": "texte note 2"}
],
"footnote_markers": ["1", "2"],
"has_hierarchy": true,
"extraction_confidence": "high",
"notes": "Tableau bien cadré",
"confidence": 0.95,
"appears_truncated": false,
"estimated_content_height": null
}

REGLES DE VALIDATION

- table_title : inclure le numéro ("Tableau XX") ET le titre s'ils sont présents en haut de l'image. Chaine vide si aucun titre visible (NE JAMAIS inventer).
- headers : liste vide si aucun en-tete visible
- indicators doit respecter l'ordre visuel du tableau
- rows : liste vide si aucune donnee visible
- footnotes_content doit respecter l'ordre visuel des notes (haut → bas)
- ne jamais trier les notes par identifiant
- si aucune note n'est visible :
  footnotes_content = []

DEFINITIONS

table_title
Titre complet et visible du tableau, incluant le numéro (ex: "Tableau 1") s'il est présent sur la même ligne ou la ligne juste au-dessus. "" si absent.

headers
Liste des en-tetes de colonnes.

indicators
Liste des libellés extraits de la première colonne.

rows
Liste de listes de chaînes : toutes les lignes de données du tableau.

footnotes_content
Liste ORDONNEE des notes (ordre visuel strict, haut → bas).

footnote_markers
Liste simple des marqueurs détectés dans le tableau.

has_hierarchy
true si le tableau utilise l'indentation ou des sous-catégories.

extraction_confidence
"high" (lisible et structuré), "medium" (lisible mais doutes), "low" (illisible ou cassé).

notes
Brefs commentaires sur la qualité de l'image ou du tableau (ex: "flou", "coupé", "ok").

confidence
Score numérique (0.0 - 1.0) basé sur la lisibilité globale.

appears_truncated
true si le tableau ou les footnotes semblent coupés.

estimated_content_height
Pourcentage estimé du contenu visible.
Mettre null si impossible à estimer.
"""

_PROMPT_JSON_LEAN = """
MODE LEAN

Priorise la fiabilité du JSON et les champs nécessaires à la comparaison.
Ne retourne PAS le champ `rows`.
Ne retourne PAS de champs additionnels.

L'objet JSON doit rigoureusement suivre cette structure:

{
"table_title": "Tableau 1 - Titre complet ou chaine vide si absent",
"headers": ["Colonne 1", "Colonne 2", "Colonne 3"],
"indicators": ["Libelle 1", " Sous-libelle", "Total"],
"footnotes_content": [
  {"id": "1", "text": "texte note 1"},
  {"id": "2", "text": "texte note 2"}
],
"footnote_markers": ["1", "2"],
"confidence": 0.95,
"appears_truncated": false,
"estimated_content_height": null
}

REGLES DE VALIDATION

- table_title : inclure le numéro ("Tableau XX") ET le titre s'ils sont présents en haut de l'image. Chaine vide si aucun titre visible.
- headers : liste vide si aucun en-tete visible
- indicators : ordre visuel strict haut → bas
- footnotes_content : ordre visuel strict haut → bas
- footnote_markers : liste simple des marqueurs détectés
- confidence : score numérique 0.0 - 1.0
- appears_truncated : true si le contenu semble coupé
- estimated_content_height : pourcentage estimé du contenu visible, ou null
"""


class VisionFootnoteItem(BaseModel):
    """Strict item schema for one footnote entry."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Marqueur de note (ex: 1, (1), a)")
    text: str = Field(description="Texte de la note")

    @field_validator("id", "text", mode="before")
    @classmethod
    def _coerce_non_empty_str(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("must be non-empty")
        return s


class VisionIndicatorItem(BaseModel):
    """Strict item schema for one indicator with spatial coordinates."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Libellé exact de l'indicateur")
    bbox: list[float] | None = Field(
        default=None,
        description="Coordonnées normalisées [x_min, y_min, x_max, y_max] (0-1), null si incertain",
    )

    @field_validator("text", mode="before")
    @classmethod
    def _coerce_non_empty_str(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("indicator text must be non-empty")
        return s


class VisionResponseCommonSchema(BaseModel):
    """Common schema for full and fallback Vision responses."""

    model_config = ConfigDict(extra="forbid")

    table_title: str = Field(
        default="",
        description="Titre complet et visible du tableau, incluant le numéro (ex: 'Tableau 1') s'il est au-dessus. Chaine vide si aucun titre visible.",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="En-tetes de colonnes du tableau, dans l'ordre",
    )
    indicators: list[str] = Field(
        description="Libelles de la premiere colonne, ordre visuel haut vers bas",
    )
    footnotes_content: list[VisionFootnoteItem] = Field(
        description="Liste ORDONNEE de notes structurees [{id, text}] — ordre visuel strict",
        default_factory=list,
    )
    footnote_markers: list[str] = Field(
        description="Liste des marqueurs detectes (1, 2, 3 ou format parenthesique)",
        default_factory=list,
    )
    confidence: float = Field(
        description="Score numérique 0.0-1.0 de confiance globale.",
        ge=0.0,
        le=1.0,
    )
    appears_truncated: bool = Field(
        default=False,
        description="Si le contenu semble coupe (crop trop court)",
    )
    estimated_content_height: int | None = Field(
        default=None,
        description="Hauteur estimee du contenu visible en pourcentage (0-100)",
        ge=0,
        le=100,
    )

    @field_validator("indicators", mode="before")
    @classmethod
    def _coerce_indicators(cls, v: Any) -> list[str]:
        """Accept both string and legacy object indicator formats."""
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    result.append(text)
            elif isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    result.append(text)
            elif isinstance(item, VisionIndicatorItem):
                result.append(item.text)
        return result

    @field_validator("headers", mode="after")
    @classmethod
    def _normalize_headers(cls, v: list[str]) -> list[str]:
        return [str(x).strip() for x in v]

    @field_validator("footnotes_content", mode="before")
    @classmethod
    def _coerce_footnotes_content(cls, v: Any) -> list[dict[str, str]]:
        # Migration shim: accept legacy dict marker->text and normalize to ordered list.
        # The dict form loses visual order — items are added in insertion order.
        if isinstance(v, dict):
            out: list[dict[str, str]] = []
            for k, val in v.items():
                marker = str(k).strip()
                text = str(val).strip()
                if marker and text:
                    out.append({"id": marker, "text": text})
            return out
        if isinstance(v, list):
            out = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                marker = str(
                    item.get("id") or item.get("marker") or item.get("ref") or ""
                ).strip()
                text = str(item.get("text") or item.get("value") or "").strip()
                if marker and text:
                    out.append({"id": marker, "text": text})
            return out
        return []

    @field_validator("footnote_markers", mode="after")
    @classmethod
    def _normalize_footnote_markers(cls, v: list[str]) -> list[str]:
        return [str(x).strip() for x in v if str(x).strip()]


class VisionFullResponseSchema(VisionResponseCommonSchema):
    """Strict schema for normal Vision extraction output."""

    rows: list[list[str]] = Field(
        default_factory=list,
        description="Lignes de donnees du tableau (liste de listes de chaines)",
    )
    has_hierarchy: bool = Field(
        description="True si le tableau contient des sous-catégories indentées ou une structure hiérarchique explicite.",
        default=False,
    )
    extraction_confidence: str = Field(
        description="Niveau de confiance qualitatif ('high', 'medium', 'low').",
        default="medium",
        pattern="^(high|medium|low)$",
    )
    notes: str = Field(
        description="Observations pertinentes sur la qualité ou la structure (ex: 'flou', 'ratures', 'colonnes décalées').",
        default="",
    )

    @field_validator("rows", mode="before")
    @classmethod
    def _coerce_rows(cls, v: Any) -> list[list[str]]:
        if not isinstance(v, list):
            return []
        result: list[list[str]] = []
        for row in v:
            if isinstance(row, list):
                result.append([str(cell) for cell in row])
            elif isinstance(row, str):
                result.append([row])
        return result


class VisionLeanResponseSchema(VisionResponseCommonSchema):
    """Strict schema for lean fallback extraction output."""


class VisionSchemaContractError(RuntimeError):
    """Raised when OpenAI Structured Outputs schema contract is invalid."""


def _schema_model_for_mode(mode: str) -> type[BaseModel]:
    if mode == "lean":
        return VisionLeanResponseSchema
    return VisionFullResponseSchema


def _build_openai_json_schema(mode: str = "full") -> dict[str, Any]:
    """Build OpenAI json_schema format from Pydantic model for Structured Outputs."""
    schema_model = _schema_model_for_mode(mode)
    schema = schema_model.model_json_schema()
    props = schema.get("properties", {})
    defs = schema.get("$defs", {})
    required = list(props.keys())
    strict_schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
    if isinstance(defs, dict) and defs:
        # OpenAI strict mode: every object must have required == all properties.
        # Pydantic omits fields with defaults from required — fix that recursively.
        for def_name, def_schema in defs.items():
            if isinstance(def_schema, dict) and def_schema.get("type") == "object":
                def_props = def_schema.get("properties", {})
                if isinstance(def_props, dict):
                    def_schema["required"] = list(def_props.keys())
        strict_schema["$defs"] = defs

    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"vision_{mode}_extraction",
            "strict": True,
            "schema": strict_schema,
        },
    }


def _validate_openai_strict_schema_contract(schema: dict[str, Any]) -> None:
    """Validate local Structured Outputs strict contract before API call."""
    try:
        if schema.get("type") != "json_schema":
            raise VisionSchemaContractError("schema.type must be 'json_schema'")
        json_schema = schema["json_schema"]
        block = json_schema["schema"]
        properties = block["properties"]
        required = block["required"]
    except Exception as exc:
        raise VisionSchemaContractError(
            f"schema malformed for Structured Outputs: {exc}"
        ) from exc

    if not isinstance(properties, dict):
        raise VisionSchemaContractError("schema.properties must be a dict")
    if not isinstance(required, list):
        raise VisionSchemaContractError("schema.required must be a list")

    prop_keys = set(properties.keys())
    req_keys = {str(k) for k in required}
    if prop_keys != req_keys:
        missing = sorted(prop_keys - req_keys)
        extra = sorted(req_keys - prop_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing_in_required={missing}")
        if extra:
            details.append(f"unknown_in_required={extra}")
        joined = ", ".join(details) if details else "required/properties mismatch"
        raise VisionSchemaContractError(
            "Structured Outputs strict contract invalid: "
            f"required must exactly match properties ({joined})"
        )
    _validate_no_map_like_objects(block, path="$")


def _validate_no_map_like_objects(node: Any, path: str) -> None:
    if not isinstance(node, dict):
        return
    node_type = node.get("type")
    if node_type == "object":
        # OpenAI strict schema handling is fragile with map-like additionalProperties objects.
        if "additionalProperties" in node and node.get("additionalProperties") not in (
            False,
            None,
        ):
            raise VisionSchemaContractError(
                f"Structured Outputs strict contract invalid: map-like object not allowed at {path}"
            )
        props = node.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                _validate_no_map_like_objects(sub, f"{path}.properties.{key}")
    if node_type == "array":
        _validate_no_map_like_objects(node.get("items"), f"{path}.items")
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for idx, sub in enumerate(variants):
                _validate_no_map_like_objects(sub, f"{path}.{key}[{idx}]")
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for key, sub in defs.items():
            _validate_no_map_like_objects(sub, f"{path}.$defs.{key}")


def _classify_openai_error(exc: Exception) -> str:
    """Classify OpenAI API error to choose deterministic handling."""
    msg = str(exc).lower()
    if "invalid schema for response_format" in msg or (
        "missing '" in msg and "response_format" in msg and "required" in msg
    ):
        return "schema_contract_invalid"
    if (
        "json_schema" in msg
        and "response_format" in msg
        and ("not supported" in msg or "unsupported" in msg)
    ):
        return "structured_output_unsupported"
    if "rate" in msg and "limit" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "connect" in msg or "network" in msg:
        return "connection"
    return "other"


@dataclass
class VisionFullResult:
    """Result of Vision full extraction (full table content + footnotes).

    Source of truth for ALL content fields when Vision extracts a table.
    footnotes_content is an ORDERED LIST preserving visual order (haut → bas).
    No content field is ever backfilled from Docling when Vision is used.
    """

    # Content fields — all come from Vision, never from Docling
    table_title: str
    headers: list[str]
    indicators: list[dict[str, Any]]
    rows: list[list[str]]
    # footnotes as ordered list (visual order preserved, never sorted by marker)
    footnotes_content: list[dict[str, str]]
    footnote_markers: list[str]
    confidence: float
    extraction_method: str = _EXTRACTION_METHOD
    appears_truncated: bool = False
    estimated_content_height: int | None = None
    # Status and warnings
    vision_status: str = "ok"  # "ok" | "partial" | "failed"
    warnings: list[str] = field(default_factory=list)
    # Recrop quality pass (for debug_metrics)
    recrop_attempted: bool = False
    recrop_used: bool = False
    recrop_failed_incomplete: bool = False

    def to_footnotes_list(self) -> list[dict[str, str]]:
        """Return footnotes as ordered list (visual order). No sorting by marker."""
        # footnotes_content is already the ordered list — return a copy.
        return list(self.footnotes_content)


def _build_prompt(
    bank_code: str,
    vision_cfg: dict[str, Any],
    reference_text: str | None = None,
    mode: str = "full",
) -> str:
    """Build prompt with bank-specific footnote marker hints and optional OCR reference text."""
    marker_type = str(vision_cfg.get("footnote_marker_type", "")).strip().lower()
    expected = vision_cfg.get("expected_markers")
    hints = []
    if marker_type == "parenthetical":
        hints.append("Format attendu: parenthesique (1), (2), (3)")
    elif marker_type == "superscript":
        hints.append("Format attendu: superscript ou chiffres 1, 2, 3 (ou 1 2 3 4 5)")
    if expected and isinstance(expected, list):
        hints.append(f"Marqueurs possibles: {expected[:5]}")
    suffix = "\n".join(hints) if hints else ""

    # Multimodal Grounding: inject OCR reference text as spelling dictionary
    reference_section = ""
    reference_text_max_chars = int(
        vision_cfg.get(
            "vision_reference_text_max_chars", _DEFAULT_REFERENCE_TEXT_MAX_CHARS
        )
    )
    if (
        mode == "full"
        and reference_text
        and len(reference_text.strip()) > 20
        and reference_text_max_chars > 0
    ):
        truncated = reference_text.strip()[:reference_text_max_chars]
        reference_section = (
            "\n\n=== DICTIONNAIRE DE RÉFÉRENCE (Texte OCR du tableau) ===\n"
            f"{truncated}\n"
            "=== FIN DICTIONNAIRE ===\n\n"
            "CONSIGNE : Utilise l'image pour l'ordre visuel et la structure du tableau. "
            "Utilise le Dictionnaire de Référence ci-dessus pour VÉRIFIER L'ORTHOGRAPHE EXACTE "
            "des libellés d'indicateurs, en-têtes et notes de bas de page. "
            "En cas de conflit entre l'image et le dictionnaire, privilégie l'orthographe du dictionnaire.\n"
        )

    return (
        _PROMPT_BASE
        + (f"\n{suffix}\n" if suffix else "")
        + reference_section
        + (_PROMPT_JSON_LEAN if mode == "lean" else _PROMPT_JSON_STRICT)
    )


def _build_content(prompt: str, image_b64: str) -> list[Any]:
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}",
                "detail": "high",
            },
        },
    ]


def _build_repair_prompt(base_prompt: str, raw_content: str) -> str:
    return (
        "Le contenu precedent n'etait pas exploitable. "
        "Retourne UNIQUEMENT un objet JSON valide conforme au schema demande, "
        "sans markdown, sans commentaire, sans texte avant ou apres.\n\n"
        f"{base_prompt}\n\n"
        "Reponse precedente a corriger:\n"
        f"{raw_content[:2000]}"
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from GPT response and find JSON boundaries."""
    stripped = text.strip()

    # Étape 1 : Retirer les balises markdown si présentes
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    # Étape 2 : Chercher l'objet JSON (accolades)
    # L'API Vision / JSON mode retourne toujours un objet (dictionnaire) dans ce contexte
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]

    return stripped


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """Parse JSON from response. Returns None on failure."""
    try:
        cleaned = _strip_markdown_fences(raw)
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _preview_response_text(raw: str, limit: int = 500) -> str:
    """Return a compact response preview suitable for logs."""
    text = (raw or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head} ... {tail}"


def _response_keys_for_mode(mode: str) -> tuple[frozenset[str], frozenset[str]]:
    if mode == "lean":
        return (
            frozenset(
                {
                    "table_title",
                    "headers",
                    "indicators",
                    "footnotes_content",
                    "footnote_markers",
                    "confidence",
                    "appears_truncated",
                    "estimated_content_height",
                }
            ),
            frozenset({"indicators", "confidence"}),
        )
    return (
        frozenset(
            {
                "table_title",
                "headers",
                "indicators",
                "rows",
                "footnotes_content",
                "footnote_markers",
                "has_hierarchy",
                "extraction_confidence",
                "notes",
                "confidence",
                "appears_truncated",
                "estimated_content_height",
            }
        ),
        frozenset({"indicators", "confidence"}),
    )


def _extract_embedded_schema_candidate(
    raw: dict[str, Any], *, schema_mode: str = "full"
) -> dict[str, Any] | None:
    """Return the most likely nested payload matching the expected Vision schema."""
    if not isinstance(raw, dict):
        return None
    response_keys, required_keys = _response_keys_for_mode(schema_mode)
    raw_keys = set(raw.keys())
    if required_keys.issubset(raw_keys):
        return raw

    best_candidate: dict[str, Any] | None = None
    best_score = 0
    queue: list[dict[str, Any]] = [raw]
    seen_ids: set[int] = set()

    while queue:
        candidate = queue.pop(0)
        obj_id = id(candidate)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        keys = set(candidate.keys())
        score = len(keys & response_keys)
        if required_keys.issubset(keys):
            return candidate
        if score > best_score:
            best_candidate = candidate
            best_score = score

        for value in candidate.values():
            if isinstance(value, dict):
                queue.append(value)

    if best_candidate is not None and best_score >= 2:
        return best_candidate
    return None


def _parse_vision_result(
    raw: str | dict[str, Any], *, schema_mode: str = "full"
) -> VisionFullResult | None:
    """Parse and validate JSON into VisionFullResult via Pydantic. Returns None on validation error."""
    schema_model = _schema_model_for_mode(schema_mode)
    try:
        if isinstance(raw, dict):
            validated = schema_model.model_validate(raw)
        else:
            validated = schema_model.model_validate_json(raw)
    except Exception as e:
        if isinstance(raw, dict):
            candidate = _extract_embedded_schema_candidate(raw, schema_mode=schema_mode)
            if candidate is not None and candidate is not raw:
                try:
                    validated = schema_model.model_validate(candidate)
                    logger.info(
                        "Vision response recovered from nested wrapper keys: %s",
                        list(raw.keys())[:3],
                    )
                except Exception:
                    logger.warning(
                        "Vision response validation failed (Pydantic schema error): %s",
                        e,
                    )
                    return None
            else:
                logger.warning(
                    "Vision response validation failed (Pydantic schema error): %s", e
                )
                return None
        else:
            logger.warning(
                "Vision response validation failed (Pydantic schema error): %s", e
            )
            return None

    # Build ordered footnotes list — preserves visual order from Pydantic validation.
    # The _coerce_footnotes_content validator already normalized dict->list.
    footnotes_ordered: list[dict[str, str]] = [
        {"marker": str(item.id).strip(), "text": str(item.text).strip()}
        for item in validated.footnotes_content
        if str(item.id).strip() and str(item.text).strip()
    ]

    # Build structured indicators list — preserves text + bbox from Pydantic validation.
    indicators_ordered: list[dict[str, Any]] = [
        {
            "text": str(item).strip(),
            "bbox": None,
        }
        for item in validated.indicators
        if str(item).strip()
    ]

    return VisionFullResult(
        table_title=validated.table_title or "",
        headers=validated.headers or [],
        indicators=indicators_ordered,
        rows=list(getattr(validated, "rows", []) or []),
        footnotes_content=footnotes_ordered,
        footnote_markers=validated.footnote_markers,
        confidence=validated.confidence,
        extraction_method=_EXTRACTION_METHOD,
        appears_truncated=validated.appears_truncated,
        estimated_content_height=validated.estimated_content_height,
        vision_status="ok",
        warnings=[],
    )


class VisionFullExtractor:
    """
    Extract indicators + footnotes from a table crop image via GPT-4o Vision.

    One call per table minimum. Supports:
    - Retry on invalid JSON (with fix prompt)
    - Multi-pass: re-crop with bottom_extension if confidence < 0.85 or indicators empty
    - Cache via vision_cache (pdf_sha + page + bbox)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_retries_json: int = 2,
        use_cache: bool = False,
    ):
        self._api_key = api_key or get_openai_api_key()
        self._model = model
        self._max_retries_json = max_retries_json
        self._use_cache = use_cache
        self._client: Any = None
        self._disabled_reason: str | None = None
        self._schema_contract_checked: set[str] = set()
        self._schema_contract_error_logged = False

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI

            if not self._api_key:
                raise ValueError("OPENAI_API_KEY required for Vision extraction")
            self._client = OpenAI(api_key=self._api_key)
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

    def extract(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        bottom_extension_used: float = 0.0,
        reference_text: str | None = None,
    ) -> VisionFullResult | None:
        """
        Extract indicators and footnotes from a table crop.

        Args:
            crop_bytes: PNG bytes of the cropped table image
            bank_code: Bank code for prompt hints (bnc, rbc, td, etc.)
            pdf_sha: PDF hash for cache key (optional)
            page_number: Page number for cache key (optional)
            bbox_norm: Normalized bbox for cache key (optional)
            vision_cfg: Config overrides (footnote_marker_type, expected_markers)
            bottom_extension_used: Bottom extension already applied (for cache key variant)

        Returns:
            VisionFullResult or None on failure
        """
        if self._disabled_reason:
            raise VisionSchemaContractError(self._disabled_reason)

        vision_cfg = vision_cfg or {}
        cache_key = ""
        if (
            self._use_cache
            and pdf_sha
            and page_number
            and bbox_norm
            and len(bbox_norm) == 4
        ):
            bbox_with_ext = list(bbox_norm)
            if len(bbox_with_ext) >= 4:
                bbox_with_ext[3] = min(1.0, bbox_with_ext[3] + bottom_extension_used)
            cache_key = make_cache_key(pdf_sha, page_number, bbox_with_ext)
            if cache_key:
                cache_dir = get_vision_cache_dir()
                cached = cache_get(cache_dir, cache_key)
                if cached:
                    indicators = cached.get("indicators")
                    fn_content_raw = cached.get("footnotes_content", [])
                    fn_markers = cached.get("footnote_markers", [])
                    confidence = float(cached.get("confidence", 0.0))
                    appears_truncated = bool(cached.get("appears_truncated", False))
                    estimated_content_height = cached.get("estimated_content_height")
                    # Migration shim: legacy cache stores indicators as list[str],
                    # new format is list[dict] with {text, bbox}.
                    if isinstance(indicators, list):
                        migrated_indicators: list[dict[str, Any]] = []
                        for item in indicators:
                            if isinstance(item, str) and item.strip():
                                migrated_indicators.append(
                                    {"text": item.strip(), "bbox": None}
                                )
                            elif isinstance(item, dict) and item.get("text"):
                                migrated_indicators.append(item)
                        indicators = migrated_indicators
                    else:
                        indicators = None
                    if isinstance(indicators, list):
                        # Migration shim: legacy cache may store footnotes as dict.
                        if isinstance(fn_content_raw, dict):
                            fn_content: list[dict[str, str]] = [
                                {"marker": str(k), "text": str(v)}
                                for k, v in fn_content_raw.items()
                                if str(k).strip() and str(v).strip()
                            ]
                        elif isinstance(fn_content_raw, list):
                            fn_content = [
                                item
                                for item in fn_content_raw
                                if isinstance(item, dict)
                                and (item.get("marker") or item.get("id"))
                                and item.get("text")
                            ]
                            # Normalize legacy {id,text} -> {marker,text}
                            fn_content = [
                                {
                                    "marker": str(
                                        item.get("marker") or item.get("id", "")
                                    ).strip(),
                                    "text": str(item.get("text", "")).strip(),
                                }
                                for item in fn_content
                            ]
                        else:
                            fn_content = []
                        if isinstance(fn_markers, list):
                            fn_markers = [str(x) for x in fn_markers]
                        else:
                            fn_markers = []
                        logger.info(
                            "VisionFull cache hit: %d indicators, conf=%.2f",
                            len(indicators),
                            confidence,
                        )
                        return VisionFullResult(
                            table_title=str(cached.get("table_title") or ""),
                            headers=list(cached.get("headers") or []),
                            indicators=indicators,
                            rows=list(cached.get("rows") or []),
                            footnotes_content=fn_content,
                            footnote_markers=fn_markers,
                            confidence=max(0.0, min(1.0, confidence)),
                            extraction_method=_EXTRACTION_METHOD,
                            appears_truncated=appears_truncated,
                            estimated_content_height=(
                                int(estimated_content_height)
                                if estimated_content_height is not None
                                else None
                            ),
                            vision_status=str(cached.get("vision_status") or "ok"),
                            warnings=list(cached.get("warnings") or []),
                        )

        try:
            self._ensure_client()
        except (ImportError, ValueError) as e:
            logger.warning("VisionFullExtractor: client init failed: %s", e)
            return None

        client = self._client
        if client is None:
            return None

        try:
            from .vision_image_preprocessor import preprocess_for_vision

            processed = preprocess_for_vision(crop_bytes)
            image_b64 = base64.standard_b64encode(processed).decode("ascii")
        except Exception as e:
            logger.debug("Vision preprocessing failed, using raw: %s", e)
            image_b64 = base64.standard_b64encode(crop_bytes).decode("ascii")

        prompt = _build_prompt(bank_code, vision_cfg, reference_text=reference_text)
        prompt_lean = _build_prompt(
            bank_code,
            vision_cfg,
            reference_text=reference_text,
            mode="lean",
        )
        max_completion_tokens_full = int(
            vision_cfg.get(
                "vision_max_completion_tokens_full",
                _DEFAULT_MAX_COMPLETION_TOKENS_FULL,
            )
        )
        max_completion_tokens_lean = int(
            vision_cfg.get(
                "vision_max_completion_tokens_lean",
                _DEFAULT_MAX_COMPLETION_TOKENS_LEAN,
            )
        )
        openai_schemas: dict[str, dict[str, Any]] = {
            "full": _build_openai_json_schema("full"),
            "lean": _build_openai_json_schema("lean"),
        }
        for schema_mode, schema in openai_schemas.items():
            if schema_mode in self._schema_contract_checked:
                continue
            try:
                _validate_openai_strict_schema_contract(schema)
                self._schema_contract_checked.add(schema_mode)
            except VisionSchemaContractError as exc:
                self._schema_contract_checked.add(schema_mode)
                self._disabled_reason = str(exc)
                if not self._schema_contract_error_logged:
                    logger.error(
                        "Vision schema contract invalid (local validation, mode=%s): %s",
                        schema_mode,
                        exc,
                    )
                    self._schema_contract_error_logged = True
                raise

        api_retry_max = int(vision_cfg.get("api_retry_max", 3))
        api_retry_backoff_ms = float(vision_cfg.get("api_retry_backoff_ms", 1000))

        def _issue_request(
            prompt_text: str,
            *,
            structured: bool,
            schema_mode: str,
            max_completion_tokens: int,
            label: str,
        ) -> tuple[str, str, bool] | None:
            local_use_structured = structured
            transport_attempt = 0
            while transport_attempt <= api_retry_max:
                if transport_attempt > 0:
                    backoff_sec = (api_retry_backoff_ms / 1000.0) * (
                        2 ** (transport_attempt - 1)
                    )
                    logger.info(
                        "Vision %s: transport retry %s/%s after %.1fs backoff",
                        label,
                        transport_attempt,
                        api_retry_max,
                        backoff_sec,
                    )
                    time.sleep(backoff_sec)
                try:
                    response_format: dict[str, Any] = (
                        (openai_schemas.get(schema_mode) or {"type": "json_object"})
                        if local_use_structured
                        else {"type": "json_object"}
                    )
                    response = client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {
                                "role": "user",
                                "content": _build_content(prompt_text, image_b64),
                            }
                        ],
                        response_format=response_format,
                        temperature=0,
                        max_completion_tokens=max_completion_tokens,
                    )
                    return (
                        response.choices[0].message.content or "",
                        str(getattr(response.choices[0], "finish_reason", "") or ""),
                        local_use_structured,
                    )
                except Exception as e:
                    err_kind = _classify_openai_error(e)
                    if err_kind == "schema_contract_invalid":
                        self._disabled_reason = f"Vision schema contract invalid: {e}"
                        if not self._schema_contract_error_logged:
                            logger.error("%s", self._disabled_reason)
                            self._schema_contract_error_logged = True
                        raise VisionSchemaContractError(self._disabled_reason) from e
                    if (
                        local_use_structured
                        and err_kind == "structured_output_unsupported"
                    ):
                        logger.debug(
                            "Structured Outputs unsupported for %s, falling back to json_object: %s",
                            label,
                            e,
                        )
                        local_use_structured = False
                        continue
                    if err_kind in ("rate_limit", "timeout", "connection"):
                        transport_attempt += 1
                        if transport_attempt <= api_retry_max:
                            continue
                        logger.warning(
                            "Vision %s API error (retries exhausted): %s",
                            label,
                            e,
                        )
                        return None
                    logger.warning("Vision %s API error: %s", label, e)
                    return None
            logger.warning("Vision %s: no successful API response", label)
            return None

        attempt_profiles: list[dict[str, Any]] = [
            {
                "label": "full",
                "prompt": prompt,
                "structured": True,
                "schema_mode": "full",
                "max_completion_tokens": max_completion_tokens_full,
            }
        ]
        if self._max_retries_json >= 1:
            attempt_profiles.append(
                {
                    "label": "lean",
                    "prompt": prompt_lean,
                    "structured": True,
                    "schema_mode": "lean",
                    "max_completion_tokens": max_completion_tokens_lean,
                }
            )
        if self._max_retries_json >= 2:
            attempt_profiles.append(
                {
                    "label": "lean-json",
                    "prompt": prompt_lean,
                    "structured": False,
                    "schema_mode": "lean",
                    "max_completion_tokens": max_completion_tokens_lean,
                }
            )

        previous_raw_content = ""
        failure_causes: list[str] = []

        for attempt_index, profile in enumerate(attempt_profiles):
            prompt_text = profile["prompt"]
            if profile["label"] == "lean-json" and previous_raw_content:
                prompt_text = _build_repair_prompt(prompt_text, previous_raw_content)

            issued = _issue_request(
                prompt_text,
                structured=bool(profile["structured"]),
                schema_mode=str(profile["schema_mode"]),
                max_completion_tokens=int(profile["max_completion_tokens"]),
                label=str(profile["label"]),
            )
            if issued is None:
                return None

            raw_content, finish_reason, used_structured = issued
            previous_raw_content = raw_content

            if finish_reason == "length":
                failure_causes.append("vision_truncated")
                logger.warning(
                    "Vision %s: response truncated (raw_len=%d)",
                    profile["label"],
                    len(raw_content),
                )
                if attempt_index + 1 < len(attempt_profiles):
                    continue
                break

            data = _parse_json_response(raw_content)
            if data is None:
                failure_causes.append("vision_invalid_json")
                logger.info(
                    "Vision %s: JSON parse failed (raw_len=%d, preview=%r)",
                    profile["label"],
                    len(raw_content),
                    _preview_response_text(raw_content),
                )
                if attempt_index + 1 < len(attempt_profiles):
                    continue
                break

            result = _parse_vision_result(data, schema_mode=str(profile["schema_mode"]))
            if result is None:
                failure_causes.append("vision_schema_validation_failed")
                logger.info(
                    "Vision %s: schema validation failed (raw_len=%d, keys=%s)",
                    profile["label"],
                    len(raw_content),
                    sorted(data.keys()),
                )
                if attempt_index + 1 < len(attempt_profiles):
                    continue
                break

            if profile["label"] != "full":
                result.rows = []
                failure_causes.append("vision_rows_missing_from_fallback")
            if used_structured is False and bool(profile["structured"]):
                failure_causes.append("vision_structured_output_fallback")
            if profile["label"] != "full" or failure_causes:
                result.vision_status = "partial"
                deduped = list(dict.fromkeys(failure_causes))
                if profile["label"] != "full":
                    deduped.append("vision_lean_mode")
                result.warnings = list(dict.fromkeys(deduped))

            if (
                self._use_cache
                and cache_key
                and str(profile["label"]) == "full"
                and result.vision_status == "ok"
            ):
                cache_dir = get_vision_cache_dir()
                cache_put(
                    cache_dir,
                    cache_key,
                    {
                        "table_title": result.table_title,
                        "headers": result.headers,
                        "indicators": result.indicators,
                        "rows": result.rows,
                        # footnotes persisted as ordered list {marker, text}
                        "footnotes_content": result.footnotes_content,
                        "footnote_markers": result.footnote_markers,
                        "confidence": result.confidence,
                        "appears_truncated": result.appears_truncated,
                        "estimated_content_height": result.estimated_content_height,
                        "vision_status": result.vision_status,
                        "warnings": result.warnings,
                    },
                )
            return result

        logger.warning(
            "Vision full extraction: invalid content after retries (%s)",
            ", ".join(dict.fromkeys(failure_causes + ["vision_retry_exhausted"])),
        )
        return None

    def extract_with_quality_pass(
        self,
        crop_bytes: bytes,
        bank_code: str,
        pdf_sha: str = "",
        page_number: int = 0,
        bbox_norm: list[float] | None = None,
        vision_cfg: dict[str, Any] | None = None,
        initial_bottom_extension: float = 0.0,
        get_recrop_fn: Any = None,
        reference_text: str | None = None,
    ) -> VisionFullResult | None:
        """
        Extract with optional second pass if quality is low.

        If confidence < 0.85 OR indicators empty OR expected markers missing:
        - Re-crop with bottom_extension + 0.06 via get_recrop_fn
        - Retry extraction
        - Return best result (max confidence, markers coherence)

        get_recrop_fn(bottom_extension: float) -> bytes | None
        """
        vision_cfg = vision_cfg or {}
        expected_markers = vision_cfg.get("expected_markers")
        if isinstance(expected_markers, list):
            expected_set = {str(m).strip() for m in expected_markers[:10]}
        else:
            expected_set = set()

        def _indicator_count(r: VisionFullResult) -> int:
            return len(r.indicators) if r.indicators else 0

        def _row_count(r: VisionFullResult) -> int:
            return len(r.rows) if r.rows else 0

        def _rows_degraded(r: VisionFullResult) -> bool:
            warnings = list(getattr(r, "warnings", []) or [])
            return bool(
                {"vision_rows_limited", "vision_rows_missing_from_fallback"} & set(warnings)
            )

        def _needs_recrop(result: VisionFullResult | None) -> bool:
            if result is None:
                return True
            if result.appears_truncated:
                return True
            if result.confidence < _CONFIDENCE_RETRY_THRESHOLD:
                return True
            if not result.indicators:
                return True
            if expected_set and result.footnote_markers:
                found = {str(m).strip() for m in result.footnote_markers}
                if not (found & expected_set) and len(found) < len(expected_set):
                    return True
            if _rows_degraded(result):
                # Fallback rows are intentionally degraded and should not trigger recrop by themselves.
                return False
            # Completeness: indicators present but rows nearly empty
            n_ind = _indicator_count(result)
            n_row = _row_count(result)
            if n_ind >= 5 and n_row < max(1, int(0.3 * n_ind)):
                return True
            # Suspicious indicator/row mismatch (many indicators, few rows, non-empty)
            if n_ind >= 8 and 0 < n_row < n_ind // 2:
                return True
            # Missing title when crop is near page top (bbox_norm top < 0.15)
            if (
                bbox_norm
                and len(bbox_norm) >= 2
                and bbox_norm[1] < 0.15
                and not (result.table_title or "").strip()
            ):
                return True
            # Suspiciously small output: many indicators but very few data rows
            if n_ind >= 10 and n_row <= 1:
                return True
            return False

        first = self.extract(
            crop_bytes=crop_bytes,
            bank_code=bank_code,
            pdf_sha=pdf_sha,
            page_number=page_number,
            bbox_norm=bbox_norm,
            vision_cfg=vision_cfg,
            bottom_extension_used=initial_bottom_extension,
            reference_text=reference_text,
        )

        if not _needs_recrop(first):
            return first

        if get_recrop_fn is None:
            return first

        recrop_ext = initial_bottom_extension + _RECROP_EXTENSION_INCREMENT
        recrop_bytes = get_recrop_fn(recrop_ext)
        if not recrop_bytes:
            # Recrop failed; if first pass was incomplete, mark it
            first_incomplete = _needs_recrop(first)
            if first_incomplete and first is not None:
                return VisionFullResult(
                    table_title=first.table_title,
                    headers=first.headers,
                    indicators=first.indicators,
                    rows=first.rows,
                    footnotes_content=first.footnotes_content,
                    footnote_markers=first.footnote_markers,
                    confidence=first.confidence,
                    extraction_method=first.extraction_method,
                    appears_truncated=first.appears_truncated,
                    estimated_content_height=first.estimated_content_height,
                    vision_status=first.vision_status,
                    warnings=first.warnings,
                    recrop_attempted=True,
                    recrop_used=False,
                    recrop_failed_incomplete=True,
                )
            return first

        second = self.extract(
            crop_bytes=recrop_bytes,
            bank_code=bank_code,
            pdf_sha=pdf_sha,
            page_number=page_number,
            bbox_norm=bbox_norm,
            vision_cfg=vision_cfg,
            bottom_extension_used=recrop_ext,
            reference_text=reference_text,
        )

        if second is None:
            out = first
            if first is not None:
                out = VisionFullResult(
                    table_title=first.table_title,
                    headers=first.headers,
                    indicators=first.indicators,
                    rows=first.rows,
                    footnotes_content=first.footnotes_content,
                    footnote_markers=first.footnote_markers,
                    confidence=first.confidence,
                    extraction_method=first.extraction_method,
                    appears_truncated=first.appears_truncated,
                    estimated_content_height=first.estimated_content_height,
                    vision_status=first.vision_status,
                    warnings=first.warnings,
                    recrop_attempted=True,
                    recrop_used=False,
                    recrop_failed_incomplete=_needs_recrop(first),
                )
            return out
        if first is None:
            return VisionFullResult(
                table_title=second.table_title,
                headers=second.headers,
                indicators=second.indicators,
                rows=second.rows,
                footnotes_content=second.footnotes_content,
                footnote_markers=second.footnote_markers,
                confidence=second.confidence,
                extraction_method=second.extraction_method,
                appears_truncated=second.appears_truncated,
                estimated_content_height=second.estimated_content_height,
                vision_status=second.vision_status,
                warnings=second.warnings,
                recrop_attempted=True,
                recrop_used=True,
                recrop_failed_incomplete=False,
            )

        def _completeness(r: VisionFullResult) -> float:
            c = 0.0
            n_ind = _indicator_count(r)
            n_row = _row_count(r)
            if n_ind > 0 and n_row > 0:
                ratio = n_row / n_ind
                c += 0.2 * min(1.0, ratio)  # row/indicator balance
            if (r.table_title or "").strip():
                c += 0.15
            if not r.appears_truncated:
                c += 0.15
            if r.indicators:
                c += 0.1 * min(1.0, len(r.indicators) / 20.0)
            return c

        def _score(r: VisionFullResult) -> float:
            s = r.confidence
            s += _completeness(r)
            if r.indicators:
                s += 0.05 * min(1.0, len(r.indicators) / 20.0)
            if expected_set and r.footnote_markers:
                found = {str(m).strip() for m in r.footnote_markers}
                s += 0.1 * (len(found & expected_set) / max(1, len(expected_set)))
            return s

        chosen = second if _score(second) >= _score(first) else first
        return VisionFullResult(
            table_title=chosen.table_title,
            headers=chosen.headers,
            indicators=chosen.indicators,
            rows=chosen.rows,
            footnotes_content=chosen.footnotes_content,
            footnote_markers=chosen.footnote_markers,
            confidence=chosen.confidence,
            extraction_method=chosen.extraction_method,
            appears_truncated=chosen.appears_truncated,
            estimated_content_height=chosen.estimated_content_height,
            vision_status=chosen.vision_status,
            warnings=chosen.warnings,
            recrop_attempted=True,
            recrop_used=(chosen is second),
            recrop_failed_incomplete=False,
        )

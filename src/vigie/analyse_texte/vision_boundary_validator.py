"""Arbitrage Vision ciblé des frontières textuelles ambiguës."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vigie.support.config import resolve_openai_model
from vigie.extraction.vision_cache import compute_pdf_sha256
from vigie.support.utils.genai import get_openai_api_key
from vigie.support.utils.proof_rendering import render_full_proof_bytes

logger = logging.getLogger(__name__)

_BOUNDARY_VISION_SCHEMA_VERSION = "v1"


class VisionBoundaryAssessment(BaseModel):
    """Réponse structurée demandée au modèle Vision."""

    model_config = ConfigDict(extra="forbid")

    same_sentence: Literal["yes", "no", "uncertain"]
    reading_order: Literal["previous_then_next", "reversed", "uncertain"]
    previous_block_type: Literal["narrative", "title", "list", "table", "chrome", "uncertain"]
    next_block_type: Literal["narrative", "title", "list", "table", "chrome", "uncertain"]
    allow_remove_previous: bool
    allow_remove_next: bool
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = Field(max_length=500)


class VisionBoundaryDecision(BaseModel):
    """Décision d'application enrichie de son statut d'audit."""

    model_config = ConfigDict(extra="forbid")

    apply_merge: bool = False
    status: str
    cached: bool = False
    model: str
    same_sentence: Literal["yes", "no", "uncertain"] = "uncertain"
    reading_order: Literal["previous_then_next", "reversed", "uncertain"] = "uncertain"
    previous_block_type: Literal["narrative", "title", "list", "table", "chrome", "uncertain"] = "uncertain"
    next_block_type: Literal["narrative", "title", "list", "table", "chrome", "uncertain"] = "uncertain"
    allow_remove_previous: bool = False
    allow_remove_next: bool = False
    confidence: float = 0.0
    justification: str = ""


def _segment_value(segment: Any, name: str, default: Any = None) -> Any:
    return getattr(segment, name, default)


class OpenAITextBoundaryValidator:
    """Valide au plus quelques frontières ambiguës à partir de la page PDF."""

    def __init__(
        self,
        *,
        pdf_path: Path,
        cache_dir: Path,
        client: Any,
        model: str,
        confidence_threshold: float = 0.90,
        max_calls: int = 12,
        render_dpi: int = 200,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.cache_dir = Path(cache_dir)
        self.client = client
        self.model = str(model)
        self.confidence_threshold = float(confidence_threshold)
        self.max_calls = max(int(max_calls), 0)
        self.render_dpi = max(int(render_dpi), 72)
        self.calls_made = 0
        self.pdf_sha = compute_pdf_sha256(str(self.pdf_path))

    def _cache_path(self, previous: Any, current: Any) -> Path:
        payload = {
            "version": _BOUNDARY_VISION_SCHEMA_VERSION,
            "pdf_sha": self.pdf_sha,
            "model": self.model,
            "previous": {
                "page": _segment_value(previous, "page"),
                "bbox": _segment_value(previous, "bbox_norm"),
                "text": str(_segment_value(previous, "text", "")),
            },
            "current": {
                "page": _segment_value(current, "page"),
                "bbox": _segment_value(current, "bbox_norm"),
                "text": str(_segment_value(current, "text", "")),
            },
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> VisionBoundaryDecision | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return VisionBoundaryDecision.model_validate(payload)
        except Exception:  # noqa: BLE001
            return None

    def _write_cache(self, path: Path, decision: VisionBoundaryDecision) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(decision.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Impossible d'écrire le cache Vision des frontières: %s", exc)

    def _render_images(self, previous: Any, current: Any) -> list[dict[str, Any]]:
        by_page: dict[int, list[list[float]]] = {}
        for segment in (previous, current):
            page = _segment_value(segment, "page")
            bbox = _segment_value(segment, "bbox_norm")
            if page is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            by_page.setdefault(int(page), []).append([float(value) for value in bbox])

        image_parts: list[dict[str, Any]] = []
        for page, boxes in sorted(by_page.items()):
            union_bbox = [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
            image_bytes, status, _mode = render_full_proof_bytes(
                self.pdf_path,
                page=page,
                bbox=union_bbox,
                dpi=self.render_dpi,
                allow_full_page_fallback=True,
            )
            if status != "ok" or not image_bytes:
                continue
            encoded = base64.standard_b64encode(image_bytes).decode("ascii")
            image_parts.extend(
                [
                    {"type": "text", "text": f"Page PDF {page}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                            "detail": "high",
                        },
                    },
                ]
            )
        return image_parts

    def _request_assessment(self, previous: Any, current: Any) -> VisionBoundaryAssessment:
        image_parts = self._render_images(previous, current)
        if not image_parts:
            raise ValueError("aucune preuve visuelle exploitable")

        previous_text = str(_segment_value(previous, "text", "")).strip()
        current_text = str(_segment_value(current, "text", "")).strip()
        prompt = (
            "Tu arbitres une frontière ambiguë produite par l'extraction d'un rapport bancaire. "
            "Décide uniquement si les fragments A et B forment visuellement la même phrase et dans quel ordre. "
            "N'invente, ne corrige et ne retranscris aucun texte. "
            "Une colonne différente, un titre, une liste, un tableau ou un pied de page implique same_sentence=no. "
            "Utilise uncertain dès que la preuve visuelle n'est pas suffisante.\n\n"
            f"FRAGMENT A (précédent dans l'extraction):\n{previous_text}\n\n"
            f"FRAGMENT B (suivant dans l'extraction):\n{current_text}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un validateur de mise en page. Réponds strictement selon le schéma. "
                    "Tu n'as jamais l'autorisation de proposer du texte de remplacement."
                ),
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *image_parts],
            },
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=VisionBoundaryAssessment,
            temperature=0.0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("réponse structurée Vision vide")
        return parsed

    def validate(self, previous: Any, current: Any) -> VisionBoundaryDecision:
        """Retourne une décision; toute incertitude conserve la frontière."""
        cache_path = self._cache_path(previous, current)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached
        if self.calls_made >= self.max_calls:
            return VisionBoundaryDecision(
                status="budget_exhausted_fail_closed",
                model=self.model,
                justification="Limite d'appels Vision atteinte; frontière conservée.",
            )

        self.calls_made += 1
        try:
            assessment = self._request_assessment(previous, current)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Validation Vision de frontière impossible: %s", exc)
            return VisionBoundaryDecision(
                status="error_fail_closed",
                model=self.model,
                justification=str(exc)[:500],
            )

        apply_merge = bool(
            assessment.same_sentence == "yes"
            and assessment.reading_order == "previous_then_next"
            and assessment.previous_block_type == "narrative"
            and assessment.next_block_type == "narrative"
            and assessment.confidence >= self.confidence_threshold
        )
        decision = VisionBoundaryDecision(
            apply_merge=apply_merge,
            status="applied" if apply_merge else "reviewed_fail_closed",
            model=self.model,
            **assessment.model_dump(),
        )
        self._write_cache(cache_path, decision)
        return decision


def build_text_boundary_validator(
    *,
    pdf_path: Path,
    project_root: Path,
    config: dict[str, Any],
    client: Any | None = None,
) -> OpenAITextBoundaryValidator | None:
    """Construit le validateur optionnel lorsque Vision et la clé sont disponibles."""
    if not bool(config.get("boundary_vision_enabled", True)):
        return None
    api_key = get_openai_api_key()
    if client is None and not api_key:
        logger.info("Vision des frontières désactivée: OPENAI_API_KEY absente.")
        return None
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=float(config.get("boundary_vision_timeout_sec", 120)))
    model = str(config.get("boundary_vision_model") or resolve_openai_model("default_genai"))
    return OpenAITextBoundaryValidator(
        pdf_path=pdf_path,
        cache_dir=project_root / "outputs" / "text_boundary_vision_cache",
        client=client,
        model=model,
        confidence_threshold=float(config.get("boundary_vision_confidence_min", 0.90)),
        max_calls=int(config.get("boundary_vision_max_calls_per_report", 12)),
        render_dpi=int(config.get("boundary_vision_dpi", 200)),
    )

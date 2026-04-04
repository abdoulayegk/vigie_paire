"""Resume executif GenAI et etiquetage de pertinence pour les resultats de comparaison.

Active par feature flag via ``genai_settings`` dans ``bank_profiles.yaml``.
Necessite ``OPENAI_API_KEY`` dans l'environnement.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_GENAI_SCHEMA = {
    "resume_executif": ["string"],
    "pertinence_globale": "ELEVEE|MOYENNE|FAIBLE",
    "mentions_reglementaires": ["string"],
    "tableaux": [
        {
            "table_uid": "string",
            "label_pertinence": "REGLEMENTAIRE|RISQUE|CAPITAL|STRUCTURE|COSMETIQUE|INCONNU",
            "raison": "string",
            "changements_cles": ["string"],
        }
    ],
}

_VALID_RELEVANCE_LABELS = frozenset(
    {
        "REGLEMENTAIRE",
        "RISQUE",
        "CAPITAL",
        "STRUCTURE",
        "COSMETIQUE",
        "INCONNU",
    }
)

_EN_TO_FR_RELEVANCE_LABELS: dict[str, str] = {
    "REGULATORY": "REGLEMENTAIRE",
    "RISK": "RISQUE",
    "COSMETIC": "COSMETIQUE",
    "UNKNOWN": "INCONNU",
}

_VALID_OVERALL = frozenset({"ELEVEE", "MOYENNE", "FAIBLE"})

_EN_TO_FR_OVERALL: dict[str, str] = {
    "HIGH": "ELEVEE",
    "MEDIUM": "MOYENNE",
    "LOW": "FAIBLE",
}


def _load_genai_settings() -> dict[str, Any]:
    """Charge les parametres GenAI depuis bank_profiles.yaml."""
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "bank_profiles.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("genai_settings", {}) or {}
    except Exception:
        return {}


def is_genai_enabled() -> bool:
    """Indique si le resume GenAI est active dans la configuration.

    Returns:
        ``True`` si le flag ``enable_genai_summary`` est actif.
    """
    cfg = _load_genai_settings()
    return bool(cfg.get("enable_genai_summary", False))


def _build_genai_input(result: dict[str, Any], max_tables: int = 50) -> str:
    """Construit un prompt textuel compact a partir du resultat de comparaison canonique.

    Exclut les valeurs numeriques brutes conformement a la specification ; inclut
    les titres, noms d'indicateurs, changements de notes de bas de page et
    signaux structurels.

    Args:
        result: Resultat de comparaison canonique.
        max_tables: Nombre maximal de tableaux a inclure dans le prompt.

    Returns:
        Chaine de texte formatee pour le prompt utilisateur LLM.
    """
    parts: list[str] = []
    summary = result.get("summary", {})
    parts.append(
        f"Comparaison : {summary.get('tables_matched', 0)} tableaux apparies, "
        f"{summary.get('tables_added', 0)} ajoutes, {summary.get('tables_removed', 0)} supprimes. "
        f"Indicateurs : +{summary.get('total_added_indicators', 0)} "
        f"-{summary.get('total_removed_indicators', 0)} "
        f"~{summary.get('total_renamed_indicators', 0)}. "
        f"Notes de bas de page : +{summary.get('total_footnotes_added', 0)} "
        f"-{summary.get('total_footnotes_removed', 0)} "
        f"mod {summary.get('total_footnotes_modified', 0)}."
    )

    tables_with_changes: list[dict[str, Any]] = []
    for comp in result.get("table_comparisons", []):
        n = (
            len(comp.get("added_indicators", []))
            + len(comp.get("removed_indicators", []))
            + len(comp.get("renamed_indicators", []))
            + sum(
                comp.get("footnotes_counts", {}).get(k, 0)
                for k in ("added", "removed", "modified")
            )
        )
        if n > 0:
            tables_with_changes.append(comp)

    for ta in result.get("tables_added", []):
        tables_with_changes.append(
            {
                "table_id_t1": "",
                "table_id_t2": ta.get("table_id", ""),
                "title_t1": "",
                "title_t2": ta.get("title", ""),
                "section": ta.get("section", ""),
                "added_indicators": ta.get("indicators", [])[:10],
                "removed_indicators": [],
                "renamed_indicators": [],
                "table_status": "ajoute",
                "footnotes_counts": {"added": 0, "removed": 0, "modified": 0},
            }
        )
    for tr in result.get("tables_removed", []):
        tables_with_changes.append(
            {
                "table_id_t1": tr.get("table_id", ""),
                "table_id_t2": "",
                "title_t1": tr.get("title", ""),
                "title_t2": "",
                "section": tr.get("section", ""),
                "added_indicators": [],
                "removed_indicators": tr.get("indicators", [])[:10],
                "renamed_indicators": [],
                "table_status": "supprime",
                "footnotes_counts": {"added": 0, "removed": 0, "modified": 0},
            }
        )

    tables_with_changes = tables_with_changes[:max_tables]

    for i, comp in enumerate(tables_with_changes):
        uid = comp.get("table_id_t2") or comp.get("table_id_t1") or f"table_{i}"
        title = comp.get("title_t2") or comp.get("title_t1") or "(sans titre)"
        section = comp.get("section", "")
        status = comp.get("table_status", "")

        added = comp.get("added_indicators", [])[:10]
        removed = comp.get("removed_indicators", [])[:10]
        renamed = comp.get("renamed_indicators", [])[:10]
        fn_counts = comp.get("footnotes_counts", {})

        lines = [
            f"\n--- Tableau {uid} : {title} (section : {section}, statut : {status})"
        ]
        if added:
            lines.append(
                f"  Indicateurs ajoutes : {', '.join(str(a)[:80] for a in added)}"
            )
        if removed:
            lines.append(
                f"  Indicateurs supprimes : {', '.join(str(r)[:80] for r in removed)}"
            )
        if renamed:
            ren_strs = []
            for r in renamed:
                if isinstance(r, dict):
                    ren_strs.append(f"{r.get('from', '')} -> {r.get('to', '')}")
                else:
                    ren_strs.append(str(r))
            lines.append(f"  Renommes : {', '.join(s[:80] for s in ren_strs)}")
        fn_total = sum(fn_counts.get(k, 0) for k in ("added", "removed", "modified"))
        if fn_total:
            lines.append(
                f"  Notes modifiees : +{fn_counts.get('added', 0)} -{fn_counts.get('removed', 0)} mod {fn_counts.get('modified', 0)}"
            )
        parts.extend(lines)

    return "\n".join(parts)


_SYSTEM_PROMPT = """\
Tu es un assistant analyste bancaire reglementaire. Tu analyses les resultats \
de comparaison entre trimestre courant et trimestre precedent pour les banques canadiennes. \
Ton role est de produire un resume executif concis et de classifier chaque \
tableau modifie par pertinence.

Regles :
- Se concentrer sur les changements REGLEMENTAIRES, de RISQUE, de CAPITAL \
  (pas les renommages cosmetiques ou la mise en forme).
- Ignorer les changements purement numeriques.
- Les changements de notes de bas de page peuvent signaler des mises a jour \
  methodologiques ou reglementaires -- les signaler.
- Retourner UNIQUEMENT du JSON valide correspondant au schema ci-dessous. \
  Aucun markdown, aucun commentaire.

IMPORTANT : Toutes les reponses doivent etre en francais.

Schema :
{
  "resume_executif": [<string en francais, 3-7 puces>],
  "pertinence_globale": "ELEVEE" | "MOYENNE" | "FAIBLE",
  "mentions_reglementaires": [<string en francais, 0-5 elements>],
  "tableaux": [
    {
      "table_uid": <string>,
      "label_pertinence": "REGLEMENTAIRE" | "RISQUE" | "CAPITAL" | "STRUCTURE" | "COSMETIQUE" | "INCONNU",
      "raison": <string en francais, 1 phrase>,
      "changements_cles": [<string en francais, 1-3 elements>]
    }
  ]
}
"""


def _call_openai(prompt: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Appelle l'API OpenAI chat completion.

    Args:
        prompt: Texte du prompt utilisateur.
        cfg: Parametres GenAI (modele, timeout, etc.).

    Returns:
        Dictionnaire JSON parse ou ``None`` en cas d'echec.
    """
    try:
        from vigilance.utils.genai import get_openai_api_key

        api_key = get_openai_api_key()
        if not api_key:
            logger.warning("GenAI: no OPENAI_API_KEY configured")
            return None
    except Exception:
        return None

    model = str(cfg.get("genai_model", "gpt-4o"))
    timeout = int(cfg.get("genai_timeout_s", 60))

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=timeout)
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        elapsed = time.monotonic() - t0
        logger.info("GenAI call completed in %.1fs (model=%s)", elapsed, model)

        raw = response.choices[0].message.content or ""
        return json.loads(raw)
    except Exception as exc:
        logger.warning("GenAI call failed: %s", exc)
        return None


def _validate_genai_response(data: dict[str, Any]) -> dict[str, Any]:
    """Valide et nettoie la reponse GenAI. Remplit les valeurs par defaut pour les champs manquants.

    Accepte les cles en francais (``resume_executif``) et les cles heritees en
    anglais (``executive_summary_bullets``). Les etiquettes de pertinence sont
    normalisees en francais.

    Args:
        data: Dictionnaire brut retourne par le LLM.

    Returns:
        Dictionnaire valide et normalise avec les cles francaises.
    """
    bullets = data.get("resume_executif") or data.get("executive_summary_bullets")
    if not isinstance(bullets, list):
        bullets = []
    bullets = [str(b) for b in bullets if b][:10]

    overall = str(
        data.get("pertinence_globale") or data.get("overall_relevance") or "MOYENNE"
    ).upper()
    overall = _EN_TO_FR_OVERALL.get(overall, overall)
    if overall not in _VALID_OVERALL:
        overall = "MOYENNE"

    reg_mentions = data.get("mentions_reglementaires") or data.get(
        "regulatory_mentions"
    )
    if not isinstance(reg_mentions, list):
        reg_mentions = []
    reg_mentions = [str(m) for m in reg_mentions if m][:10]

    tables_raw = data.get("tableaux") or data.get("tables")
    if not isinstance(tables_raw, list):
        tables_raw = []
    tables: list[dict[str, Any]] = []
    for tbl in tables_raw:
        if not isinstance(tbl, dict):
            continue
        label = str(
            tbl.get("label_pertinence") or tbl.get("relevance_label") or "INCONNU"
        ).upper()
        label = _EN_TO_FR_RELEVANCE_LABELS.get(label, label)
        if label not in _VALID_RELEVANCE_LABELS:
            label = "INCONNU"
        key_changes = tbl.get("changements_cles") or tbl.get("key_changes")
        if not isinstance(key_changes, list):
            key_changes = []
        tables.append(
            {
                "table_uid": str(tbl.get("table_uid", "")),
                "label_pertinence": label,
                "raison": str(tbl.get("raison") or tbl.get("why") or "")[:500],
                "changements_cles": [str(c)[:200] for c in key_changes][:5],
            }
        )

    return {
        "resume_executif": bullets,
        "pertinence_globale": overall,
        "mentions_reglementaires": reg_mentions,
        "tableaux": tables,
    }


def _heuristic_fallback(result: dict[str, Any]) -> dict[str, Any]:
    """Produit un resume basique lorsque GenAI est indisponible ou echoue.

    Args:
        result: Resultat de comparaison canonique.

    Returns:
        Dictionnaire de resume heuristique avec les cles standard.
    """
    summary = result.get("summary", {})
    bullets = []
    matched = summary.get("tables_matched", 0)
    added_ind = summary.get("total_added_indicators", 0)
    removed_ind = summary.get("total_removed_indicators", 0)
    tables_added = summary.get("tables_added", 0)
    tables_removed = summary.get("tables_removed", 0)

    if matched:
        bullets.append(
            f"{matched} tableaux apparies entre le trimestre courant et le trimestre precedent."
        )
    if added_ind:
        bullets.append(f"{added_ind} indicateur(s) ajoute(s).")
    if removed_ind:
        bullets.append(f"{removed_ind} indicateur(s) supprime(s).")
    if tables_added:
        bullets.append(
            f"{tables_added} nouveau(x) tableau(x) dans le trimestre courant."
        )
    if tables_removed:
        bullets.append(
            f"{tables_removed} tableau(x) retire(s) depuis le trimestre precedent."
        )

    fn_added = summary.get("total_footnotes_added", 0)
    fn_removed = summary.get("total_footnotes_removed", 0)
    fn_modified = summary.get("total_footnotes_modified", 0)
    fn_total = fn_added + fn_removed + fn_modified
    if fn_total:
        bullets.append(f"{fn_total} changement(s) de notes de bas de page.")

    if not bullets:
        bullets = ["Aucun changement significatif detecte."]

    return {
        "resume_executif": bullets,
        "pertinence_globale": "MOYENNE"
        if (added_ind + removed_ind + fn_total) > 0
        else "FAIBLE",
        "mentions_reglementaires": [],
        "tableaux": [],
    }


def generate_genai_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Point d'entree principal : generer un resume executif GenAI.

    Retourne un dict avec les cles : ``resume_executif``, ``pertinence_globale``,
    ``mentions_reglementaires``, ``tableaux``, ``source`` ("llm" ou "heuristic").

    Respecte les feature flags. Fallback heuristique en cas d'echec.
    """
    cfg = _load_genai_settings()
    enabled = bool(cfg.get("enable_genai_summary", False))

    if not enabled:
        fallback = _heuristic_fallback(result)
        fallback["source"] = "heuristic"
        return fallback

    max_tables = int(cfg.get("genai_max_tables", 50))
    prompt = _build_genai_input(result, max_tables=max_tables)

    raw_response = _call_openai(prompt, cfg)
    if raw_response is None:
        fallback = _heuristic_fallback(result)
        fallback["source"] = "heuristic"
        return fallback

    try:
        validated = _validate_genai_response(raw_response)
        validated["source"] = "llm"
        return validated
    except Exception as exc:
        logger.warning("GenAI response validation failed: %s", exc)
        fallback = _heuristic_fallback(result)
        fallback["source"] = "heuristic"
        return fallback


def enrich_result_with_genai(result: dict[str, Any]) -> dict[str, Any]:
    """Enrichit un resultat de comparaison canonique avec le resume GenAI et la pertinence par tableau.

    Modifie ``result`` en place et le retourne.

    Args:
        result: Resultat de comparaison canonique a enrichir.

    Returns:
        Le meme dictionnaire ``result``, enrichi avec les champs GenAI.
    """
    genai_output = generate_genai_summary(result)

    meta = result.setdefault("meta", {})
    exec_summary = meta.setdefault("resume_executif", {})
    exec_summary["llm"] = {
        "puces": genai_output.get("resume_executif", []),
        "pertinence_globale": genai_output.get("pertinence_globale", "MOYENNE"),
        "mentions_reglementaires": genai_output.get("mentions_reglementaires", []),
        "source": genai_output.get("source", "heuristic"),
    }

    table_genai_map: dict[str, dict[str, Any]] = {}
    for tbl in genai_output.get("tableaux", []):
        uid = tbl.get("table_uid", "")
        if uid:
            table_genai_map[uid] = tbl

    for comp in result.get("table_comparisons", []):
        uid = comp.get("table_id_t2") or comp.get("table_id_t1") or ""
        genai_info = table_genai_map.get(uid, {})
        comp["genai"] = {
            "label_pertinence": genai_info.get("label_pertinence", "INCONNU"),
            "raison": genai_info.get("raison", ""),
            "changements_cles": genai_info.get("changements_cles", []),
        }

    return result

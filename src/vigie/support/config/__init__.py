"""Helpers de configuration pour vigie.

Systeme de configuration YAML de l'application vigie. La configuration
est chargee depuis ``bank_profiles.yaml`` (ou un chemin personnalise) avec
les couches suivantes :

- **Profils bancaires** : mapping ``banks`` (bank_code -> profil dict).
- **Defauts en couches** : chaque getter fusionne le bloc global avec les
  surcharges optionnelles par banque, puis comble les cles manquantes avec
  les valeurs par defaut integrees.
- **Surcharges par variable d'environnement** : la selection du modele LLM
  peut etre surchargee via des variables d'environnement
  (ex. OPENAI_MODEL_EXTRACTION_PRIMARY, OPENAI_MODEL_DEFAULT_GENAI).

Resolution des chemins : les chemins relatifs sont resolus par rapport a la
racine du depot (repertoire contenant ``pyproject.toml``) quand le chemin
n'existe pas depuis le repertoire courant. Les chemins absolus sont utilises
tels quels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vigie.support.config.loader import _resolve_config_path, get_bank_cfg, load_config


_DEFAULT_OPENAI_MODELS: dict[str, str] = {
    "extraction_primary": "gpt-5.4",
    "default_genai": "gpt-4o",
}

_MODEL_ENV_OVERRIDES: dict[str, str] = {
    "extraction_primary": "OPENAI_MODEL_EXTRACTION_PRIMARY",
    "default_genai": "OPENAI_MODEL_DEFAULT_GENAI",
}


def load_bank_profiles(
    config_path: str | Path = "configs/bank_profiles.yaml",
) -> dict[str, Any]:
    """Charger la table des profils bancaires depuis la configuration YAML.

    Args:
        config_path: Chemin vers le fichier YAML. Les chemins relatifs sont
            resolus par rapport a la racine du depot.

    Returns:
        Dictionnaire associant le code banque (minuscule) a son profil.
        Dictionnaire vide si la configuration est absente, invalide ou
        ne contient pas de cle ``banks``.
    """
    cfg = load_config(config_path)
    banks = cfg.get("banks")
    if isinstance(banks, dict):
        return banks
    return {}


def get_matching_thresholds(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Charger les seuils de matching depuis la configuration avec surcharges bancaires.

    Supporte deux formats :
    - ``matching_thresholds: {...}`` a la racine
    - ``matching: { thresholds: {...} }``

    Si ``bank_code`` est fourni et que ``banks.<bank_code>.matching_overrides``
    existe, ces surcharges sont fusionnees par-dessus les seuils de base.

    La configuration du diff d'indicateurs (indicator_hungarian_enabled,
    indicator_rename_min_score, etc.) est incluse dans le dictionnaire retourne.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        bank_code: Code banque optionnel pour les surcharges par banque.

    Returns:
        Dictionnaire des seuils de matching et des parametres de diff
        d'indicateurs, avec les valeurs par defaut appliquees pour les cles
        manquantes. Dictionnaire vide si la configuration est absente.
    """
    path = _resolve_config_path(config_path)
    if not path.exists():
        return {}

    try:
        cfg = load_config(path)
    except Exception:
        return {}

    thresholds = cfg.get("matching_thresholds")
    if isinstance(thresholds, dict):
        base = dict(thresholds)
    else:
        matching = cfg.get("matching")
        if isinstance(matching, dict):
            nested = matching.get("thresholds")
            if isinstance(nested, dict):
                base = dict(nested)
            else:
                base = {}
        else:
            base = {}

    if bank_code:
        banks = cfg.get("banks")
        if isinstance(banks, dict):
            key = str(bank_code).strip().lower()
            if key in banks:
                bank_cfg = banks[key]
                if isinstance(bank_cfg, dict):
                    overrides = bank_cfg.get("matching_overrides")
                    if isinstance(overrides, dict):
                        base = {**base, **overrides}

    # Apply indicator diff defaults (PASS 2) when keys absent
    _indicator_defaults: dict[str, Any] = {
        "indicator_hungarian_enabled": True,
        "indicator_rename_min_score": 0.86,
        "indicator_gate_min_len_ratio": 0.55,
        "indicator_gate_min_token_overlap": 1,
        "indicator_similarity_weights": {"ratio": 0.4, "token_set": 0.6},
        "neighbor_aligned_filter_enabled": True,
        "indicator_short_guard_enabled": True,
        "indicator_short_guard_max_tokens": 3,
        "indicator_short_guard_min_stable_tokens": 5,
    }
    for k, v in _indicator_defaults.items():
        if k not in base:
            base[k] = v

    # Embedding defaults (opt-in, config flag use_embeddings default false)
    _embedding_defaults: dict[str, Any] = {
        "use_embeddings": False,
        "embedding_weight_table": 0.12,
        "embedding_weight_indicator": 0.35,
        "embedding_model": "text-embedding-3-small",
    }
    for k, v in _embedding_defaults.items():
        if k not in base:
            base[k] = v

    # Recall-first engine defaults
    _recall_first_defaults: dict[str, Any] = {
        "recall_first_engine_enabled": True,
        "rf_min_match_score": 0.25,
        "rf_min_match_margin": 0.04,
        "rf_strong_pair_min_score": 0.50,
        "rf_strong_pair_min_margin": 0.08,
        "rf_review_candidate_min_score": 0.15,
        "rf_max_elimination_rounds": 5,
        "rf_cross_section_rescue_min_score": 0.20,
        "rf_min_indicator_signal": 0.10,
    }
    for k, v in _recall_first_defaults.items():
        if k not in base:
            base[k] = v

    return base


def get_vision_extraction_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Charger la configuration d'extraction Vision avec surcharges bancaires.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        bank_code: Code banque optionnel pour les surcharges par banque
            (ex. footnote_marker_type, expected_markers).

    Returns:
        Dictionnaire avec les cles enabled, bottom_extension_footnotes,
        run_on_all_tables, fallback_to_docling_on_error, etc. Les surcharges
        par banque sont fusionnees par-dessus le bloc global. Dictionnaire
        vide si la configuration est absente ou invalide.
    """
    path = _resolve_config_path(config_path)
    if not path.exists():
        return {}

    try:
        cfg = load_config(path)
    except Exception:
        return {}

    global_block = cfg.get("vision_extraction")
    if not isinstance(global_block, dict):
        base: dict[str, Any] = {}
    else:
        base = dict(global_block)

    if bank_code:
        banks = cfg.get("banks")
        if isinstance(banks, dict):
            key = str(bank_code).strip().lower()
            if key in banks:
                bank_cfg = banks[key]
                if isinstance(bank_cfg, dict):
                    bank_ve = bank_cfg.get("vision_extraction")
                    if isinstance(bank_ve, dict):
                        base = {**base, **bank_ve}

    return base


def get_text_extraction_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Charge les options de nettoyage et d'arbitrage du Markdown canonique."""
    defaults: dict[str, Any] = {
        "boundary_vision_enabled": True,
        "boundary_vision_confidence_min": 0.90,
        "boundary_vision_max_calls_per_report": 12,
        "boundary_vision_timeout_sec": 120,
        "boundary_vision_dpi": 200,
    }
    path = _resolve_config_path(config_path)
    if not path.exists():
        return defaults
    try:
        cfg = load_config(path)
    except Exception:
        return defaults
    global_block = cfg.get("text_extraction")
    base = {**defaults, **global_block} if isinstance(global_block, dict) else defaults
    if bank_code:
        banks = cfg.get("banks")
        bank_cfg = banks.get(str(bank_code).strip().lower()) if isinstance(banks, dict) else None
        bank_block = bank_cfg.get("text_extraction") if isinstance(bank_cfg, dict) else None
        if isinstance(bank_block, dict):
            base = {**base, **bank_block}
    return base


def get_llm_model_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
) -> dict[str, str]:
    """Charger le routage des modeles OpenAI depuis la configuration YAML.

    N'applique pas les surcharges par variable d'environnement ; utiliser
    ``resolve_openai_model`` pour cela.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.

    Returns:
        Dictionnaire associant les noms de roles (ex. extraction_primary,
        default_genai) aux identifiants de modeles. Repli sur les valeurs
        par defaut integrees si la configuration est absente ou ne contient
        pas de bloc ``llm_models``.
    """
    path = _resolve_config_path(config_path)
    base = dict(_DEFAULT_OPENAI_MODELS)
    if not path.exists():
        return base

    try:
        cfg = load_config(path)
    except Exception:
        return base

    raw = cfg.get("llm_models")
    if not isinstance(raw, dict):
        return base

    for role in _DEFAULT_OPENAI_MODELS:
        value = raw.get(role)
        if isinstance(value, str) and value.strip():
            base[role] = value.strip()
    return base


def resolve_openai_model(
    role: str,
    config_path: str | Path = "configs/bank_profiles.yaml",
) -> str:
    """Resoudre le modele OpenAI pour un role donne avec support des surcharges.

    Ordre de resolution : (1) variable d'environnement, (2) bloc llm_models
    dans la configuration, (3) valeur par defaut integree.
    Roles supportes : extraction_primary, default_genai.

    Args:
        role: Role du modele (ex. extraction_primary, default_genai).
            Normalise en minuscule. Doit etre un role connu.
        config_path: Chemin vers le fichier YAML pour le bloc llm_models.

    Returns:
        Identifiant du modele (ex. gpt-5.4, gpt-4o).

    Raises:
        ValueError: Si le role n'est pas dans les roles connus.
    """
    key = str(role or "").strip().lower()
    if key not in _DEFAULT_OPENAI_MODELS:
        known = ", ".join(sorted(_DEFAULT_OPENAI_MODELS))
        raise ValueError(f"Unknown OpenAI model role '{role}'. Known roles: {known}")

    env_name = _MODEL_ENV_OVERRIDES.get(key)
    if env_name:
        env_value = os.getenv(env_name)
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip()

    cfg = get_llm_model_config(config_path=config_path)
    value = cfg.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _DEFAULT_OPENAI_MODELS[key]


def get_validation_config(
    config_path: str | Path = "configs/bank_profiles.yaml",
    bank_code: str | None = None,
) -> dict[str, Any]:
    """Charger la configuration de validation post-matching avec surcharges bancaires.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.
        bank_code: Code banque optionnel pour les surcharges par banque.

    Returns:
        Dictionnaire avec les cles vision_unmatched_rescue_enabled,
        cross_section_rescue_enabled, cross_section_rescue_rerank_min,
        cross_section_rescue_vision_confidence_min,
        cross_section_rescue_max_candidates_per_table.
    """
    path = _resolve_config_path(config_path)
    cfg: dict[str, Any] = {}
    if path.exists():
        try:
            cfg = load_config(path)
        except Exception:
            pass

    global_block = cfg.get("validation")
    if not isinstance(global_block, dict):
        base: dict[str, Any] = {}
    else:
        base = dict(global_block)

    if bank_code and isinstance(cfg.get("banks"), dict):
        key = str(bank_code).strip().lower()
        bank_cfg = cfg["banks"].get(key)
        if isinstance(bank_cfg, dict):
            bank_val = bank_cfg.get("validation")
            if isinstance(bank_val, dict):
                base = {**base, **bank_val}

    # Cross-section rescue defaults when keys absent
    if "cross_section_rescue_enabled" not in base:
        base["cross_section_rescue_enabled"] = False
    if "cross_section_rescue_rerank_min" not in base:
        base["cross_section_rescue_rerank_min"] = 0.30
    if "cross_section_rescue_vision_confidence_min" not in base:
        base["cross_section_rescue_vision_confidence_min"] = 0.85
    if "cross_section_rescue_max_candidates_per_table" not in base:
        base["cross_section_rescue_max_candidates_per_table"] = 3

    return base


__all__ = [
    "load_config",
    "get_bank_cfg",
    "get_matching_thresholds",
    "get_llm_model_config",
    "load_bank_profiles",
    "resolve_openai_model",
    "get_vision_extraction_config",
    "get_text_extraction_config",
    "get_validation_config",
]

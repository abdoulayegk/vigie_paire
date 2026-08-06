"""Orchestration du triage AMF des changements textuels."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import ValidationError

from vigie.comparaison.triage.amf_taxonomy import (
    THEMES_AMF_DESCRIPTIONS,
    THEMES_AMF_PIPELINE_2,
    TriageAMFCompactLLMBatch,
    TriageValidationError,
)
from vigie.comparaison.analyst_change_presentation import bank_subject as analyst_bank_subject
from vigie.analyse_texte.constants import _TRIAGE_BATCH_SIZE, _TRIAGE_SOURCE_SNIPPET_LIMIT
from vigie.analyse_texte.normalization import _json_dumps
from vigie.analyse_texte.openai_client import (
    _call_structured_completion_with_correction,
    _truncate_prompt_text,
)
from vigie.analyse_texte.text_comparison.change_segments import build_change_segments

from .alignment import (
    _alignment_review_result,
    _change_index_from_validation_error,
    _coherence_review_triage,
    _is_single_semantic_alignment_group,
    _requires_alignment_review,
    _semantic_move_result,
    _verify_triage_coherence,
)
from .constants import (
    _COMPACT_COMPLETION_BASE_TOKENS,
    _COMPACT_COMPLETION_MAX_TOKENS,
    _COMPACT_COMPLETION_TOKENS_PER_CHANGE,
    _MAX_TRIAGE_LLM_WORKERS,
    _SEMANTIC_REASON_FIELDS,
)
from .dedup import (
    _FEW_SHOT_TRIAGE_AMF,
    _group_semantic_triage_duplicates,
    _propagate_triage_to_group,
)
from .evidence import (
    _build_full_evidence_packets,
    _collect_full_evidence_observations,
    _evidence_read_review_triage,
    _requires_full_evidence_packets,
)
from .exclusions import (
    _deterministic_bank_specific_exclusion,
    _deterministic_cosmetic_exclusion,
    _is_semantic_text_move,
)
from .results import (
    _default_triage,
    _persisted_triage_from_compact,
    _prefilter_triage_result,
)
from .themes import (
    _candidate_themes_for_change,
    _normalize_themes_amf,
)


logger = logging.getLogger("vigie.analyse_texte.triage")


def _triage_section_changes(
    *,
    client: Any,
    model: str,
    section_key: str,
    changes: list[dict[str, Any]],
    bank_code: str = "",
) -> list[dict[str, Any]]:
    """Qualifie metier les changements detectes et fusionne le triage.

    Le triage ne recalcule pas la diff textuelle: il prend les changements deja
    identifies, demande une qualification selective au modele, puis rattache le
    resultat a chaque changement pour la retention finale et le resume global.

    Aligne sur la taxonomie AMF appliquee au suivi prudentiel canadien. Le modèle produit
    le schéma AMF v2 (themes_amf multi-label, exclusion_reason, ...) ; les
    champs hérités (category, signals, ...) sont dérivés localement pour
    préserver la compatibilité aval.
    """
    if not changes:
        return []
    effective_bank_code = str(bank_code or "").strip().lower() or str(changes[0].get("bank_code") or "").strip().lower()
    bank_subject = analyst_bank_subject(effective_bank_code)

    # The first GPT call arbitrates the semantic relationship.  Only an
    # explicit ``uncertain`` result remains for a human; same and distinct
    # disclosures proceed to the AMF triage normally.
    if any(_requires_alignment_review(change) or _is_semantic_text_move(change) for change in changes):
        enriched: list[dict[str, Any]] = []
        for change in changes:
            if _requires_alignment_review(change):
                enriched.append(
                    _alignment_review_result(
                        change,
                        bank_code=effective_bank_code,
                    )
                )
            elif _is_semantic_text_move(change):
                enriched.append(
                    _semantic_move_result(
                        change,
                        bank_code=effective_bank_code,
                    )
                )
            else:
                enriched.extend(
                    _triage_section_changes(
                        client=client,
                        model=model,
                        section_key=section_key,
                        changes=[change],
                        bank_code=effective_bank_code,
                    )
                )
        return enriched

    # Deterministic cosmetic + bank-noise pre-filter before any AMF GPT call.
    pending: list[dict[str, Any]] = []
    prefiltered: list[dict[str, Any]] = []
    for change in changes:
        exclusion = _deterministic_cosmetic_exclusion(change) or _deterministic_bank_specific_exclusion(change)
        if exclusion:
            prefiltered.append(
                _prefilter_triage_result(
                    change,
                    exclusion,
                    bank_code=effective_bank_code,
                )
            )
        else:
            pending.append(change)
    if not pending:
        return prefiltered

    # Semantic near-duplicate grouping: one representative is triaged, then
    # the verdict is propagated with an auditable regrouping trace.
    groups = _group_semantic_triage_duplicates(pending, client=client)
    if any(len(group) > 1 for group in groups):
        grouped_results: list[dict[str, Any]] = []
        for group_index, member_indexes in enumerate(groups, start=1):
            members = [pending[index] for index in member_indexes]
            if len(members) == 1:
                grouped_results.extend(
                    _triage_section_changes(
                        client=client,
                        model=model,
                        section_key=section_key,
                        changes=members,
                        bank_code=effective_bank_code,
                    )
                )
                continue
            representative_results = _triage_section_changes(
                client=client,
                model=model,
                section_key=section_key,
                changes=[members[0]],
                bank_code=effective_bank_code,
            )
            if not representative_results:
                continue
            group_id = f"{section_key}_triage_group_{group_index:03d}"
            grouped_results.extend(
                _propagate_triage_to_group(
                    representative=representative_results[0],
                    members=members,
                    group_id=group_id,
                    bank_code=effective_bank_code,
                )
            )
        return [*prefiltered, *grouped_results]

    if len(pending) > _TRIAGE_BATCH_SIZE and not _is_single_semantic_alignment_group(pending):
        chunks = [pending[start : start + _TRIAGE_BATCH_SIZE] for start in range(0, len(pending), _TRIAGE_BATCH_SIZE)]
        max_workers = min(_MAX_TRIAGE_LLM_WORKERS, len(chunks))
        results_by_index: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _triage_section_changes,
                    client=client,
                    model=model,
                    section_key=section_key,
                    changes=chunk,
                    bank_code=effective_bank_code,
                ): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results_by_index[index] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Section triage failed for {section_key}/batch t{index:02d}: {exc}") from exc

        enriched_batches: list[dict[str, Any]] = []
        for index in range(len(chunks)):
            enriched_batches.extend(results_by_index.get(index, []))
        return [*prefiltered, *enriched_batches]

    changes = pending
    triage_inputs = []
    exact_segments_by_index: dict[int, list[dict[str, str]]] = {}
    full_evidence_by_index: dict[int, list[dict[str, Any]]] = {}
    full_evidence_packets_by_index: dict[int, list[dict[str, Any]]] = {}
    full_evidence_failures_by_index: dict[int, str] = {}
    for idx, change in enumerate(changes, start=1):
        exact_segments = build_change_segments(change)
        exact_segments_by_index[idx] = exact_segments
        full_evidence = []
        if _requires_full_evidence_packets(change):
            full_evidence_packets_by_index[idx] = _build_full_evidence_packets(change)
            try:
                full_evidence = _collect_full_evidence_observations(
                    client=client,
                    model=model,
                    change=change,
                    bank_code=effective_bank_code,
                    section_key=section_key,
                    change_index=idx,
                )
            except Exception as exc:
                failure_reason = str(exc)
                full_evidence_failures_by_index[idx] = failure_reason
                logger.error(
                    "full evidence read requires review section=%s change_index=%d error=%s",
                    section_key,
                    idx,
                    failure_reason,
                )
            else:
                full_evidence_by_index[idx] = full_evidence
        exact_segments_for_prompt = [
            {
                "kind": str(segment.get("kind") or ""),
                "text_t1": _truncate_prompt_text(
                    str(segment.get("text_t1") or ""),
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "text_t2": _truncate_prompt_text(
                    str(segment.get("text_t2") or ""),
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
            }
            for segment in exact_segments
        ]
        triage_inputs.append(
            {
                "bank_subject": bank_subject,
                "change_index": idx,
                "diff_type": change["diff_type"],
                "source_snippet_t1": _truncate_prompt_text(
                    change.get("source_text_t1") or change.get("semantic_text_t1") or "",
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "source_snippet_t2": _truncate_prompt_text(
                    change.get("source_text_t2") or change.get("semantic_text_t2") or "",
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "exact_change_segments": exact_segments_for_prompt,
                "alignment_decision": str(change.get("alignment_decision") or ""),
                "alignment_confidence": str(change.get("alignment_confidence") or ""),
                "alignment_rationale": _truncate_prompt_text(
                    str(change.get("alignment_rationale") or ""),
                    _TRIAGE_SOURCE_SNIPPET_LIMIT,
                ),
                "change_summary": change.get("change_summary", ""),
                "full_evidence_observations": full_evidence,
                "candidate_themes": _candidate_themes_for_change(
                    change,
                    section_key=section_key,
                ),
            }
        )

    system_prompt = (
        "Tu qualifies des changements de divulgation d’une banque canadienne "
        "pour une vigie AMF. Réponds uniquement avec le schéma compact demandé. "
        f"La banque analysée est {bank_subject} et le champ "
        f"`changement_constate` doit commencer exactement par « {bank_subject} » "
        "suivi d’un verbe d’action direct, par exemple ajoute, retire, modifie, "
        "précise, transfère ou renomme. N’utilise jamais « le rapport courant », "
        "« le rapport précédent », « le passage », T1 ou T2 comme sujet du texte "
        "analyste. Sois factuel, sans analyse IT, posture, niveau d’impact, action "
        "recommandée ni répétition des textes sources. Rédige séparément, en "
        "français, des phrases complètes, professionnelles et faciles à comprendre "
        "dans `changement_constate`, `signification_metier`, "
        "`comparaison_interbanques`, `limite_interpretation` et "
        "`motif_non_pertinence`. Ne produis pas `relevance_reason`; il sera "
        "assemblé localement. La longueur du changement ne détermine jamais sa "
        "pertinence : une modification très courte peut être substantielle si "
        "elle touche la gouvernance."
    )

    user_prompt = (
        f"Retourne exactement {len(changes)} entrée(s) dans `triages`, une par "
        "changement, avec les mêmes `change_index`, sans doublon ni entrée "
        "supplémentaire.\n\n"
        "Règles strictes :\n"
        "1. `is_relevant=true` seulement pour un changement substantiel utile "
        "à la vigie AMF; dans ce cas, choisis un ou deux codes. Préfère les "
        "`candidate_themes` de l’entrée ; tu peux aussi utiliser tout code de "
        "la taxonomie AMF complète listée ci-dessous ; si aucun ne convient, "
        "utilise `SUJET_EMERGENT_HORS_GRILLE`.\n"
        f"   Taxonomie AMF autorisée : {', '.join(THEMES_AMF_PIPELINE_2)}.\n"
        "2. `is_relevant=false` exige `themes_amf=[]` et `nouvelle_idee=false`. "
        "Une variation chiffrée propre à la banque, une opération interne "
        "(acquisition, rachat, émission, dividende), une mise à jour de calendrier "
        "d’application d’une exigence réglementaire déjà connue, un déplacement "
        "identique, du formatage ou une reformulation sans nouveau fond sont non "
        "pertinents. Exception : le changement explicite du nom d’un comité ou "
        "d’une instance de gouvernance reste pertinent même si son mandat demeure "
        "identique; utilise alors `GOUVERNANCE_RISQUES` et `nouvelle_idee=false`.\n"
        "   Un renommage ciblé d’un comité, d’un intitulé de section, d’une politique, "
        "d’un code de conduite ou d’une terminologie réglementaire reste pertinent "
        "lorsqu’il modifie le cadrage, la portée ou la comparabilité de la divulgation ; "
        "dans ce cas, considère `nouvelle_idee=true`.\n"
        f"3. `nouvelle_idee=true` seulement si {bank_subject} ajoute, retire "
        "ou modifie substantiellement une information qui n’était pas divulguée "
        "auparavant sous cette forme. Pour la gouvernance, considère comme substantiel "
        "tout changement démontré d’autorité décisionnelle, de mandat ou de rôle "
        "d’un comité, de ligne de défense, de responsabilité, de supervision, de reddition de "
        "comptes, de périodicité de reporting ou de suivi prudentiel, de culture de risque, "
        "de rémunération liée au risque (y compris les comportements attendus) ou "
        "d’appétit pour le risque. Une phrase courte peut donc être une nouvelle idée; un "
        "simple renommage sans effet sur le mandat ne l’est pas.\n"
        "   Une modification réelle de méthodologie ou de processus est toujours "
        "substantielle et prioritaire : utilise `MODIFICATION_METHODOLOGIE` pour "
        "la méthode ou l’approche, et `CONTROLE_CONFORMITE` pour un processus de "
        "contrôle ou de conformité, avec `nouvelle_idee=true`. Une reformulation "
        "qui ne change ni le fonctionnement, ni les étapes, ni les acteurs, ni les "
        "contrôles demeure non substantielle.\n"
        "4. Chaque champ renseigné doit être non vide, lexical et terminé par "
        "« . », « ! » ou « ? ». Si `is_relevant=true`, renseigne "
        "`changement_constate`, `signification_metier`, "
        "`comparaison_interbanques` et `limite_interpretation`, puis laisse "
        "`motif_non_pertinence` vide. `changement_constate` décrit factuellement "
        f"l’action de {bank_subject}; `signification_metier` explique sa "
        "signification concrète; `comparaison_interbanques` précise les dimensions "
        "à comparer entre banques; `limite_interpretation` indique uniquement ce "
        "que la preuve ne démontre ou ne précise pas. Si `is_relevant=false`, "
        "renseigne seulement `changement_constate` et `motif_non_pertinence`, puis "
        "laisse les trois champs analytiques vides. N’écris pas "
        "« Ce changement est pertinent pour la vigie AMF », « Ce changement "
        "n’est pas pertinent », « Pour la vigie », « Cette information est "
        "importante », « Il convient de noter que » ni « Dans le cadre de cette "
        "analyse ». "
        "Aucun titre, aucune liste, aucune rubrique et aucune consigne adressée "
        "à l’analyste. Interdit : fragment, chunk, T1, T2, termes anglais.\n"
        "5. Ne produis aucun champ d’impact, d’action, de posture, d’impact IT, "
        "d’explication générale, de justification multi-rubriques ou "
        "`relevance_reason`.\n\n"
        f"Adapte les exemples à la banque analysée : remplace toujours leur sujet "
        f"par {bank_subject} dans la réponse réelle.\n\n"
        f"{_FEW_SHOT_TRIAGE_AMF}\n\n"
        f"Banque analysée : {bank_subject}\n"
        f"Section : {section_key}\n"
        f"Changements :\n{_json_dumps(triage_inputs)}"
    )
    compact_max_tokens = min(
        _COMPACT_COMPLETION_MAX_TOKENS,
        _COMPACT_COMPLETION_BASE_TOKENS + _COMPACT_COMPLETION_TOKENS_PER_CHANGE * len(changes),
    )

    try:
        batch = _call_structured_completion_with_correction(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TriageAMFCompactLLMBatch,
            max_tokens=compact_max_tokens,
            max_retries=2,
            validation_retry_message=(
                "Renvoie le batch compact complet. Chaque change_index doit être "
                "présent exactement une fois. is_relevant=true exige un ou deux "
                "thèmes AMF (préfère candidate_themes, sinon tout code de la "
                "taxonomie AMF, sinon SUJET_EMERGENT_HORS_GRILLE); "
                "is_relevant=false exige themes_amf=[] et "
                "nouvelle_idee=false. Corrige uniquement les cinq champs "
                "sémantiques : is_relevant=true exige changement_constate, "
                "signification_metier, comparaison_interbanques et "
                "limite_interpretation non vides, avec motif_non_pertinence vide; "
                "is_relevant=false exige changement_constate et "
                "motif_non_pertinence non vides, avec les trois autres champs "
                f"vides. Chaque changement_constate commence par {bank_subject} "
                "et chaque champ renseigné est lexical et ponctué. Ne produis "
                "pas relevance_reason."
            ),
            length_retry_message=(
                "Renvoie immédiatement le même batch compact complet, sans aucun "
                "commentaire hors schéma. Raccourcis séparément les champs "
                "changement_constate, signification_metier, "
                "comparaison_interbanques, limite_interpretation et "
                "motif_non_pertinence sans les fusionner. Respecte les champs "
                f"vides applicables et commence changement_constate par "
                f"{bank_subject}. Ne produis pas relevance_reason."
            ),
        )
    except ValidationError as exc:
        raise TriageValidationError(
            section_key=section_key,
            change_index=_change_index_from_validation_error(exc),
            raw_payload=None,
            validation_error=exc,
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Section triage failed for {section_key}: {exc}") from exc

    expected_indexes = list(range(1, len(changes) + 1))
    received_indexes = [triage.change_index for triage in batch.triages]
    if len(received_indexes) != len(expected_indexes) or sorted(received_indexes) != expected_indexes:
        validation_error = ValueError(
            f"Le batch compact doit contenir exactement les change_index {expected_indexes}; reçu {received_indexes}"
        )
        raise TriageValidationError(
            section_key=section_key,
            change_index=None,
            raw_payload=batch.model_dump(),
            validation_error=validation_error,
        )

    for triage_obj in batch.triages:
        candidate_codes = {
            candidate["code"] for candidate in triage_inputs[triage_obj.change_index - 1]["candidate_themes"]
        }
        raw_themes = list(triage_obj.themes_amf or [])
        outside_candidates = [
            code
            for code in raw_themes
            if str(code or "").strip().upper() not in candidate_codes
            and str(code or "").strip().upper() in THEMES_AMF_DESCRIPTIONS
        ]
        if outside_candidates:
            logger.debug(
                "theme_accepted_outside_candidates section=%s change_index=%d themes=%s",
                section_key,
                triage_obj.change_index,
                outside_candidates,
            )
        # Soft-normalize: full AMF taxonomy accepted; unknown -> hors grille.
        triage_obj.themes_amf = _normalize_themes_amf(raw_themes)

    triage_map: dict[int, dict[str, Any]] = {}
    relevant_count = 0
    nouvelle_idee_count = 0
    for triage_obj in batch.triages:
        change = changes[triage_obj.change_index - 1]
        compact_dict = triage_obj.model_dump(exclude={"change_index"})
        triage = _persisted_triage_from_compact(
            compact_dict,
            change=change,
            bank_code=effective_bank_code,
        )
        triage["change_segments"] = (
            exact_segments_by_index.get(triage_obj.change_index, []) if triage_obj.is_relevant else []
        )
        triage_map[triage_obj.change_index] = triage
        if triage_obj.is_relevant:
            relevant_count += 1
        if triage_obj.nouvelle_idee:
            nouvelle_idee_count += 1
        logger.info(
            "compact triage validated section=%s change_index=%d is_relevant=%s themes=%s nouvelle_idee=%s semantic_fields=%s",
            section_key,
            triage_obj.change_index,
            triage_obj.is_relevant,
            triage_obj.themes_amf,
            triage_obj.nouvelle_idee,
            [field_name for field_name in _SEMANTIC_REASON_FIELDS if getattr(triage_obj, field_name)],
        )

    logger.info(
        "triage section summary section=%s total=%d relevant=%d nouvelles_idees=%d",
        section_key,
        len(batch.triages),
        relevant_count,
        nouvelle_idee_count,
    )

    enriched: list[dict[str, Any]] = []
    for idx, change in enumerate(changes, start=1):
        evidence_failure_reason = full_evidence_failures_by_index.get(idx)
        if evidence_failure_reason:
            enriched.append(
                _evidence_read_review_triage(
                    change,
                    evidence_failure_reason,
                    bank_code=effective_bank_code,
                )
            )
            continue
        triage = triage_map.get(idx, _default_triage(effective_bank_code))
        evidence_observations = full_evidence_by_index.get(idx, [])
        if evidence_observations:
            coherent, coherence_reason = _verify_triage_coherence(
                client=client,
                model=model,
                change=change,
                triage=triage,
                evidence_packets=full_evidence_packets_by_index[idx],
            )
            if not coherent:
                enriched.append(
                    _coherence_review_triage(
                        change,
                        coherence_reason,
                        bank_code=effective_bank_code,
                    )
                )
                continue
            triage["full_evidence_verified"] = True
            triage["full_evidence_observations"] = evidence_observations
        enriched_change = dict(change)
        enriched_change["genai_triage"] = triage
        enriched.append(enriched_change)
    return [*prefiltered, *enriched]

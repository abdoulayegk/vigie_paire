"""Filet de securite deterministe pour les differences de tableaux."""

from __future__ import annotations

from typing import Any

from vigie.comparaison.differences.normalisation_elements import (
    _normalize_footnote_text,
    _normalize_indicator_text,
    _token_overlap_ratio,
)


def _deterministic_indicator_diff(
    prev_indicators: list[str],
    curr_indicators: list[str],
    *,
    fuzzy_threshold: float = 0.80,
) -> dict[str, Any]:
    """Calcule le diff d'indicateurs par ensemble avant l'appel GPT."""
    prev_norm = {_normalize_indicator_text(ind): ind for ind in prev_indicators}
    curr_norm = {_normalize_indicator_text(ind): ind for ind in curr_indicators}

    prev_keys = set(prev_norm.keys())
    curr_keys = set(curr_norm.keys())

    only_prev = prev_keys - curr_keys
    only_curr = curr_keys - prev_keys

    # Attempt fuzzy matching between the unmatched sets
    det_renamed: list[dict[str, str]] = []
    matched_prev: set[str] = set()
    matched_curr: set[str] = set()
    for pkey in sorted(only_prev):
        best_score = 0.0
        best_ckey = ""
        for ckey in sorted(only_curr):
            if ckey in matched_curr:
                continue
            score = _token_overlap_ratio(pkey, ckey)
            if score > best_score:
                best_score = score
                best_ckey = ckey
        if best_score >= fuzzy_threshold and best_ckey:
            det_renamed.append({"previous": prev_norm[pkey], "current": curr_norm[best_ckey]})
            matched_prev.add(pkey)
            matched_curr.add(best_ckey)

    det_removed = [prev_norm[k] for k in sorted(only_prev - matched_prev)]
    det_added = [curr_norm[k] for k in sorted(only_curr - matched_curr)]

    return {
        "det_added": det_added,
        "det_removed": det_removed,
        "det_renamed": det_renamed,
    }


def _deterministic_footnote_diff(
    prev_footnotes: list[dict[str, str]],
    curr_footnotes: list[dict[str, str]],
) -> dict[str, Any]:
    """Calcule le diff de notes de bas de page par IDENTITE DE CONTENU.

    L'identifiant (1), (2), (3)... n'est pas une identite stable : il change
    des qu'une note est inseree ou supprimee plus haut dans la liste. On
    apparie donc d'abord par texte normalise (meme meaning = meme note, meme
    si le numero a change). L'id ne sert qu'a (a) preferer une paire au meme
    numero quand le texte est identique a plusieurs endroits, et (b) detecter
    une revision de wording a numero stable (``det_modified``).
    """
    prev_norm = [_normalize_footnote_text(fn.get("text", "")) for fn in prev_footnotes]
    curr_norm = [_normalize_footnote_text(fn.get("text", "")) for fn in curr_footnotes]
    prev_ids = [str(fn.get("id", "") or "").strip() for fn in prev_footnotes]
    curr_ids = [str(fn.get("id", "") or "").strip() for fn in curr_footnotes]

    prev_matched = [False] * len(prev_footnotes)
    curr_matched = [False] * len(curr_footnotes)

    # Pass 1: appariement par texte normalise. On prefere un appariement au
    # meme id pour preserver les notes stables ; a defaut, n'importe quelle
    # correspondance de texte (cas typique : renumerotation suite a une
    # insertion/suppression plus haut dans la liste).
    for pi, p_text in enumerate(prev_norm):
        if prev_matched[pi] or not p_text:
            continue
        same_id_idx = -1
        any_id_idx = -1
        for ci, c_text in enumerate(curr_norm):
            if curr_matched[ci] or c_text != p_text:
                continue
            if prev_ids[pi] and curr_ids[ci] == prev_ids[pi]:
                same_id_idx = ci
                break
            if any_id_idx < 0:
                any_id_idx = ci
        match_idx = same_id_idx if same_id_idx >= 0 else any_id_idx
        if match_idx >= 0:
            prev_matched[pi] = True
            curr_matched[match_idx] = True

    det_added: list[dict[str, str]] = [
        curr_footnotes[ci] for ci in range(len(curr_footnotes)) if not curr_matched[ci]
    ]
    det_removed: list[dict[str, str]] = [
        prev_footnotes[pi] for pi in range(len(prev_footnotes)) if not prev_matched[pi]
    ]

    # Pass 2: revision de wording a numero stable. Si apres l'appariement par
    # texte il reste un residu prev et un residu curr partageant le meme id,
    # on les classe comme modifies (meme position, texte materiellement
    # different) plutot que added+removed.
    det_modified: list[dict[str, Any]] = []
    still_removed: list[dict[str, str]] = []
    for prev_fn in det_removed:
        prev_id = str(prev_fn.get("id", "") or "").strip()
        if not prev_id:
            still_removed.append(prev_fn)
            continue
        match_idx = -1
        for i, curr_fn in enumerate(det_added):
            curr_id = str(curr_fn.get("id", "") or "").strip()
            if curr_id == prev_id:
                match_idx = i
                break
        if match_idx >= 0:
            curr_fn = det_added.pop(match_idx)
            det_modified.append(
                {
                    "previous_id": prev_id,
                    "current_id": str(curr_fn.get("id", "") or "").strip(),
                    "previous_text": prev_fn.get("text", ""),
                    "current_text": curr_fn.get("text", ""),
                }
            )
        else:
            still_removed.append(prev_fn)
    det_removed = still_removed

    return {
        "det_added": det_added,
        "det_removed": det_removed,
        "det_modified": det_modified,
    }

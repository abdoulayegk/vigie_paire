"""Localisation structurelle de la TDM annuelle (T4) hors mots-cles lexicaux.

Ancre la table des matieres sur la page d'ouverture « Rapport de gestion »,
rejette le leurre EDTF/GTDAR par structure, lit les entrees multi-colonnes
(titre + page imprimee) et prepare l'offset imprime -> physique.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from vigie.extraction.localisation_sections.models import normalize_text

logger = logging.getLogger(__name__)

_RG_TITLE_RE = re.compile(
    r"^(rapport\s+de\s+gestion|management'?s?\s+discussion\s+and\s+analysis|"
    r"\bmda\b)$",
    re.IGNORECASE,
)
_PRINTED_PAGE_TAIL_RE = re.compile(r"^(?P<title>.+?)\s+[.\u2022\u00b7\-]{0,40}\s*(?P<page>\d{1,3})\s*$")
_LEADING_INDEX_RE = re.compile(r"^\d{1,2}[\.\)]\s+")


@dataclass(slots=True)
class TocStructureEntry:
    """Entree TDM avec page imprimee (et physique si resolue)."""

    title: str
    printed_page: int
    physical_page: int | None = None
    source: str = "geometric"


@dataclass(slots=True)
class TocStructure:
    """Structure TDM ancree sur l'ouverture Rapport de gestion."""

    rg_page: int | None
    entries: list[TocStructureEntry] = field(default_factory=list)
    offset: int = 0
    confidence: float = 0.0
    anomalies: list[str] = field(default_factory=list)
    rejected_edtf_pages: list[int] = field(default_factory=list)


def is_edtf_decoy_page(page_text: str) -> bool:
    """Detecter une page EDTF/GTDAR par structure (~lignes numerotees), pas par mot-cle."""
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    if len(lines) < 20:
        return False
    numbered = 0
    sequential_hits = 0
    last_idx: int | None = None
    for line in lines:
        match = re.match(r"^(\d{1,2})[\.\)]\s+", line)
        if not match:
            continue
        numbered += 1
        idx = int(match.group(1))
        if last_idx is not None and idx == last_idx + 1:
            sequential_hits += 1
        last_idx = idx
    # Table EDTF typique: ~32 items numerotes 1..N en sequence dense.
    if numbered >= 20 and sequential_hits >= 12:
        return True
    if numbered >= 24 and numbered / max(len(lines), 1) >= 0.45:
        return True
    return False


def find_management_report_opening_page(
    text_by_page: dict[int, str],
    *,
    max_page: int = 35,
) -> tuple[int | None, list[int]]:
    """Trouver la page d'ouverture « Rapport de gestion » et les pages EDTF rejetees."""
    rejected_edtf: list[int] = []
    best_page: int | None = None
    best_score = -1.0
    upper = min(int(max_page), max(text_by_page.keys(), default=0))
    for page_num in range(1, upper + 1):
        page_text = text_by_page.get(page_num, "")
        if not page_text:
            continue
        if is_edtf_decoy_page(page_text):
            rejected_edtf.append(page_num)
            continue
        score = _score_rg_opening_page(page_text, page_num)
        if score > best_score:
            best_score = score
            best_page = page_num
    if best_score < 20.0:
        return None, rejected_edtf
    return best_page, rejected_edtf


def _score_rg_opening_page(page_text: str, page_num: int) -> float:
    """Scorer une page candidate d'ouverture Rapport de gestion."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    score = 0.0
    top = lines[:12]
    for idx, line in enumerate(top):
        norm = normalize_text(line).strip()
        compact = re.sub(r"\s+", " ", norm)
        if _RG_TITLE_RE.match(compact) or compact.startswith("rapport de gestion"):
            score += 80.0 if idx <= 3 else 50.0
            if len(line) <= 48:
                score += 15.0
        if "management discussion" in compact or compact == "mda":
            score += 70.0 if idx <= 3 else 40.0
    # Une vraie ouverture RG porte souvent une mini-TDM (titres + pages).
    toc_like = 0
    for line in lines:
        if _PRINTED_PAGE_TAIL_RE.match(line) and len(line) < 160:
            toc_like += 1
    score += min(toc_like, 10) * 4.0
    if 10 <= page_num <= 28:
        score += 8.0
    return score


def collapse_letter_spaced_text(text: str) -> str:
    """Reconstituer les titres lettres espacees (ex. « G e s t i o n » -> « Gestion »)."""
    tokens = [tok for tok in str(text or "").split() if tok]
    if len(tokens) < 4:
        return " ".join(tokens)
    short_alpha = sum(1 for tok in tokens if tok.isalpha() and len(tok) <= 2)
    if short_alpha < max(4, int(0.45 * len(tokens))):
        return " ".join(tokens)

    merged: list[str] = []
    buffer = ""
    for tok in tokens:
        if tok.isdigit():
            if buffer:
                merged.append(buffer)
                buffer = ""
            # Fusionner les chiffres adjacents d'un numero de page (7 2 -> 72).
            if merged and merged[-1].isdigit() and len(merged[-1]) <= 2 and len(tok) <= 2:
                merged[-1] = merged[-1] + tok
            else:
                merged.append(tok)
            continue
        if tok.isalpha() and len(tok) <= 2:
            buffer += tok
            continue
        if buffer:
            merged.append(buffer)
            buffer = ""
        merged.append(tok)
    if buffer:
        merged.append(buffer)
    collapsed = " ".join(merged)
    return _rehydrate_compact_title(collapsed)


def _compact_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


def _rehydrate_compact_title(text: str) -> str:
    """Reintroduire les espaces d'un titre compact via prototypes connus."""
    parts = text.rsplit(" ", 1)
    page_suffix = ""
    body = text
    if len(parts) == 2 and parts[1].isdigit():
        body, page_suffix = parts[0], parts[1]
    compact = _compact_norm(body)
    if len(compact) < 8:
        return text
    try:
        from vigie.extraction.localisation_sections.patterns import SECTION_TITLE_ALIASES, T4_SECTION_TITLE_PROFILES
    except Exception:
        return text
    candidates: list[str] = []
    for titles in SECTION_TITLE_ALIASES.values():
        candidates.extend(titles)
    for bank_profile in T4_SECTION_TITLE_PROFILES.values():
        for section_profile in bank_profile.values():
            candidates.extend(section_profile.get("start", []))
            candidates.extend(section_profile.get("end", []))
    best = ""
    for cand in candidates:
        cand_compact = _compact_norm(cand)
        if not cand_compact:
            continue
        if compact == cand_compact or compact.startswith(cand_compact) or cand_compact in compact:
            if len(cand_compact) > len(_compact_norm(best)):
                best = cand
    if best and abs(len(_compact_norm(best)) - len(compact)) <= 4:
        return f"{best} {page_suffix}".strip() if page_suffix else best
    return text


def parse_toc_entries_from_text(page_text: str) -> list[TocStructureEntry]:
    """Parser les entrees titre+page depuis un flux texte (y compris multi-colonnes aplaties)."""
    entries: list[TocStructureEntry] = []
    seen: set[tuple[str, int]] = set()
    for raw_line in str(page_text or "").splitlines():
        line = collapse_letter_spaced_text(raw_line.strip())
        if len(line) < 5 or len(line) > 220:
            continue
        # Une ligne multi-colonnes peut contenir plusieurs couples titre+page.
        segments = _split_multi_toc_segments(line)
        for segment in segments:
            match = _PRINTED_PAGE_TAIL_RE.match(segment)
            if not match:
                continue
            title = match.group("title").strip(" .\u2022\u00b7-")
            title = _LEADING_INDEX_RE.sub("", title).strip()
            title = collapse_letter_spaced_text(title)
            if len(title) < 4:
                continue
            # Titres encore lettres-espacees residuelles: trop peu de lettres utiles.
            if len(normalize_text(title).replace(" ", "")) < 6:
                continue
            page = int(match.group("page"))
            if page <= 0 or page > 500:
                continue
            key = (normalize_text(title), page)
            if key in seen:
                continue
            seen.add(key)
            entries.append(TocStructureEntry(title=title, printed_page=page, source="text"))
    return entries


def _split_multi_toc_segments(line: str) -> list[str]:
    """Decouper une ligne aplatie multi-colonnes en segments titre+page."""
    matches = list(re.finditer(r"\b(\d{1,3})\b", line))
    if len(matches) <= 1:
        return [line]
    segments: list[str] = []
    start = 0
    for match in matches:
        page = int(match.group(1))
        if page <= 0 or page > 500:
            continue
        end = match.end()
        segment = line[start:end].strip()
        # Ignorer les segments encore trop longs (pollution multi-colonnes).
        if 5 <= len(segment) <= 90:
            segments.append(segment)
        start = end
    return segments or [line]


def parse_toc_entries_geometric(pdf_path: Path, page_num: int) -> list[TocStructureEntry]:
    """Lire une page TDM via pdfplumber en regroupant par colonnes puis lignes."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber indisponible pour parse geometrique TDM")
        return []

    entries: list[TocStructureEntry] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False) or []
            if not words:
                return []
            page_width = float(page.width or 1.0)
            xs = [float(w.get("x0", 0.0)) for w in words]
            if not xs:
                return []
            span = max(xs) - min(xs)
            n_cols = 3 if span > page_width * 0.55 else 2
            col_width = page_width / n_cols
            columns: list[list[dict]] = [[] for _ in range(n_cols)]
            for word in words:
                x0 = float(word.get("x0", 0.0))
                col_idx = min(n_cols - 1, max(0, int(x0 / max(col_width, 1.0))))
                columns[col_idx].append(word)

            for col_words in columns:
                entries.extend(_entries_from_column_words(col_words))
    except Exception as exc:
        logger.warning("Parse geometrique TDM impossible p.%s: %s", page_num, exc)
        return []
    return _dedupe_entries(entries)


def _entries_from_column_words(words: list[dict]) -> list[TocStructureEntry]:
    """Reconstruire les lignes d'une colonne puis extraire titre+page."""
    if not words:
        return []
    # Tolerance Y serree pour ne pas fusionner deux titres lettres-espacees empiles.
    words = sorted(words, key=lambda w: (round(float(w.get("top", 0.0)) / 1.5), float(w.get("x0", 0.0))))
    lines: list[list[dict]] = []
    current: list[dict] = []
    current_y: float | None = None
    for word in words:
        top = float(word.get("top", 0.0))
        if current_y is None or abs(top - current_y) <= 1.6:
            current.append(word)
            current_y = top if current_y is None else (current_y + top) / 2.0
        else:
            if current:
                lines.append(current)
            current = [word]
            current_y = top
    if current:
        lines.append(current)

    entries: list[TocStructureEntry] = []
    pending_title_parts: list[str] = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: float(w.get("x0", 0.0)))
        raw_tokens = [str(w.get("text") or "") for w in line_words]
        text = collapse_letter_spaced_text(" ".join(raw_tokens)).strip()
        if not text:
            continue
        # Titre wrappe sans numero: garder pour fusion avec la ligne suivante.
        if not re.search(r"\d{1,3}\s*$", text):
            if len(text) >= 4 and not text.isdigit():
                pending_title_parts.append(text)
            continue
        if pending_title_parts:
            text = " ".join([*pending_title_parts, text])
            pending_title_parts = []
        parsed = parse_toc_entries_from_text(text)
        for entry in parsed:
            entry.source = "geometric"
            entries.append(entry)
    return entries


def _dedupe_entries(entries: list[TocStructureEntry]) -> list[TocStructureEntry]:
    """Dedoublonner; preferer le titre le plus propre (anti-fusion multi-colonnes)."""
    out: list[TocStructureEntry] = []
    for entry in entries:
        entry_norm = normalize_text(entry.title)
        if len(entry_norm) > 90:
            continue
        replaced = False
        for idx, existing in enumerate(out):
            if int(existing.printed_page) != int(entry.printed_page):
                continue
            existing_norm = normalize_text(existing.title)
            if entry_norm == existing_norm:
                replaced = True
                break
            if entry_norm in existing_norm or existing_norm in entry_norm:
                # Si l'un contient l'autre, garder le plus court (moins de pollution colonnes).
                prefer_entry = len(entry_norm) < len(existing_norm)
                # Sauf si le plus court est un suffixe tronque trop faible (< 18).
                if prefer_entry and len(entry_norm) >= 12:
                    out[idx] = entry
                elif (not prefer_entry) and len(existing_norm) < 12 and len(entry_norm) >= 18:
                    out[idx] = entry
                replaced = True
                break
        if not replaced:
            out.append(entry)
    return out


def resolve_printed_to_physical_offset(
    text_by_page: dict[int, str],
    entries: list[TocStructureEntry],
    *,
    configured_offset: int = 0,
    scan_from: int = 40,
    preferred_titles: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Estimer offset imprime->physique en ancrant un titre de chapitre fort."""
    anomalies: list[str] = []
    if configured_offset:
        return int(configured_offset), anomalies

    preferred_norm = [normalize_text(t).strip() for t in (preferred_titles or []) if t]
    preferred_compact = {_compact_norm(t) for t in preferred_norm if len(_compact_norm(t)) >= 12}

    ranked: list[TocStructureEntry] = []
    for entry in entries:
        title_norm = normalize_text(entry.title).strip()
        title_compact = _compact_norm(title_norm)
        if len(title_norm) < 10:
            continue
        if (
            preferred_compact
            and title_compact not in preferred_compact
            and not any(
                title_compact == p or (len(title_compact) >= 14 and title_compact in p) for p in preferred_compact
            )
        ):
            continue
        ranked.append(entry)
    if not ranked:
        ranked = [e for e in entries if len(normalize_text(e.title).strip()) >= 18][:8]

    total_pages = max(text_by_page.keys(), default=0)
    votes: list[tuple[int, str]] = []
    for entry in ranked[:12]:
        title_norm = normalize_text(entry.title).strip()
        title_compact = _compact_norm(title_norm)
        for phys in range(max(scan_from, 1), total_pages + 1):
            page_text = text_by_page.get(phys, "")
            if not page_text:
                continue
            top_lines = [line.strip() for line in page_text.splitlines() if line.strip()][:12]
            matched = False
            for line in top_lines:
                line_norm = normalize_text(line).strip()
                line_compact = _compact_norm(line_norm)
                # Exiger egalite (ou compact egal) sur une ligne courte de titre.
                if line_norm == title_norm or (
                    title_compact and line_compact == title_compact and len(line) <= max(80, len(entry.title) + 20)
                ):
                    matched = True
                    break
            if matched:
                offset = int(phys) - int(entry.printed_page)
                if 0 <= offset <= 15:
                    votes.append((offset, entry.title))
                else:
                    anomalies.append(f"offset_out_of_range:{entry.title}:{offset}")
                break
    if votes:
        # Preferer l'offset le plus frequent; a egalite, le plus petit.
        counts: dict[int, int] = {}
        for offset, _ in votes:
            counts[offset] = counts.get(offset, 0) + 1
        best_offset = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return best_offset, anomalies
    if entries:
        anomalies.append("offset_unresolved_using_zero")
    return 0, anomalies


def apply_offset_to_entries(entries: list[TocStructureEntry], offset: int) -> list[TocStructureEntry]:
    """Renseigner physical_page = printed_page + offset."""
    out: list[TocStructureEntry] = []
    for entry in entries:
        out.append(
            TocStructureEntry(
                title=entry.title,
                printed_page=entry.printed_page,
                physical_page=int(entry.printed_page) + int(offset),
                source=entry.source,
            )
        )
    return out


def locate_toc_structure(
    text_by_page: dict[int, str],
    *,
    pdf_path: str | Path | None = None,
    configured_offset: int = 0,
) -> TocStructure:
    """Localiser la TDM structurelle T4 et retourner TocStructure."""
    anomalies: list[str] = []
    rg_page, rejected_edtf = find_management_report_opening_page(text_by_page)
    if rg_page is None:
        anomalies.append("rg_opening_not_found")
        return TocStructure(
            rg_page=None,
            entries=[],
            offset=0,
            confidence=0.0,
            anomalies=anomalies,
            rejected_edtf_pages=rejected_edtf,
        )

    entries: list[TocStructureEntry] = []
    path = Path(pdf_path) if pdf_path else None
    pages_to_read = [rg_page]
    if rg_page + 1 in text_by_page:
        pages_to_read.append(rg_page + 1)

    for page in pages_to_read:
        geometric: list[TocStructureEntry] = []
        if path is not None and path.exists():
            geometric = parse_toc_entries_geometric(path, page)
            entries.extend(geometric)
        # Toujours fusionner le parse texte: complete les titres lettres-espacees rates.
        entries.extend(parse_toc_entries_from_text(text_by_page.get(page, "")))

    entries = _dedupe_entries(entries)
    if len(entries) < 3:
        anomalies.append("toc_entries_sparse")

    from vigie.extraction.localisation_sections.patterns import SECTION_TITLE_ALIASES, T4_SECTION_TITLE_PROFILES

    preferred_titles: list[str] = []
    for section_key in ("gestion_capital", "gestion_risques"):
        preferred_titles.extend(SECTION_TITLE_ALIASES.get(section_key, []))
        for bank_profile in T4_SECTION_TITLE_PROFILES.values():
            preferred_titles.extend(bank_profile.get(section_key, {}).get("start", []))

    offset, offset_anomalies = resolve_printed_to_physical_offset(
        text_by_page,
        entries,
        configured_offset=configured_offset,
        preferred_titles=preferred_titles,
    )
    anomalies.extend(offset_anomalies)
    entries = apply_offset_to_entries(entries, offset)

    confidence = 0.35
    if entries:
        confidence += min(0.45, 0.05 * len(entries))
    if rejected_edtf:
        confidence += 0.1
    if "toc_entries_sparse" in anomalies:
        confidence -= 0.2
    confidence = max(0.0, min(1.0, confidence))

    logger.info(
        "TDM structurelle: rg_page=%s entries=%d offset=%d confidence=%.2f edtf_rejected=%s",
        rg_page,
        len(entries),
        offset,
        confidence,
        rejected_edtf,
    )
    return TocStructure(
        rg_page=rg_page,
        entries=entries,
        offset=offset,
        confidence=confidence,
        anomalies=anomalies,
        rejected_edtf_pages=rejected_edtf,
    )

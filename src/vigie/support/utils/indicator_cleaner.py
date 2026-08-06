"""Utilitaires de nettoyage des libelles d'indicateurs et titres de tableaux."""

from __future__ import annotations

import logging
import re
import unicodedata

from vigie.support.utils.indicator_line_merge import (
    IndicatorLineMergeConfig,
    merge_indicator_lines,
)
from vigie.support.utils.matching_normalizer import (
    _UNIT_HEADER_RE,
    _UNIT_MILLIERS_ACTIONS_RE,
    is_date_only_line,
    is_non_indicator_line,
    normalize_label,
    strip_temporal_expressions,
)

logger = logging.getLogger(__name__)

# Extended pattern for trailing note markers: (1), [2], *, (a), (A), a), †, ‡, ¹²³
_TRAILING_NOTE_RE = re.compile(
    r"\s*(?:"
    r"\(\d+\)"  # (1), (2)
    r"|\[\d+\]"  # [1], [2]
    r"|\([a-zA-Z]\)"  # (a), (b), (A), (B)
    r"|[a-zA-Z]\)"  # a), b)
    r"|[\*†‡]+"  # *, †, ‡
    r"|[¹²³⁴⁵⁶⁷⁸⁹⁰]+"  # superscript digits
    r")\s*$"
)
_TRAILING_NOTE_CLUSTER_RE = re.compile(
    r"(?:\s*,?\s*(?:[\(\[]\d+[\)\]]|\([a-zA-Z]\)|[a-zA-Z]\)|[\*†‡]+|[¹²³⁴⁵⁶⁷⁸⁹⁰]+))+[\s,;:.]*$"
)
_TRAILING_NUM_RE = re.compile(r"\s+\d{1,4}(?:[.,]\d+)?\s*$")
# Digits attached to last word (no space): "Total des actifs1" -> "Total des actifs", "Revenue2024" -> "Revenue"
_TRAILING_WORD_DIGITS_RE = re.compile(r"([^\d\s]+)(\d+)$")
_DATE_IN_TITLE_RE = re.compile(r"\b(?:au|as at|for the quarter ended|trimestre termine le)\b.*$", re.IGNORECASE)
# CIBC-style dates at start or end: "31 janvier 2025", "30 avril 2025 1, 2"
_DATE_STANDALONE_RE = re.compile(
    r"^\s*\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
    r"septembre|octobre|novembre|decembre)(?:\s+\d{4})?\s*$",
    re.IGNORECASE,
)
_LEADING_DATE_RE = re.compile(
    r"^\s*\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
    r"septembre|octobre|novembre|decembre)(?:\s+\d{4})?\s+",
    re.IGNORECASE,
)
_TRAILING_DATE_RE = re.compile(
    r"\s+\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
    r"septembre|octobre|novembre|decembre)(?:\s+\d{4})?\s*$",
    re.IGNORECASE,
)

# RBC header/footer: "24 Banque Royale du Canada Premier trimestre de 2025"
_HEADER_FOOTER_RBC_RE = re.compile(
    r"(?:\d+\s+)?Banque\s+Royale\s+du\s+Canada\s+"
    r"(?:Premier|Deuxieme|Deuxième|Troisieme|Troisième|Quatrieme|Quatrième)\s+trimestre",
    re.IGNORECASE,
)
_HEADER_FOOTER_PAGE_BANK_RE = re.compile(r"^\s*\d+\s+[A-Za-z].*(?:trimestre|quarter)\b", re.IGNORECASE)

# --- Patterns pour strip_dates_from_indicator_label ---
_MONTHS_FR = (
    r"janv(?:ier)?|fevr(?:ier)?|f[eé]vr(?:ier)?|mars|avr(?:il)?|mai|juin|juill(?:et)?|"
    r"ao[uû]t|aout|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|d[eé]c(?:embre)?"
)
_MONTHS_EN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# Prefixes temporels: au, a (preposition seule), as at, for the period ended, etc.
# "a" doit etre suivi d'espace pour eviter de matcher "Actifs", "Actifs greves", etc.
_INDICATOR_DATE_PREFIX_RE = re.compile(
    r"^\s*(?:au\b|a\s|as\s+at|as\s+of|for\s+the\s+(?:period|quarter)\s+ended|"
    r"trimestre\s+termin[eé]\s+le|pour\s+la\s+periode\s+close\s+le)\s*",
    re.IGNORECASE,
)
# Suffixe date: " au 30 avril 2025", " - 31/01/2025", " (30 avril 2025)"
_INDICATOR_DATE_SUFFIX_RE = re.compile(
    r"(?:[\s\-–—,;]+(?:au|a|as\s+at|as\s+of|le|du)\s+)?"
    r"(?:\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4}"
    r"|\d{4}[\s./-]\d{1,2}[\s./-]\d{1,2}"
    r"|\d{1,2}\s+(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*(?:\d{2,4})?"
    r"|(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*\d{2,4}"
    r"|[tq][1-4]\s+20\d{2}"
    r"|(?:1er|premier|deuxieme|troisieme|quatrieme)\s+trimestre\s+20\d{2}"
    r"|exercice\s+20\d{2})\s*$",
    re.IGNORECASE,
)
# Ligne entiere = date seule (pour early return "")
_INDICATOR_DATE_STANDALONE_RE = re.compile(
    r"^\s*(?:au|a|as\s+at|as\s+of|for\s+the\s+(?:period|quarter)\s+ended)\s+"
    r"(?:\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4}|\d{4}[\s./-]\d{1,2}[\s./-]\d{1,2}|"
    r"\d{1,2}\s+(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*\d{4}?|"
    r"(?:" + _MONTHS_FR + r"|" + _MONTHS_EN + r")[\s.]*\d{4})\s*$",
    re.IGNORECASE,
)

# --- Patterns pour strip_units_currency_from_indicator_label ---
# \b ensures we don't match "en" inside words (e.g. "canadiens")
_UNITS_IN_INDICATOR_PHRASE_RE = re.compile(
    r"[\s\-–—]+(?:"
    r"(?:en\s+)?(?:millions?|milliards?|milliers?)\s+de\s+dollars?\s+canadiens?"
    r"|(?:en\s+)?(?:millions?|milliards?|milliers?)\s+de\s+dollars?"
    r"|(?:en\s+)?(?:millions?|milliards?|milliers?)\s+dollars?"
    r"|en\s+milliers?\s+(?:d\s*['']?\s*actions?|de\s+parts?)"
    r"|en\s+\$|en\s+G\s*\$|en\s+M\s*\$|en\s+k\s*\$"
    r"|(?:en\s+)?pourcentage|points?\s+de\s+base|\bpb\b|\bbps\b"
    r")\s*$",
    re.IGNORECASE,
)
_UNITS_IN_INDICATOR_PAREN_RE = re.compile(
    r"\s*\(\s*(?:"
    r"(?:en\s+)?(?:millions?|milliards?|milliers?)\s*(?:de\s+dollars?)?"
    r"|M\s*\$?|G\s*\$?|k\s*\$?"
    r"|%|CAD|USD|EUR"
    r")\s*\)\s*",
    re.IGNORECASE,
)

_REDUNDANT_PAREN_EXPANSION_RE = re.compile(r"^\s*(?P<prefix>[^()]+?)\s*\(\s*(?P<inner>[^()]+?)\s*\)\s*([.])?\s*$")
_REDUNDANT_SEPARATOR_EXPANSION_RE = re.compile(r"^\s*(?P<prefix>[^:;–—-]+?)\s*[:;–—-]\s*(?P<suffix>.+?)\s*$")


def _normalized_phrase_key(text: str) -> str:
    """Cle de phrase legere pour les verifications de prefixe structurel."""
    if not text:
        return ""
    value = unicodedata.normalize("NFD", text)
    value = value.encode("ascii", "ignore").decode("utf-8")
    value = re.sub(r"[^a-zA-Z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def collapse_redundant_parenthetical_expansion(text: str) -> str:
    """Reduit ``X (X ...)`` en la forme developpee ``X ...``.

    Exemple::

        Fonds propres de categorie 1 (fonds propres de categorie 1 sous forme ...)
        -> fonds propres de categorie 1 sous forme ...

    Neutralise les parenthetiques explicatives qui repetent le meme prefixe
    semantique et gonfleraient sinon les scores de renommage flou.

    Args:
        text: Texte potentiellement contenant une expansion parenthetique redondante.

    Returns:
        Texte reduit si une redondance est detectee, sinon le texte original.
    """
    if not text or "(" not in text or ")" not in text:
        return text or ""
    match = _REDUNDANT_PAREN_EXPANSION_RE.match(text)
    if not match:
        return text or ""

    prefix = re.sub(r"\s+", " ", match.group("prefix") or "").strip(" -:;,")
    inner = re.sub(r"\s+", " ", match.group("inner") or "").strip(" -:;,")
    if not prefix or not inner:
        return text or ""

    prefix_key = _normalized_phrase_key(prefix)
    inner_key = _normalized_phrase_key(inner)
    if not prefix_key or not inner_key:
        return text or ""

    if inner_key == prefix_key:
        return prefix
    if inner_key.startswith(prefix_key + " "):
        return inner
    return text or ""


def collapse_redundant_separator_expansion(text: str) -> str:
    """Reduit ``X : X ...`` ou ``X - X ...`` en le suffixe developpe.

    L'extraction OCR/layout produit parfois un titre court suivi d'un segment
    explicatif repete, separe par ``:``, ``-`` ou un tiret cadratin.
    On conserve la forme developpee quand le suffixe reexprime clairement le meme prefixe.

    Args:
        text: Texte potentiellement contenant une expansion separee par un separateur.

    Returns:
        Texte reduit si une redondance est detectee, sinon le texte original.
    """
    if not text:
        return ""
    match = _REDUNDANT_SEPARATOR_EXPANSION_RE.match(text)
    if not match:
        return text or ""

    prefix = re.sub(r"\s+", " ", match.group("prefix") or "").strip(" -:;,")
    suffix = re.sub(r"\s+", " ", match.group("suffix") or "").strip(" -:;,")
    if not prefix or not suffix:
        return text or ""

    prefix_key = _normalized_phrase_key(prefix)
    suffix_key = _normalized_phrase_key(suffix)
    if not prefix_key or not suffix_key:
        return text or ""

    if suffix_key == prefix_key:
        return prefix
    if suffix_key.startswith(prefix_key + " "):
        return suffix
    return text or ""


def strip_dates_from_indicator_label(text: str) -> str:
    """Supprime les fragments de date/temporels des libelles d'indicateurs (premiere colonne).

    Gere : dates autonomes (``Au 30 avril 2025``), suffixes (``Prets garantis au 30 avril 2025``),
    prefixes et formats multiples (FR, ISO, numerique, abreviations, periodes).
    Retourne une chaine vide si le libelle est purement une expression de date.

    Args:
        text: Libelle d'indicateur potentiellement contamine par des dates.

    Returns:
        Libelle nettoye sans fragments temporels.
    """
    if not text or not (text or "").strip():
        return text or ""
    value = (text or "").strip()
    if _INDICATOR_DATE_STANDALONE_RE.match(value):
        return ""
    value = _INDICATOR_DATE_PREFIX_RE.sub("", value)
    value = _INDICATOR_DATE_SUFFIX_RE.sub("", value)
    value = strip_temporal_expressions(value, target="title", aggressive=True)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def strip_units_currency_from_indicator_label(text: str) -> str:
    """Supprime les mentions d'unite/devise des libelles d'indicateurs (premiere colonne).

    Retire : ``en millions de dollars``, ``(M$)``, ``(CAD)``, ``(%)``, ``en milliers``,
    ``points de base``, etc. Preserve le noyau semantique (ex. ``Ratio CET1`` de ``Ratio CET1 (%)``).

    Args:
        text: Libelle d'indicateur potentiellement contamine par des unites.

    Returns:
        Libelle nettoye sans mentions d'unite ou de devise.
    """
    if not text or not (text or "").strip():
        return text or ""
    value = (text or "").strip()
    value = _UNITS_IN_INDICATOR_PAREN_RE.sub("", value)
    value = _UNITS_IN_INDICATOR_PHRASE_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_header_footer_table_title(title: str, bank_code: str | None = None) -> bool:
    """Retourne True si le titre est un en-tete/pied de page (num page + banque + trimestre), non semantique.

    Args:
        title: Titre de tableau a verifier.
        bank_code: Code de la banque (ex. ``rbc``) pour les heuristiques specifiques.

    Returns:
        True si le titre est un artefact d'en-tete/pied de page.
    """
    if not title or not (title or "").strip():
        return False
    t = (title or "").strip()
    if _HEADER_FOOTER_RBC_RE.search(t):
        return True
    if (bank_code or "").lower() == "rbc" and _HEADER_FOOTER_PAGE_BANK_RE.match(t):
        return True
    return bool(_HEADER_FOOTER_PAGE_BANK_RE.match(t))


def clean_spaced_out_text(text: str) -> str:
    """Recombine les lettres isolees issues de l'OCR : ``T o t a l`` -> ``Total``, ``A s s e t s`` -> ``Assets``.

    Gere les artefacts OCR ou les lettres sont incorrectement separees par des espaces.
    Fusionne uniquement les jetons alphabetiques mono-caractere consecutifs (preserve ``A``
    dans ``Section A``, etc., quand il est suivi d'un jeton multi-caractere).

    Args:
        text: Texte potentiellement contamine par un espacement lettre a lettre.

    Returns:
        Texte avec les lettres isolees recombinees en mots.
    """
    if not text or not text.strip():
        return text or ""
    tokens = text.split()
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if len(tok) == 1 and tok.isalpha():
            letters = [tok]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                letters.append(tokens[j])
                j += 1
            result.append("".join(letters))
            i = j
        else:
            result.append(tok)
            i += 1
    return " ".join(result)


def strip_trailing_note_or_column_value(text: str) -> str:
    """Supprime les marqueurs de note en fin de chaine et les residus de colonnes numeriques.

    Inclut la suppression (style Jad) des chiffres colles au dernier mot (sans espace) :
    ex. ``Total des actifs1`` -> ``Total des actifs``, ``Revenue2024`` -> ``Revenue``.

    Args:
        text: Libelle d'indicateur brut.

    Returns:
        Libelle nettoye sans marqueurs de note ni residus numeriques.
    """
    value = text or ""
    value = _TRAILING_WORD_DIGITS_RE.sub(r"\1", value)
    value = _TRAILING_NOTE_RE.sub("", value)
    value = _TRAILING_NUM_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def is_trailing_number_semantic(text: str) -> bool:
    """Detecte les cas ou les nombres en fin de chaine sont semantiquement pertinents.

    Args:
        text: Libelle d'indicateur a analyser.

    Returns:
        True si les nombres finaux font partie du sens (ex. ``Pilier 1``, ``CET1``).
    """
    value = normalize_label(text)
    if not value:
        return False
    return bool(
        re.search(r"\b(?:pilier|pillar|niveau|tier|phase|etape)\s+[0-9]+\b", value)
        or re.search(r"\b(?:ratio|note|scenario)\s+[0-9]+\b", value)
        or re.search(r"\b(?:serie|series|tranche|classe|class)\s+[0-9]+\b", value)
        or re.search(r"\b(?:categorie|category)\s+[0-9]+\b", value)
        # Compact regulatory tokens without space must be preserved (e.g. CET1, Tier1)
        or re.search(
            r"\b(?:cet|tier|pilier|pillar|at|tlac|lcr|nsfr)\s*[0-9]+\b",
            value,
        )
    )


def strip_dates_from_table_title(title: str) -> str:
    """Supprime les fragments de type date des titres de tableaux (CIBC, RBC).

    Gere : ``au 31 octobre...``, ``31 janvier 2025``, ``30 avril 2025`` en debut/fin.

    Args:
        title: Titre de tableau brut.

    Returns:
        Titre nettoye sans fragments de date.
    """
    value = title or ""
    value = _DATE_IN_TITLE_RE.sub("", value)
    if _DATE_STANDALONE_RE.match(value.strip()):
        return ""
    value = _LEADING_DATE_RE.sub("", value)
    value = _TRAILING_DATE_RE.sub("", value)
    value = strip_temporal_expressions(value, target="title", aggressive=True)
    return re.sub(r"\s+", " ", value).strip()


def strip_note_refs_from_title(title: str) -> str:
    """Supprime les references de notes en fin de titre de tableau (ex. ``NOTATIONS DE CREDIT1``).

    Args:
        title: Titre de tableau brut.

    Returns:
        Titre nettoye sans references de notes.
    """
    value = title or ""
    # Trailing (1), [2], 1), or bare digit 1, 2 when likely a ref
    value = re.sub(r"\s*[\(\[]\d+[\)\]]\s*$", "", value)
    value = re.sub(r"\s*\d+\)\s*$", "", value)
    value = re.sub(r"([a-zA-ZÀ-ÿ])\d+\s*$", r"\1", value)  # CREDIT1 -> CREDIT, garde "Tableau 12"
    value = re.sub(r"\s+\d+\s*,\s*\d+(?:\s*,\s*\d+)*\s*$", "", value)  # " 1, 2", " 1, 2, 3" (virgule requise)
    return re.sub(r"\s+", " ", value).strip()


# --- Table title contamination (amounts / column values in title) ---
# Token is "numeric" if it looks like a number (digits, optional commas/dots, e.g. 79, 772, 1,234.56)
_NUMERIC_TOKEN_RE = re.compile(r"^[\d\s,.]*$")
# Trailing run of 2+ numbers (so we don't strip "Tableau 28"): "79 772 76 163" or "79,772 76,163"
_TRAILING_NUMERIC_RUN_RE = re.compile(r"\s+\d+(?:[.,]\d+)?(?:\s*[,]?\s*\d+(?:[.,]\d+)?)+\s*$")
# Leading run of 2+ numbers at start of title (rare but possible)
_LEADING_NUMERIC_RUN_RE = re.compile(r"^\s*\d+(?:[.,]\d+)?(?:\s*[,]?\s*\d+(?:[.,]\d+)?)+\s+")
# Minimum consecutive numeric tokens to consider a "long" run (amount column)
_NUMERIC_RUN_MIN_LENGTH = 3
# Max fraction of tokens that can be numeric before title is contaminated (0.4 = 40%)
_CONTAMINATION_NUMERIC_TOKEN_RATIO = 0.4
# Minimum trailing numeric tokens to treat as contamination (e.g. "X 79 772")
_CONTAMINATION_TRAILING_NUMBERS_MIN = 2


def _token_is_numeric(tok: str) -> bool:
    """True si le jeton est purement numerique (chiffres, virgules, points, espaces)."""
    if not tok or not tok.strip():
        return False
    # Normalize: remove spaces, then check
    n = tok.replace(" ", "").replace("\u00a0", "")
    return bool(n) and _NUMERIC_TOKEN_RE.match(n)


def is_table_title_contaminated(
    title: str,
    *,
    numeric_ratio_threshold: float = _CONTAMINATION_NUMERIC_TOKEN_RATIO,
    trailing_numbers_min: int = _CONTAMINATION_TRAILING_NUMBERS_MIN,
    long_run_min: int = _NUMERIC_RUN_MIN_LENGTH,
) -> bool:
    """Retourne True si le titre de tableau semble contamine par des valeurs numeriques (montants/colonnes).

    Contamination : longue suite numerique (3+ nombres), ou >40 % des jetons sont des nombres,
    ou se termine par 2+ nombres.

    Args:
        title: Titre de tableau a verifier.
        numeric_ratio_threshold: Ratio maximal de jetons numeriques avant contamination.
        trailing_numbers_min: Nombre minimal de jetons numeriques en fin de titre.
        long_run_min: Longueur minimale d'une suite consecutive de jetons numeriques.

    Returns:
        True si le titre est considere comme contamine.
    """
    value = (title or "").strip()
    if not value:
        return False
    tokens = value.split()
    if not tokens:
        return False

    # Count numeric tokens and find runs
    numeric_count = sum(1 for t in tokens if _token_is_numeric(t))
    if numeric_count >= long_run_min:
        # Check for a consecutive run of 3+ numeric tokens
        run = 0
        for t in tokens:
            if _token_is_numeric(t):
                run += 1
                if run >= long_run_min:
                    return True
            else:
                run = 0

    # > X% of tokens are numbers (and at least 2 numeric tokens to avoid "Tableau 28")
    if numeric_count >= 2 and len(tokens) >= 2 and (numeric_count / len(tokens)) > numeric_ratio_threshold:
        return True

    # Ends with 2+ numeric tokens (column values captured)
    trailing_n = 0
    for t in reversed(tokens):
        if _token_is_numeric(t):
            trailing_n += 1
            if trailing_n >= trailing_numbers_min:
                return True
        else:
            break

    return False


def clean_table_title_contamination(title: str) -> str:
    """Supprime la contamination par montants/valeurs de colonnes d'un titre de tableau.

    - Retire les suites numeriques en fin (nombres multiples, ex. ``79 772 76 163``).
    - Retire les suites numeriques en debut si presentes.
    - Preserve les vrais marqueurs de note : ``(1)``, ``(2)`` en fin sont conserves
      (geres par l'appelant ou ``strip_note_refs``).

    Args:
        title: Titre de tableau brut.

    Returns:
        Titre nettoye sans contamination numerique.
    """
    value = (title or "").strip()
    if not value:
        return value
    # Strip trailing numeric run (space/comma separated numbers)
    value = _TRAILING_NUMERIC_RUN_RE.sub("", value)
    value = _LEADING_NUMERIC_RUN_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


_UNITS_IN_TITLE_RE = re.compile(
    r"\s*[\(\[]?\s*"
    r"(?:en\s+)?(?:millions?|milliards?)\s*(?:de\s+dollars?|\s*\$|dollars?)?"
    r"|[\(\[]?\s*en\s+\$\s*[\)\]]?"
    r"|[\(\[]?\s*(?:en\s+)?dollars?\s*[\)\]]?"
    r"|[\(\[]?\s*%\s*[\)\]]?"
    r"\s*[\)\]]?\s*",
    re.IGNORECASE,
)


# --- Indicators dedupe and line-merge (quality passes) ---
_DEDUPE_DUPLICATE_RATIO_THRESHOLD = 0.15
_LINE_MERGE_MAX_COMBINED_LENGTH = 120


def _normalize_for_dedupe(text: str) -> str:
    """Normalisation des espaces pour la cle de deduplication."""
    return re.sub(r"\s+", " ", (text or "").strip())


def dedupe_indicators(
    indicators: list[str],
    *,
    duplicate_ratio_threshold: float = _DEDUPE_DUPLICATE_RATIO_THRESHOLD,
) -> tuple[list[str], float, int]:
    """Supprime les doublons exacts en preservant l'ordre. Normalise les espaces pour la comparaison.

    Args:
        indicators: Liste de libelles d'indicateurs bruts.
        duplicate_ratio_threshold: Seuil de ratio de doublons pour appliquer la deduplication.

    Returns:
        Tuple ``(liste_dedoublonnee, ratio_doublons, nombre_supprimes)``.
        ``ratio_doublons = 1 - unique/total``. La deduplication n'est appliquee que si
        le ratio >= seuil.
    """
    if not indicators:
        return [], 0.0, 0
    total = len(indicators)
    seen: set[str] = set()
    result: list[str] = []
    for ind in indicators:
        key = _normalize_for_dedupe(ind)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(ind)
    unique = len(result)
    duplicate_ratio = 1.0 - (unique / total) if total > 0 else 0.0
    removed = total - unique
    if duplicate_ratio >= duplicate_ratio_threshold and removed > 0:
        return result, duplicate_ratio, removed
    return indicators, duplicate_ratio, 0


def merge_line_split_indicators(
    indicators: list[str],
    *,
    max_combined_length: int = _LINE_MERGE_MAX_COMBINED_LENGTH,
) -> tuple[list[str], int]:
    """Fusionne les lignes d'indicateurs lorsque la ligne suivante est probablement une continuation.

    Args:
        indicators: Liste de libelles d'indicateurs potentiellement fragmentes.
        max_combined_length: Longueur maximale du libelle fusionne.

    Returns:
        Tuple ``(liste_fusionnee, nombre_fusions)``.
    """
    cfg = IndicatorLineMergeConfig(
        max_next_tokens=6,
        max_combined_length=max_combined_length,
    )
    return merge_indicator_lines(indicators, config=cfg)


def strip_units_from_table_title(title: str, bank_code: str | None = None) -> str:
    """Supprime les mentions d'unite des titres de tableaux pour la comparaison semantique (ex. RBC).

    Retire les fragments en debut/fin comme ``(en millions)``, ``en $``, ``(en millions de dollars)``.

    Args:
        title: Titre de tableau brut.
        bank_code: Code de la banque pour les heuristiques specifiques (ex. ``rbc``).

    Returns:
        Titre nettoye sans mentions d'unite.
    """
    value = title or ""
    value = _UNITS_IN_TITLE_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if bank_code and (bank_code or "").strip().lower() == "rbc":
        # RBC-specific: trim trailing " - long subtitle" beyond first separator
        parts = value.split(" - ", 1)
        if len(parts) == 2 and len(parts[1].split()) > 12:
            value = (parts[0] + " - " + " ".join(parts[1].split()[:12])).strip()
    return value


def normalize_indicator_variants(text: str) -> str:
    """Normalise les variantes d'indicateur en une chaine canonique.

    Args:
        text: Libelle d'indicateur brut.

    Returns:
        Libelle canonique apres suppression des marqueurs de notes et normalisation.
    """
    value = text or ""
    value = _TRAILING_NOTE_CLUSTER_RE.sub("", value)
    value = _TRAILING_NOTE_RE.sub("", value)
    if not is_trailing_number_semantic(value):
        value = _TRAILING_NUM_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return normalize_label(value)


def _word_char_count(s: str) -> int:
    """Compte les caracteres alphanumeriques pour la logique de repli."""
    return len(re.findall(r"\w", s or ""))


# --- PART A: camelCase / concatenated-token boundary splitter ---
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")


def split_camel_case_concatenation(text: str) -> tuple[str, bool]:
    """Insere un espace aux frontieres minuscule-majuscule.

    Gere les jetons concatenes comme ``impotAJOUTactions`` ou ``fondsPropresTier1``.

    Args:
        text: Texte potentiellement en camelCase.

    Returns:
        Tuple ``(texte_nettoye, True si un decoupage a ete applique)``.
    """
    if not text:
        return text or "", False
    result = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", text)
    return result, result != text


# --- PART B: space after change-tag prefix ---
_CHANGE_TAG_PREFIX_RE = re.compile(r"^(AJOUT|SUPPRESSION|RENOMMAGE)(?=[a-zA-ZÀ-ÿ])", re.UNICODE)


def insert_space_after_change_tag(text: str) -> tuple[str, bool]:
    """Insere un espace apres AJOUT/SUPPRESSION/RENOMMAGE si directement suivi d'une lettre.

    Gere les indicateurs comme ``AJOUTactions ordinaires`` -> ``AJOUT actions ordinaires``.

    Args:
        text: Texte potentiellement contenant un tag de changement colle.

    Returns:
        Tuple ``(texte_nettoye, True si une correction a ete appliquee)``.
    """
    if not text:
        return text or "", False
    result = _CHANGE_TAG_PREFIX_RE.sub(r"\1 ", text)
    return result, result != text


def post_normalize_indicator(text: str) -> tuple[str, bool, bool]:
    """Applique le decoupage camelCase et la correction d'espace de tag apres normalisation canonique.

    Doit etre appele sur la valeur d'indicateur *nettoyee*, pas brute.

    Args:
        text: Libelle d'indicateur deja normalise.

    Returns:
        Tuple ``(texte, camel_split_declenche, tag_space_declenche)``.
    """
    if not text:
        return "", False, False
    result, camel = split_camel_case_concatenation(text)
    result, tag = insert_space_after_change_tag(result)
    return result, camel, tag


_SINGULAR_EXCEPTIONS = frozenset(
    {
        "frais",
        "moins",
        "plus",
        "cours",
        "temps",
        "cas",
        "mois",
        "pays",
        "tiers",
        "poids",
        "biais",
        "avis",
        "sens",
        "fonds",
        "corps",
        "bras",
        "choix",
        "voix",
        "prix",
        "noix",
        "index",
        "flux",
        "taux",
        "bas",
        "hors",
        "sous",
        "vers",
        "dans",
        "sans",
        "puis",
        "tres",
        "apres",
        "mais",
        "des",
        "les",
        "ces",
        "ses",
        "nos",
        "vos",
        "tes",
        "mes",
        "aux",
    }
)


def singularize_words(text: str) -> str:
    """Retire le 's' final des mots pour neutraliser les variations pluriel/singulier.

    Gere : ``actions`` -> ``action``, ``actifs`` -> ``actif``, ``garantis`` -> ``garanti``.
    Preserve les mots ou le 's' final est inherent (frais, cours, mois, fonds, etc.).
    N'opere que sur les mots de plus de 3 caracteres pour eviter de casser les mots courts.

    Args:
        text: Texte a singulariser.

    Returns:
        Texte avec les pluriels neutralises.
    """
    if not text:
        return ""
    tokens = text.split()
    result: list[str] = []
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("s") and tok not in _SINGULAR_EXCEPTIONS:
            result.append(tok[:-1])
        else:
            result.append(tok)
    return " ".join(result)


def normalize_indicator_for_comparison(text: str) -> str:
    """Cle canonique unique pour la comparaison d'indicateurs (premiere colonne).

    Utilisee par ``comparison_runner`` et ``structural_comparator`` pour que le meme
    libelle semantique (ex. ``Metaux precieux`` vs ``Metaux precieux :``) produise
    une seule cle et evite les faux ajouts/suppressions dans le detail des changements.

    - Unicode NFD et normalisation de tous les espaces (y compris U+00A0) en espace simple
    - Suppression des dates et unites/devises du libelle (ex. ``au 30 avril 2025``, ``(M$)``)
    - Variantes (CET-1, Tier-1, notes de fin)
    - Suppression des accents, refs de notes ``(1)`` ``[2]`` ``a)`` ``*``, exposants
    - Normalisation de la ponctuation et des espaces, mise en minuscules

    Args:
        text: Libelle d'indicateur brut a normaliser.

    Returns:
        Cle canonique en minuscules, sans accents, prete pour la comparaison.
    """
    if not text:
        return ""
    # Strip leading row/line numbers (e.g. "29 Actifs..." → "Actifs...")
    # that some Basel III / NSFR / LCR tables prefix to each indicator.
    text = re.sub(r"^\d{1,3}\s+(?=[A-ZÀ-Ÿa-zà-ÿ])", "", text)
    # Normalize apostrophe variants early so "d'actions" and "d’actions"
    # follow the same canonical path.
    text = re.sub(r"[’`´ʼʻ＇]", "'", text)

    # Collapse elision+space so "d' impôt" and "d'impôts" yield the same key.
    text = re.sub(r"\b([dljscnmt])'\s+", r"\1'", text, flags=re.IGNORECASE)

    # Normalize typographic quotes to space so they do not create key differences.
    text = re.sub(r'[""\u201c\u201d\u2018\u2019]', " ", text)

    # Normalize Unicode and collapse all whitespace (including U+00A0) to space
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Collapse explanatory forms like "X (X ...)" to a single expanded phrase.
    text = collapse_redundant_parenthetical_expansion(text)
    text = collapse_redundant_separator_expansion(text)

    # Strip series variant suffixes FIRST (before strip_dates removes the year)
    # "Série 2023-g(7)" -> "Série 2023" to avoid OCR confusion (g vs 9, etc.)
    # Must strip combining marks first so é -> e for the regex to match "serie"
    _text_ascii = text.encode("ascii", "ignore").decode("utf-8")
    _text_ascii = re.sub(
        r"(series?\s+\d{4})[-\s]*[a-z0-9]+(?:\(\d+\))?",
        r"\1",
        _text_ascii,
        flags=re.IGNORECASE,
    )
    if _text_ascii != text.encode("ascii", "ignore").decode("utf-8"):
        text = _text_ascii

    # OCR: recombine single-letter 'words' (e.g. "T o t a l" -> "Total", "A s s e t s" -> "Assets")
    text = clean_spaced_out_text(text)

    # Strip footnote reference digits/superscripts glued after closing paren
    # BEFORE strip_dates (which internally drops non-ASCII like ³ but keeps 3).
    # Ensures "amorti)3" and "amorti)³" both become "amorti)" consistently.
    text = re.sub(r"\)\s*[\d¹²³⁴⁵⁶⁷⁸⁹⁰]{1,3}", ")", text)

    # Strip bare trailing footnote digits/superscripts glued to letters (no space).
    # "Région2" → "Région", "Région²" → "Région", but "CET1", "Tier1" kept.
    # Requires 5+ preceding letter chars to avoid stripping compact regulatory tokens.
    text = re.sub(r"(?<=[a-zA-Z]{5})[\d¹²³⁴⁵⁶⁷⁸⁹⁰]{1,2}(?=\W|$)", "", text)

    # Lines that are purely date/unit/note/footnote should yield empty (idempotent with extraction filter)
    if is_date_only_line(text) or is_non_indicator_line(text):
        return ""

    # Strip dates and units so "Prets garantis au 30 avril 2025" = "Prets garantis (M$)" = "Prets garantis"
    raw_for_log = text
    dates_stripped = strip_dates_from_indicator_label(text)
    if not dates_stripped or not dates_stripped.strip():
        return ""
    units_stripped = strip_units_currency_from_indicator_label(dates_stripped)
    if _word_char_count(units_stripped) >= 2:
        text = units_stripped
    elif _word_char_count(dates_stripped) >= 2:
        text = dates_stripped
    else:
        text = dates_stripped
    if logger.isEnabledFor(logging.DEBUG) and raw_for_log != text:
        logger.debug("indicator raw -> cleaned: %r -> %r", raw_for_log[:60], text[:60])

    # Normalize rate formatting noise such as "4,800 %" vs "4,80 %".
    text = re.sub(
        r"\b(\d{1,3})[.,](\d{1,4})(?=\s*%)",
        lambda m: m.group(1) if not m.group(2).rstrip("0") else f"{m.group(1)},{m.group(2).rstrip('0')}",
        text,
    )

    # Strip leading unit-of-measure prefix so "En millions de dollars - X" and "X" share the same key
    for _re in (_UNIT_HEADER_RE, _UNIT_MILLIERS_ACTIONS_RE):
        m = _re.match(text)
        if m:
            rest = text[m.end() :].strip()
            if rest:
                text = rest
            break

    # Canonicalize d'impôt/d'impôts to d'impot before variants (so normalize_label sees d'impot).
    text = re.sub(r"d'imp[o\u00f4]t(s)?", "d'impot", text, flags=re.IGNORECASE)

    # Variantes frequentes (tirets, separateurs, CET-1, Tier-1)
    text = normalize_indicator_variants(text)

    # Supprimer les accents (NFD + ASCII)
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    # Canonicalize d'impot/d'impots so plural does not create false add/remove.
    # Apostrophe may already be space from normalize_indicator_variants, so cover both.
    text = re.sub(r"\bd'impots\b", "d'impot", text, flags=re.IGNORECASE)
    text = re.sub(r"\bd\s+impots\b", "d impot", text, flags=re.IGNORECASE)

    # Canonicalize common wording drift found in debt tranche labels.
    text = re.sub(r"\b(?:avec|aux)\s+termes?\b", " terme ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpremier(?:e)?\s+tranche\b", "tranche 1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdeuxiem(?:e)?\s+tranche\b", "tranche 2", text, flags=re.IGNORECASE)
    text = re.sub(r"\btroisiem(?:e)?\s+tranche\b", "tranche 3", text, flags=re.IGNORECASE)
    text = re.sub(r"\bquatriem(?:e)?\s+tranche\b", "tranche 4", text, flags=re.IGNORECASE)

    # References de notes: (1), (2), [1], [2], etc.
    text = re.sub(r"\s*[\(\[]\d+[\)\]]\s*", " ", text)

    # Appels de notes lettres avec parentheses: (a), (b), (A), (B)
    text = re.sub(r"\s*\([a-zA-Z]\)\s*", " ", text)
    # Appels de notes lettres: a), b), a,b, etc. (Spec Basel III)
    text = re.sub(r"\s*[a-z]\)\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[a-z],\s*[a-z]\s*", " ", text, flags=re.IGNORECASE)

    # Patterns numeriques sans parenthese ouvrante: 2), 3)
    text = re.sub(r"\s*\d+\)\s*", " ", text)

    # Exposants, asterisques et symboles dagger
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", text)
    text = re.sub(r"[\*†‡]+", "", text)

    # Chiffres isolés en fin (notes ou colonnes), sauf semantique (pilier, tier, serie, etc.)
    if not is_trailing_number_semantic(text):
        text = strip_trailing_note_or_column_value(text)

    # Normaliser la ponctuation
    text = re.sub(r"[:\-–—]", " ", text)

    # Variantes linguistiques: "des taux" <-> "de taux" -> meme cle
    text = re.sub(r"\bdes\s+taux\b", "de taux", text, flags=re.IGNORECASE)

    # Narrow normalization for the pension-asset regulatory line where OCR/layout
    # alternates between "actif net des regimes..." and "actif des regimes...".
    text = re.sub(
        r"\bactif net des regimes? de retraite a prestations? definies?\b",
        "actif des regimes de retraite a prestations definies",
        text,
        flags=re.IGNORECASE,
    )

    # Canonicalize lie/liee/lies/liees (after accent strip: lie, liee, etc.) so
    # "exposition liee au credit" and "exposition lie au credit" share the same key.
    text = re.sub(r"\b(lie|liee|lies|liees)\b", "lie", text, flags=re.IGNORECASE)
    # Canonicalize relatif/relative/relatifs/relatives -> relatif
    text = re.sub(
        r"\b(relatif|relative|relatifs|relatives)\b",
        "relatif",
        text,
        flags=re.IGNORECASE,
    )
    # Canonicalize residuel/residuels -> residuel
    text = re.sub(r"\bresiduels\b", "residuel", text, flags=re.IGNORECASE)

    # Espaces et minuscules
    text = re.sub(r"\s+", " ", text).strip().lower()

    # Singularize: "actions" -> "action", "actifs" -> "actif" (neutralizes plural/singular noise)
    text = singularize_words(text)

    # Strip generic "total des/du/de/d'" prefix so "total des fonds propre reglementaire"
    # and "fonds propre reglementaire" produce the same canonical key.
    # Only strip when at least 2 words remain to avoid collapsing meaningful content.
    _total_prefix_match = re.match(r"^total\s+(?:des?\s+|du\s+|d[' ]\s*)", text, flags=re.IGNORECASE)
    if _total_prefix_match:
        rest = text[_total_prefix_match.end() :].strip()
        if len(rest.split()) >= 2:
            text = rest

    return text

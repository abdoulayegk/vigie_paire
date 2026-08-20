"""Partition sémantique stricte des paragraphes narratifs complexes."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vigie.analyse_texte.openai_client import (
    _call_structured_completion_with_correction,
    _embed_texts,
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])")
_NUMBER_RE = re.compile(
    r"(?<!\w)(?:\(?[-+]?\d[\d\s.,]*(?:\s*(?:%|\$|€|£|M\$|G\$))?\)?)(?!\w)",
    flags=re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_CONTINUATION_RE = re.compile(
    r"^(?:Cette|Cet|Ces|Ce|Elle|Il|Ils|Elles|Selon cette|Dans cette|Ainsi|En conséquence)\b",
    flags=re.IGNORECASE,
)
_IDEA_SHIFT_RE = re.compile(
    r"^(?:Toutefois|Par ailleurs|En revanche|Néanmoins|Cependant|Une autre|Une nouvelle|"
    r"Le cadre réglementaire|La gouvernance|La politique)\b",
    flags=re.IGNORECASE,
)
_DEFINITION_OPENER_RE = re.compile(
    r"\b(?:se définit comme(?:\s+étant)?|s['’]entend de|est défini[e]? comme(?:\s+étant)?)\b",
    flags=re.IGNORECASE,
)

_MIN_COMPLEX_SENTENCES = 4
_MIN_COMPLEX_WORDS = 150
_HARD_MAX_WORDS = 240
_SPLIT_SCORE = 0.72
_JOIN_SCORE = 0.84


class SemanticChunkingError(RuntimeError):
    """Erreur bloquante du découpage sémantique, sans résultat de repli."""


class SemanticSentenceGroup(BaseModel):
    """Intervalle inclusif et contigu de phrases, indexé à partir de 1."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=1)
    end: int = Field(ge=1)


class SemanticPartitionResponse(BaseModel):
    """Partition proposée par le LLM pour un paragraphe ambigu."""

    model_config = ConfigDict(extra="forbid")

    groups: list[SemanticSentenceGroup] = Field(min_length=1)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ']+\b", str(text or "")))


def _split_sentences(text: str) -> list[str]:
    paragraph = str(text or "").strip()
    if not paragraph:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(paragraph) if part.strip()]


def _requires_semantic_partition(text: str) -> bool:
    """Indique si un paragraphe justifie embeddings et analyse de frontières."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return False
    if _DEFINITION_OPENER_RE.search(text):
        return True
    return len(sentences) >= _MIN_COMPLEX_SENTENCES or _word_count(text) >= _MIN_COMPLEX_WORDS


def _normalize_sentence_for_similarity(text: str) -> str:
    """Neutralise le bruit numérique uniquement pour calculer les similarités.

    Aucun retrait de marqueur de page ici : le texte reçu est déjà strippé par
    ``_extract_section_text_from_markdown``. Le motif local qui traînait
    effaçait au passage une référence française du type « [p.45] ».
    """
    value = _NUMBER_RE.sub(" <nombre> ", str(text or ""))
    value = _WHITESPACE_RE.sub(" ", value).strip().lower()
    return value


def _cosine(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b) or not vector_a:
        raise SemanticChunkingError("Embeddings invalides: dimensions absentes ou incompatibles.")
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise SemanticChunkingError("Embeddings invalides: vecteur nul.")
    return sum(a * b for a, b in zip(vector_a, vector_b, strict=True)) / (norm_a * norm_b)


def _lexical_adjacent_scores(normalized_sentences: list[str]) -> list[float]:
    if len(normalized_sentences) < 2:
        return []
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(normalized_sentences)
    except ValueError:
        return [0.0] * (len(normalized_sentences) - 1)
    return [
        float(cosine_similarity(matrix[index], matrix[index + 1])[0, 0])
        for index in range(len(normalized_sentences) - 1)
    ]


def _continuity_scores(
    sentences: list[str],
    normalized_sentences: list[str],
    embeddings: list[list[float]],
) -> list[float]:
    lexical_scores = _lexical_adjacent_scores(normalized_sentences)
    scores: list[float] = []
    for index, lexical_score in enumerate(lexical_scores):
        semantic_score = max(0.0, min(1.0, _cosine(embeddings[index], embeddings[index + 1])))
        score = 0.70 * semantic_score + 0.30 * lexical_score
        next_sentence = sentences[index + 1]
        if _CONTINUATION_RE.match(next_sentence):
            score += 0.08
        if _IDEA_SHIFT_RE.match(next_sentence):
            score -= 0.10
        if _DEFINITION_OPENER_RE.search(sentences[index]) or _DEFINITION_OPENER_RE.search(next_sentence):
            score -= 0.20
        scores.append(max(0.0, min(1.0, score)))
    return scores


def _validate_groups(groups: list[SemanticSentenceGroup], sentence_count: int) -> list[tuple[int, int]]:
    expected_start = 1
    validated: list[tuple[int, int]] = []
    for group in groups:
        if group.start != expected_start or group.end < group.start or group.end > sentence_count:
            raise SemanticChunkingError(
                "Partition LLM invalide: les groupes doivent couvrir toutes les phrases, dans l'ordre, sans trou ni chevauchement."
            )
        validated.append((group.start - 1, group.end))
        expected_start = group.end + 1
    if expected_start != sentence_count + 1:
        raise SemanticChunkingError("Partition LLM invalide: la dernière phrase n'est pas couverte.")
    return validated


def _ranges_are_overfragmented(
    ranges: list[tuple[int, int]],
    sentences: list[str],
) -> bool:
    """Détecte une partition qui revient pratiquement à un chunk par phrase."""
    if len(sentences) < 4 or len(ranges) < 4:
        return False
    range_count = len(ranges)
    singleton_count = sum(end - start == 1 for start, end in ranges)
    short_count = sum(_word_count(" ".join(sentences[start:end])) < 50 for start, end in ranges)
    return (
        range_count / len(sentences) >= 0.75
        and singleton_count / range_count >= 0.65
        and short_count / range_count >= 0.65
    )


def _partition_with_llm(
    *,
    client: Any,
    model: str,
    sentences: list[str],
    scores: list[float],
) -> list[tuple[int, int]]:
    """Demande une partition sémantique contiguë et valide strictement sa couverture.

    Une réponse surfragmentée déclenche une seconde tentative guidée. Toute
    réponse invalide ou tout échec du modèle remonte comme erreur de qualité,
    car cette étape ne doit pas masquer le problème par un découpage arbitraire.
    """
    numbered = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, start=1))
    boundary_scores = ", ".join(f"{index + 1}|{index + 2}={score:.3f}" for index, score in enumerate(scores))
    messages = [
        {
            "role": "system",
            "content": (
                "Tu partitionnes un paragraphe financier en unités d'idée autonomes pour comparer deux rapports. "
                "Chaque groupe doit être un intervalle contigu. Couvre chaque phrase exactement une fois, dans l'ordre. "
                "Sépare un changement réel de sujet, de méthode, de règle, de politique ou de gouvernance. "
                "Une phrase qui définit un concept prudentiel (se définit comme, s'entend de, est défini comme) "
                "forme son propre groupe, même si la phrase suivante le reprend avec Il/Cette. "
                "Regroupe les phrases qui décrivent les variantes, paramètres, conditions ou conséquences d'un même cadre. "
                "Ne crée pas un groupe par phrase simplement parce que chaque phrase est grammaticalement complète. "
                "Quand plusieurs phrases développent la même idée, vise généralement 80 à 180 mots par groupe. "
                "Ne crée jamais une frontière à cause d'une année, d'une date, d'un montant, d'un pourcentage, "
                "d'une acquisition ou d'une émission d'actions. Une unité courte mais complète est permise. "
                f"Évite les groupes de plus de {_HARD_MAX_WORDS} mots, sauf phrase indivisible. "
                "Exemple de logique attendue : une phrase autonome sur le calcul général de l'actif pondéré peut rester seule; "
                "une phrase autonome sur une réforme réglementaire peut rester seule; les phrases présentant l'approche NI, "
                "l'approche NI fondation et l'approche NI avancée doivent former un même groupe; les phrases expliquant leurs "
                "paramètres, limites plancher et l'approche standardisée doivent former le groupe suivant."
            ),
        },
        {
            "role": "user",
            "content": (
                "Retourne uniquement les intervalles start/end conformes au schéma.\n"
                f"Scores locaux de continuité entre phrases: {boundary_scores}\n\n{numbered}"
            ),
        },
    ]

    def request_partition(request_messages: list[dict[str, str]]) -> SemanticPartitionResponse:
        """Appelle le modèle et exige une réponse structurée corrigible une fois."""
        return _call_structured_completion_with_correction(
            client,
            model=model,
            messages=request_messages,
            response_format=SemanticPartitionResponse,
            max_retries=1,
            validation_retry_message=(
                "La partition doit couvrir toutes les phrases exactement une fois, dans l'ordre, "
                "avec des intervalles contigus sans trou ni chevauchement."
            ),
        )

    try:
        response = request_partition(messages)
    except Exception as exc:
        raise SemanticChunkingError(f"Échec du partitionnement LLM sans fallback: {exc}") from exc
    ranges = _validate_groups(response.groups, len(sentences))
    if not _ranges_are_overfragmented(ranges, sentences):
        return ranges

    correction_messages = [
        *messages,
        {"role": "assistant", "content": response.model_dump_json()},
        {
            "role": "user",
            "content": (
                "Cette partition est trop fragmentée et revient presque à produire un chunk par phrase. "
                "Corrige-la en regroupant les phrases qui développent la même idée réglementaire. "
                "Vise 80 à 180 mots pour une idée développée. Conserve une phrase courte seule uniquement "
                "si elle constitue une divulgation réellement indépendante des phrases voisines."
            ),
        },
    ]
    try:
        corrected_response = request_partition(correction_messages)
    except Exception as exc:
        raise SemanticChunkingError(
            f"Échec de la correction d'une partition LLM sur-fragmentée sans fallback: {exc}"
        ) from exc
    corrected_ranges = _validate_groups(corrected_response.groups, len(sentences))
    if _ranges_are_overfragmented(corrected_ranges, sentences):
        raise SemanticChunkingError(
            "Partition LLM toujours sur-fragmentée après correction; aucun fallback n'est autorisé."
        )
    return corrected_ranges


def _split_oversized_groups(
    ranges: list[tuple[int, int]],
    sentences: list[str],
    scores: list[float],
) -> list[tuple[int, int]]:
    """Applique le plafond dur à la frontière de continuité la plus faible."""
    pending = list(ranges)
    result: list[tuple[int, int]] = []
    while pending:
        start, end = pending.pop(0)
        if end - start <= 1 or _word_count(" ".join(sentences[start:end])) <= _HARD_MAX_WORDS:
            result.append((start, end))
            continue
        split_at = min(range(start, end - 1), key=lambda index: scores[index]) + 1
        pending[0:0] = [(start, split_at), (split_at, end)]
    return result


def _deterministic_ranges(sentences: list[str], scores: list[float]) -> list[tuple[int, int]]:
    starts = [0]
    for index, score in enumerate(scores):
        force_split = bool(_DEFINITION_OPENER_RE.search(sentences[index]))
        if score <= _SPLIT_SCORE or force_split:
            starts.append(index + 1)
    starts.append(len(sentences))
    unique_starts = sorted(dict.fromkeys(starts))
    return [(unique_starts[index], unique_starts[index + 1]) for index in range(len(unique_starts) - 1)]


def _semantic_partition_paragraphs(
    paragraphs: list[str],
    *,
    client: Any,
    embedding_model: str = "text-embedding-3-small",
    semantic_model: str = "gpt-5.4",
) -> list[list[str]]:
    """Partitionne tous les paragraphes complexes avec un seul lot d'embeddings.

    Une indisponibilité des embeddings ou du LLM lève toujours
    ``SemanticChunkingError``. Aucun ancien découpage n'est utilisé en repli.
    """
    sentence_groups = [_split_sentences(paragraph) for paragraph in paragraphs]
    normalized_groups = [
        [_normalize_sentence_for_similarity(sentence) for sentence in sentences] for sentences in sentence_groups
    ]
    unique_texts = list(dict.fromkeys(text for group in normalized_groups for text in group))
    try:
        unique_embeddings = _embed_texts(client, unique_texts, model=embedding_model)
    except Exception as exc:
        raise SemanticChunkingError(f"Échec des embeddings sans fallback: {exc}") from exc
    if len(unique_embeddings) != len(unique_texts):
        raise SemanticChunkingError(
            "Embeddings invalides: le nombre de vecteurs ne correspond pas au nombre de phrases uniques."
        )
    embedding_by_text = dict(zip(unique_texts, unique_embeddings, strict=True))

    partitions: list[list[str]] = []
    for sentences, normalized_sentences in zip(sentence_groups, normalized_groups, strict=True):
        embeddings = [embedding_by_text[text] for text in normalized_sentences]
        scores = _continuity_scores(sentences, normalized_sentences, embeddings)
        deterministic_ranges = _deterministic_ranges(sentences, scores)
        ambiguous = any(_SPLIT_SCORE < score < _JOIN_SCORE for score in scores)
        ambiguous = ambiguous or _ranges_are_overfragmented(deterministic_ranges, sentences)
        if ambiguous:
            ranges = _partition_with_llm(
                client=client,
                model=semantic_model,
                sentences=sentences,
                scores=scores,
            )
        else:
            ranges = deterministic_ranges
        ranges = _split_oversized_groups(ranges, sentences, scores)
        partitions.append([" ".join(sentences[start:end]).strip() for start, end in ranges])
    return partitions

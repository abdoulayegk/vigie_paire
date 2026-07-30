"""Mesure le rappel du pipeline de vigie face aux tableaux de vigie manuels.

Entree:
  - Reference: evaluation/reference/vigie_manuelle_2025_t4_vs_2024_t4.json
    (items manuels par banque: section, page PDF, sous-section, description du changement)
  - Sorties pipeline: outputs/resultats/<banque>/<pairing>/text_comparison.json
    et outputs/resultats/<banque>/<pairing>/comparison.json

Sortie:
  - Rappel par banque et par canal (texte retenu / texte filtre / tableaux)
  - Liste des items manuels non retrouves, avec les 3 meilleurs candidats pour verification

Le rapprochement est lexical (recouvrement pondere par IDF entre le vocabulaire
distinctif de l'item manuel et le texte du changement detecte). C'est une
approximation: le rappel affiche doit etre lu avec la liste des candidats, pas
comme une verite absolue.

Usage:
  python scripts/eval_vigie_recall.py
  python scripts/eval_vigie_recall.py --threshold 0.5 --out evaluation/rapport_rappel.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = REPO_ROOT / "evaluation/reference/vigie_manuelle_2025_t4_vs_2024_t4.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "outputs/resultats"

# Mots vides francais courants.
STOPWORDS = {
    "afin", "ainsi", "alors", "apres", "au", "aucun", "aujourd", "auparavant", "aussi",
    "autre", "autres", "aux", "avait", "avant", "avec", "avoir", "car", "ce", "cela",
    "celle", "celles", "celui", "ces", "cet", "cette", "ceux", "chaque", "comme",
    "compris", "concernant", "dans", "de", "des", "deux", "dit", "donc", "dont", "du",
    "elle", "elles", "en", "encore", "entre", "est", "et", "etait", "etaient", "etant",
    "ete", "etre", "eu", "faire", "fait", "fin", "font", "hui", "il", "ils", "issu",
    "jusqu", "la", "le", "les", "leur", "leurs", "lors", "lorsque", "lui", "mais",
    "matiere", "meme", "memes", "mes", "moins", "mon", "ne", "ni", "non", "nos",
    "notamment", "notre", "nous", "on", "ont", "ou", "par", "parmi", "pas", "peut",
    "peuvent", "plus", "plusieurs", "pour", "pourrait", "pourraient", "pres", "puis",
    "qu", "quand", "que", "quel", "quelle", "qui", "quoi", "sa", "sans", "se", "selon",
    "sera", "seront", "ses", "si", "son", "sont", "sous", "soit", "sur", "ta", "te",
    "tous", "tout", "toute", "toutes", "tres", "trois", "un", "une", "vers", "vos",
    "votre", "vous", "y", "etc", "titre", "egard", "lieu", "cours", "regard",
}

# Vocabulaire meta decrivant l'acte de changement: present cote analyste,
# absent du texte source. Le garder ferait matcher n'importe quoi.
CHANGE_VERBS = {
    "ajout", "ajoute", "ajoutee", "ajoutes", "ajoutees", "ajouter", "ajoutant",
    "suppression", "supprime", "supprimee", "supprimes", "supprimer",
    "retrait", "retraits", "retire", "retiree", "retires", "retirer", "retirant",
    "modification", "modifications", "modifie", "modifiee", "modifies", "modifiees",
    "mise", "jour", "maj", "changement", "changements", "change", "changer",
    "refonte", "majeure", "majeur", "allegement", "simplification", "restructuration",
    "nouveau", "nouvelle", "nouveaux", "nouvelles",
    "texte", "textes", "paragraphe", "paragraphes", "phrase", "phrases", "mot", "mots",
    "section", "sections", "partie", "sortie", "ligne", "lignes", "poste", "postes",
    "mention", "mentions", "mentionne", "mentionner", "mentionnant",
    "precision", "precisions", "preciser", "precise", "precisant",
    "tableau", "tableaux", "graphique", "graphiques", "note", "notes", "bas", "page",
    "pages", "pdf", "libelle", "libelles", "passage", "passant", "devient", "deviennent",
    "remplace", "remplacee", "remplaces", "remplacer", "auparavant", "anciennement",
    "ancien", "ancienne", "ancienneS", "reels", "reel", "pas", "davantage", "details",
    "detail", "presente", "presentee", "presentes", "presentees", "presentation",
    "publication", "publie", "publiee", "publies", "traitant", "traite", "expliquer",
    "explicatif", "explicative", "explicatifs", "voir", "actualite", "lien", "sujet",
    "portant", "apportant", "apporter", "introduction", "introduit", "introduire",
    "integre", "integration", "deplace", "scinde", "scindee", "fusion", "garder",
    "exemple", "exemples", "notion", "notions", "terme", "termes", "titre", "sorte",
}

DROP = STOPWORDS | CHANGE_VERBS

# Ancrage minimal sur le titre de sous-section du candidat.
MIN_ANCHOR = 0.5

# Masse IDF plancher pour l'ancrage: un titre d'une seule sous-section generique
# ("Risque", "__intro__") ne doit pas pouvoir atteindre un ancrage de 1,0 juste
# parce que son unique token apparait quelque part dans la description manuelle.
MIN_HEADING_MASS = 4.0

# Section manuelle -> cle de section du pipeline.
SECTION_MAP = {
    "GESTION DU CAPITAL": "gestion_capital",
    "GESTION DES RISQUES": "gestion_risques",
}


def strip_accents(text: str) -> str:
    """Retire les diacritiques pour comparer des libelles ecrits differemment.

    Args:
        text: Chaine source, potentiellement accentuee.

    Returns:
        La chaine sans diacritiques combinants.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def light_stem(token: str) -> str:
    """Normalise les variantes de nombre pour rapprocher "risques" et "risque".

    Args:
        token: Token deja minuscule et desaccentue.

    Returns:
        Le token sans marque de pluriel evidente.
    """
    if len(token) >= 5 and token.endswith(("s", "x")) and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, *, drop_meta: bool) -> set[str]:
    """Extrait les tokens porteurs de sens d'un fragment de texte francais.

    Args:
        text: Texte a segmenter.
        drop_meta: Si vrai, retire aussi le vocabulaire decrivant l'acte de
            changement (utile pour les descriptions d'analyste, nuisible pour
            le texte source qui ne le contient pas).

    Returns:
        Ensemble de tokens normalises (sans accents, minuscules, sans pluriel).
    """
    normalized = strip_accents(text).lower()
    raw = re.findall(r"[a-z0-9]+", normalized)
    banned = DROP if drop_meta else STOPWORDS
    tokens: set[str] = set()
    for token in raw:
        if token in banned:
            continue
        if token.isdigit():
            # Les annees et numeros de ligne directrice sont des ancres fortes.
            if len(token) >= 2:
                tokens.add(token)
            continue
        if len(token) >= 3:
            stemmed = light_stem(token)
            if stemmed not in banned:
                tokens.add(stemmed)
    return tokens


def load_text_candidates(path: Path) -> list[dict[str, Any]]:
    """Charge les changements textuels detectes, retenus et filtres.

    Args:
        path: Chemin vers text_comparison.json.

    Returns:
        Liste de candidats avec channel ("texte_retenu" ou "texte_filtre"),
        section_key, sous-section, pages et texte concatene.
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for section in payload.get("section_comparisons") or []:
        section_key = str(section.get("section_key") or "")
        retained_ids = {
            str(b.get("change_id"))
            for b in section.get("block_comparisons") or []
            if isinstance(b, dict)
        }
        blocks = section.get("all_block_comparisons") or section.get("block_comparisons") or []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            change_id = str(block.get("change_id") or "")
            parts = [
                str(block.get("subsection_heading") or ""),
                str(block.get("source_text_t1") or ""),
                str(block.get("source_text_t2") or ""),
                str(block.get("change_summary") or ""),
            ]
            pages = [
                *(block.get("pages_t1") or []),
                *(block.get("pages_t2") or []),
            ]
            candidates.append(
                {
                    "channel": "texte_retenu" if change_id in retained_ids else "texte_filtre",
                    "id": change_id,
                    "section_key": section_key,
                    "subsection": str(block.get("subsection_heading") or ""),
                    "diff_type": str(block.get("diff_type") or ""),
                    "pages": sorted({int(p) for p in pages if isinstance(p, int)}),
                    "text": " ".join(p for p in parts if p),
                    "text_t1": str(block.get("source_text_t1") or ""),
                    "text_t2": str(block.get("source_text_t2") or ""),
                    "summary": str(block.get("change_summary") or ""),
                }
            )
    return candidates


def _technical_diff_text(diff: dict[str, Any]) -> str:
    """Aplatit un technical_diff de table en texte recherchable.

    Args:
        diff: Bloc technical_diff d'un pair_comparison.

    Returns:
        Concatenation des libelles d'indicateurs et de notes ayant bouge.
    """
    chunks: list[str] = []
    for key in (
        "indicators_added",
        "indicators_removed",
        "indicators_renamed",
        "footnotes_added",
        "footnotes_removed",
        "footnotes_renamed",
    ):
        for entry in diff.get(key) or []:
            if isinstance(entry, str):
                chunks.append(entry)
            elif isinstance(entry, dict):
                chunks.extend(str(v) for v in entry.values() if isinstance(v, str))
    table_level = diff.get("table_level_change")
    if isinstance(table_level, str):
        chunks.append(table_level)
    elif isinstance(table_level, dict):
        chunks.extend(str(v) for v in table_level.values() if isinstance(v, str))
    return " ".join(chunks)


def load_table_candidates(path: Path) -> list[dict[str, Any]]:
    """Charge les changements du pipeline tables (indicateurs, notes, tables entrantes/sortantes).

    Args:
        path: Chemin vers comparison.json.

    Returns:
        Liste de candidats de canal "tableaux".
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for pair in payload.get("pair_comparisons") or []:
        if not isinstance(pair, dict):
            continue
        prev_table = pair.get("previous_table") or {}
        curr_table = pair.get("current_table") or {}
        assessment = pair.get("analyst_assessment") or {}
        analysis = pair.get("genai_analysis") or {}
        parts = [
            str(prev_table.get("title") or ""),
            str(curr_table.get("title") or ""),
            str(prev_table.get("table_summary") or ""),
            str(curr_table.get("table_summary") or ""),
            _technical_diff_text(pair.get("technical_diff") or {}),
            str(assessment.get("analyst_summary") or ""),
            str(analysis.get("resume_metier") or ""),
        ]
        pages = [
            p
            for p in (prev_table.get("page"), curr_table.get("page"))
            if isinstance(p, int)
        ]
        section = str(curr_table.get("section") or prev_table.get("section") or "")
        candidates.append(
            {
                "channel": "tableaux",
                "id": f"{prev_table.get('table_id')}->{curr_table.get('table_id')}",
                "section_key": section,
                "subsection": str(curr_table.get("title") or prev_table.get("title") or ""),
                "diff_type": "modified",
                "pages": sorted(set(pages)),
                "text": " ".join(p for p in parts if p),
            }
        )
    matching = payload.get("matching") or {}
    for key, label in (("tables_added", "added"), ("tables_removed", "removed")):
        for table in matching.get(key) or []:
            if not isinstance(table, dict):
                continue
            page = table.get("page")
            candidates.append(
                {
                    "channel": "tableaux",
                    "id": f"{label}:{table.get('table_id')}",
                    "section_key": str(table.get("section") or ""),
                    "subsection": str(table.get("title") or ""),
                    "diff_type": label,
                    "pages": [page] if isinstance(page, int) else [],
                    "text": " ".join(
                        p
                        for p in (
                            str(table.get("title") or ""),
                            str(table.get("table_summary") or ""),
                        )
                        if p
                    ),
                }
            )
    return candidates


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retire les candidats redondants produits a plusieurs granularites.

    Le pipeline emet le meme passage a l'echelle du chunk et du parent; garder
    les doublons gaspille le budget de candidats soumis au juge.

    Args:
        candidates: Candidats bruts.

    Returns:
        Candidats sans doublon, premier vu conserve.
    """
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for cand in candidates:
        key = (
            cand["channel"],
            cand["subsection"],
            cand["diff_type"],
            (cand.get("text") or "")[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def select_top(
    ranked: list[tuple[float, float, float, dict[str, Any]]],
    k: int,
    *,
    per_heading: int = 2,
) -> list[tuple[float, float, float, dict[str, Any]]]:
    """Selectionne les K meilleurs candidats en diversifiant les sous-sections.

    Sans plafond par sous-section, un titre de section parent qui colle au
    libelle de la colonne "sous-section" du tableau manuel monopolise tout le
    budget et evince la sous-section fille reellement visee.

    Args:
        ranked: Candidats classes par score decroissant.
        k: Nombre de candidats a retenir.
        per_heading: Nombre maximal de candidats par titre de sous-section.

    Returns:
        Sous-liste diversifiee, ordre de score conserve.
    """
    counts: Counter[str] = Counter()
    selected: list[tuple[float, float, float, dict[str, Any]]] = []
    for row in ranked:
        heading = row[3]["subsection"]
        if counts[heading] >= per_heading:
            continue
        counts[heading] += 1
        selected.append(row)
        if len(selected) >= k:
            break
    return selected


def build_idf(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Calcule l'IDF des tokens sur le corpus des changements detectes.

    Args:
        candidates: Candidats du pipeline pour une banque.

    Returns:
        Dictionnaire token -> poids IDF.
    """
    document_frequency: Counter[str] = Counter()
    for cand in candidates:
        document_frequency.update(cand["tokens"])
    total = max(len(candidates), 1)
    return {
        token: math.log(1.0 + total / (1.0 + freq))
        for token, freq in document_frequency.items()
    }


def score_item(
    item_tokens: set[str],
    candidates: list[dict[str, Any]],
    idf: dict[str, float],
    *,
    section_key: str,
) -> list[tuple[float, float, float, dict[str, Any]]]:
    """Classe les candidats du pipeline face au vocabulaire d'un item manuel.

    Deux signaux sont combines:
      - ancrage: part du vocabulaire du titre de sous-section du candidat que
        l'analyste a effectivement ecrite. C'est le signal discriminant, car un
        analyste nomme presque toujours la sous-section qu'il vise.
      - contenu: part du poids IDF du vocabulaire de l'item couverte par le
        texte du candidat.

    Args:
        item_tokens: Tokens distinctifs de l'item manuel.
        candidates: Candidats du pipeline pour la banque.
        idf: Poids IDF du corpus de la banque.
        section_key: Section attendue ("gestion_capital" / "gestion_risques").

    Returns:
        Liste (score, ancrage, contenu, candidat) triee par score decroissant.
    """
    default_weight = math.log(2.0)
    total_weight = sum(idf.get(t, default_weight) for t in item_tokens)
    if total_weight <= 0:
        return []
    scored: list[tuple[float, float, float, dict[str, Any]]] = []
    for cand in candidates:
        shared = item_tokens & cand["tokens"]
        if not shared:
            continue
        content = sum(idf.get(t, default_weight) for t in shared) / total_weight
        heading_tokens = cand["heading_tokens"]
        heading_mass = sum(idf.get(t, default_weight) for t in heading_tokens)
        anchor = sum(
            idf.get(t, default_weight) for t in heading_tokens & item_tokens
        ) / max(heading_mass, MIN_HEADING_MASS)
        score = 0.65 * anchor + 0.35 * content
        if section_key and cand["section_key"] and cand["section_key"] != section_key:
            score *= 0.8
        scored.append((score, anchor, content, cand))
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored


def evaluate_bank(
    bank_code: str,
    bank_ref: dict[str, Any],
    results_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    """Evalue une banque: rappel global, rappel par canal, items non retrouves.

    Args:
        bank_code: Code banque en minuscules (bmo, cibc, td, bns, rbc).
        bank_ref: Bloc de reference de la banque (items manuels).
        results_dir: Dossier de la paire, ex. outputs/resultats/bmo/2025_t4_vs_2024_t4.
        threshold: Score minimal pour considerer un item comme retrouve.

    Returns:
        Dictionnaire de resultats serialisable.
    """
    candidates = load_text_candidates(results_dir / "text_comparison.json")
    candidates += load_table_candidates(results_dir / "comparison.json")
    candidates = dedupe_candidates(candidates)
    for cand in candidates:
        cand["tokens"] = tokenize(cand["text"], drop_meta=False)
        cand["heading_tokens"] = tokenize(cand["subsection"], drop_meta=False)
    idf = build_idf(candidates)

    scorable: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    for item in bank_ref.get("items") or []:
        if item.get("truncated") or item.get("declares_no_change"):
            continue
        item_tokens = tokenize(
            f"{item.get('subsection', '')} {item.get('change', '')}", drop_meta=True
        )
        if not item_tokens:
            unscorable.append(item)
            continue
        section_key = SECTION_MAP.get(str(item.get("section") or ""), "")
        ranked = score_item(item_tokens, candidates, idf, section_key=section_key)
        top = select_top(ranked, 5)
        best_score, best_anchor, _, best_cand = (
            top[0] if top else (0.0, 0.0, 0.0, None)
        )
        # L'ancrage minimal evite les rapprochements fondes sur du vocabulaire
        # bancaire generique partage par des sous-sections sans rapport.
        is_matched = best_score >= threshold and best_anchor >= MIN_ANCHOR
        scorable.append(
            {
                "item": item,
                "n_tokens": len(item_tokens),
                "best_score": round(best_score, 3),
                "best_anchor": round(best_anchor, 3),
                "matched": is_matched,
                "best_channel": best_cand["channel"] if best_cand else None,
                "top": [
                    {
                        "score": round(score, 3),
                        "anchor": round(anchor, 3),
                        "content": round(content, 3),
                        "channel": cand["channel"],
                        "id": cand["id"],
                        "subsection": cand["subsection"],
                        "diff_type": cand["diff_type"],
                        "pages": cand["pages"],
                        "excerpt": cand["text"][:280],
                    }
                    for score, anchor, content, cand in top
                ],
            }
        )

    matched = [r for r in scorable if r["matched"]]
    channel_counts = Counter(r["best_channel"] for r in matched)
    n_retained = sum(1 for c in candidates if c["channel"] == "texte_retenu")
    n_filtered = sum(1 for c in candidates if c["channel"] == "texte_filtre")
    n_tables = sum(1 for c in candidates if c["channel"] == "tableaux")
    return {
        "bank_code": bank_code,
        "bank_label": bank_ref.get("bank_label"),
        "n_reference_items": len(bank_ref.get("items") or []),
        "n_scored": len(scorable),
        "n_unscorable": len(unscorable),
        "n_matched": len(matched),
        "recall": round(len(matched) / len(scorable), 3) if scorable else 0.0,
        "matched_by_channel": dict(channel_counts),
        "pipeline_volume": {
            "texte_retenu": n_retained,
            "texte_filtre": n_filtered,
            "tableaux": n_tables,
        },
        "results": scorable,
        "unscorable_items": unscorable,
    }


def render_markdown(report: dict[str, Any], threshold: float) -> str:
    """Met en forme le rapport d'evaluation en markdown lisible par un analyste.

    Args:
        report: Rapport produit par la boucle principale.
        threshold: Seuil de rapprochement utilise.

    Returns:
        Contenu markdown.
    """
    lines: list[str] = [
        "# Etage de recherche lexical (candidats)",
        "",
        f"Paire: `{report['pairing']}` — seuil de rapprochement lexical: {threshold}",
        "",
        "> Ce rapport n'est pas la mesure de rappel. Le rapprochement purement lexical",
        "> accepte des candidats qui partagent du vocabulaire bancaire generique sans",
        "> decrire le meme changement, ce qui surestime le rappel d'une dizaine de",
        "> points. Son role est de proposer les candidats a l'adjudication.",
        "> La mesure de reference est `evaluation/rappel_vigie_juge.md`",
        "> (`scripts/eval_vigie_judge.py`).",
        "",
        "| Banque | Items manuels | Evalues | Retrouves | Rappel | via texte retenu | via texte filtre | via tableaux |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in report["banks"]:
        by_channel = bank["matched_by_channel"]
        lines.append(
            "| {label} | {n_ref} | {n_scored} | {n_matched} | {recall:.0%} | {a} | {b} | {c} |".format(
                label=bank["bank_label"],
                n_ref=bank["n_reference_items"],
                n_scored=bank["n_scored"],
                n_matched=bank["n_matched"],
                recall=bank["recall"],
                a=by_channel.get("texte_retenu", 0),
                b=by_channel.get("texte_filtre", 0),
                c=by_channel.get("tableaux", 0),
            )
        )
    lines += ["", "## Volume produit par le pipeline", "",
              "| Banque | Changements texte retenus | Changements texte filtres | Paires de tables |",
              "|---|---:|---:|---:|"]
    for bank in report["banks"]:
        vol = bank["pipeline_volume"]
        lines.append(
            f"| {bank['bank_label']} | {vol['texte_retenu']} | {vol['texte_filtre']} | {vol['tableaux']} |"
        )

    for bank in report["banks"]:
        misses = [r for r in bank["results"] if not r["matched"]]
        lines += ["", f"## {bank['bank_label']} — items manuels non retrouves ({len(misses)})", ""]
        if not misses:
            lines.append("Aucun.")
        for miss in misses:
            item = miss["item"]
            lines += [
                f"### {item['id']}"
                + (f" — p.{item['page_pdf']}" if item.get("page_pdf") else "")
                + (f" — {item['subsection']}" if item.get("subsection") else ""),
                "",
                f"> {item['change']}",
                "",
                f"Meilleur score: {miss['best_score']}",
                "",
            ]
            for cand in miss["top"]:
                lines.append(
                    f"- `{cand['score']}` [{cand['channel']}] {cand['subsection']} "
                    f"(p.{cand['pages']}, {cand['diff_type']}) — {cand['excerpt'][:200]}"
                )
            lines.append("")
        if bank["unscorable_items"]:
            lines += [
                f"Items sans vocabulaire distinctif (non evalues): "
                f"{', '.join(i['id'] for i in bank['unscorable_items'])}",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    """Point d'entree CLI: calcule et ecrit le rapport de rappel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "evaluation/rappel_vigie.json")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "evaluation/rappel_vigie.md")
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    pairing = str(reference.get("pairing"))
    banks: list[dict[str, Any]] = []
    for bank_code, bank_ref in (reference.get("banks") or {}).items():
        results_dir = args.results_root / bank_code / pairing
        if not results_dir.exists():
            print(f"[skip] {bank_code}: {results_dir} absent")
            continue
        banks.append(evaluate_bank(bank_code, bank_ref, results_dir, args.threshold))

    total_scored = sum(b["n_scored"] for b in banks)
    total_matched = sum(b["n_matched"] for b in banks)
    report = {
        "pairing": pairing,
        "threshold": args.threshold,
        "total_scored": total_scored,
        "total_matched": total_matched,
        "overall_recall": round(total_matched / total_scored, 3) if total_scored else 0.0,
        "banks": banks,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report, args.threshold), encoding="utf-8")

    print(f"Rappel global: {total_matched}/{total_scored} = {report['overall_recall']:.0%}")
    for bank in banks:
        print(
            f"  {bank['bank_label']:<14} {bank['n_matched']:>3}/{bank['n_scored']:<3} "
            f"= {bank['recall']:.0%}   canaux={bank['matched_by_channel']}"
        )
    print(f"\nRapport: {args.out_md}")


if __name__ == "__main__":
    main()

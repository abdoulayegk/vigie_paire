"""Adjudique par LLM le rapprochement entre vigie manuelle et vigie automatique.

Le rapprochement lexical de `eval_vigie_recall.py` sert ici de simple etage de
recherche: pour chaque item manuel il propose les K meilleurs candidats, et un
modele tranche lequel (s'il y en a un) decrit le meme changement editorial.

Cela corrige le principal defaut du rappel purement lexical, qui accepte des
candidats partageant du vocabulaire bancaire generique sans decrire le meme
changement.

Usage:
  python scripts/eval_vigie_judge.py
  python scripts/eval_vigie_judge.py --top-k 6 --model gpt-4o --concurrency 8
  python scripts/eval_vigie_judge.py --banks bmo,rbc --limit 10   # essai a blanc
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_vigie_recall import (  # noqa: E402
    DEFAULT_REFERENCE,
    DEFAULT_RESULTS_ROOT,
    REPO_ROOT,
    SECTION_MAP,
    build_idf,
    dedupe_candidates,
    load_table_candidates,
    load_text_candidates,
    score_item,
    select_top,
    tokenize,
)

SYSTEM_PROMPT = """Tu compares deux lectures d'un meme couple de rapports annuels de banque canadienne (exercice 2025 vs exercice 2024).

- L'OBSERVATION est une ligne de vigie redigee a la main par un analyste: elle decrit un changement editorial qu'il a repere entre les deux rapports.
- Les CANDIDATS sont des changements detectes automatiquement: pour chacun tu vois la sous-section, le sens du changement, le texte de l'exercice precedent (AVANT), celui de l'exercice courant (APRES) et un resume genere.

Ta tache: dire si l'un des candidats couvre le MEME changement editorial que l'observation.

Regles:
- Un candidat couvre l'observation s'il porte sur le meme passage ou le meme element du rapport et va dans le meme sens (ajout, retrait, reformulation, changement de libelle). Le vocabulaire n'a pas besoin d'etre identique.
- Ne retiens PAS un candidat qui parle simplement du meme grand theme (le risque de credit, la gouvernance, le climat) sans porter sur le passage vise par l'analyste. C'est l'erreur a eviter en priorite.
- Un candidat qui couvre une partie substantielle de l'observation compte comme couverture, avec confiance "moyenne".
- Si l'observation vise un tableau, une note de bas de tableau ou un libelle de ligne, un candidat de canal "tableaux" portant sur ce tableau compte.
- Si aucun candidat ne convient, reponds match_index = null.

Reponds uniquement en JSON:
{"match_index": <entier du candidat retenu ou null>, "confiance": "haute"|"moyenne"|"faible", "raison": "<une phrase courte en francais>"}"""


def format_candidate(index: int, cand: dict[str, Any], *, excerpt_chars: int) -> str:
    """Met en forme un candidat pour le prompt du juge.

    Args:
        index: Numero affiche du candidat.
        cand: Candidat issu du pipeline.
        excerpt_chars: Longueur maximale de chaque extrait AVANT/APRES.

    Returns:
        Bloc texte decrivant le candidat.
    """
    before = (cand.get("text_t1") or "").strip()
    after = (cand.get("text_t2") or "").strip()
    if not before and not after:
        # Canal tableaux: pas de couple avant/apres, on expose le texte agrege.
        before = ""
        after = (cand.get("text") or "").strip()
    lines = [
        f"[{index}] canal={cand['channel']} | sens={cand['diff_type']} | "
        f"sous-section={cand['subsection'] or '(sans titre)'} | pages={cand['pages']}",
    ]
    if before:
        lines.append(f"    AVANT: {before[:excerpt_chars]}")
    if after:
        lines.append(f"    APRES: {after[:excerpt_chars]}")
    if cand.get("summary"):
        lines.append(f"    RESUME: {str(cand['summary'])[:300]}")
    return "\n".join(lines)


def build_prompt(
    item: dict[str, Any],
    ranked: list[tuple[float, float, float, dict[str, Any]]],
    *,
    excerpt_chars: int,
) -> str:
    """Construit le message utilisateur soumis au juge pour un item manuel.

    Args:
        item: Item de la vigie manuelle.
        ranked: Candidats classes (score, ancrage, contenu, candidat).
        excerpt_chars: Longueur maximale de chaque extrait.

    Returns:
        Prompt utilisateur complet.
    """
    candidates_block = "\n".join(
        format_candidate(i, cand, excerpt_chars=excerpt_chars)
        for i, (_, _, _, cand) in enumerate(ranked)
    )
    return (
        "OBSERVATION DE L'ANALYSTE\n"
        f"Section: {item.get('section')}\n"
        f"Page PDF: {item.get('page_pdf')}\n"
        f"Contexte / sous-section: {item.get('subsection')}\n"
        f"Changement decrit: {item.get('change')}\n\n"
        f"CANDIDATS DETECTES AUTOMATIQUEMENT ({len(ranked)})\n"
        f"{candidates_block}\n"
    )


async def judge_one(
    client: Any,
    model: str,
    item: dict[str, Any],
    ranked: list[tuple[float, float, float, dict[str, Any]]],
    semaphore: asyncio.Semaphore,
    *,
    excerpt_chars: int,
) -> dict[str, Any]:
    """Soumet un item et ses candidats au juge et normalise la reponse.

    Args:
        client: Instance AsyncOpenAI.
        model: Identifiant du modele.
        item: Item de la vigie manuelle.
        ranked: Candidats classes pour cet item.
        semaphore: Limiteur de concurrence.
        excerpt_chars: Longueur maximale de chaque extrait.

    Returns:
        Verdict serialisable pour cet item.
    """
    if not ranked:
        return {
            "item_id": item["id"],
            "item": item,
            "verdict": "aucun_candidat",
            "match_index": None,
            "confiance": "haute",
            "raison": "Aucun candidat lexical propose.",
            "matched_candidate": None,
        }
    prompt = build_prompt(item, ranked, excerpt_chars=excerpt_chars)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001 - on veut tracer l'echec, pas l'avaler
            return {
                "item_id": item["id"],
                "item": item,
                "verdict": "erreur",
                "match_index": None,
                "confiance": None,
                "raison": f"{type(exc).__name__}: {exc}",
                "matched_candidate": None,
            }

    raw_index = payload.get("match_index")
    index = raw_index if isinstance(raw_index, int) and 0 <= raw_index < len(ranked) else None
    matched_candidate = None
    if index is not None:
        score, anchor, content, cand = ranked[index]
        matched_candidate = {
            "channel": cand["channel"],
            "id": cand["id"],
            "subsection": cand["subsection"],
            "diff_type": cand["diff_type"],
            "pages": cand["pages"],
            "lexical_score": round(score, 3),
            "lexical_anchor": round(anchor, 3),
            "lexical_content": round(content, 3),
            "excerpt": (cand.get("text") or "")[:300],
        }
    return {
        "item_id": item["id"],
        "item": item,
        "verdict": "couvert" if index is not None else "non_couvert",
        "match_index": index,
        "confiance": payload.get("confiance"),
        "raison": payload.get("raison"),
        "matched_candidate": matched_candidate,
        "lexical_best_score": round(ranked[0][0], 3),
    }


def prepare_bank(
    bank_code: str,
    bank_ref: dict[str, Any],
    results_dir: Path,
    top_k: int,
) -> tuple[list[tuple[dict[str, Any], list[tuple[float, float, float, dict[str, Any]]]]], dict[str, int]]:
    """Prepare les couples (item, candidats) a soumettre au juge pour une banque.

    Args:
        bank_code: Code banque.
        bank_ref: Bloc de reference de la banque.
        results_dir: Dossier de la paire.
        top_k: Nombre de candidats proposes par item.

    Returns:
        Les couples a juger, et le volume de changements produits par canal.
    """
    candidates = load_text_candidates(results_dir / "text_comparison.json")
    candidates += load_table_candidates(results_dir / "comparison.json")
    candidates = dedupe_candidates(candidates)
    for cand in candidates:
        cand["tokens"] = tokenize(cand["text"], drop_meta=False)
        cand["heading_tokens"] = tokenize(cand["subsection"], drop_meta=False)
    idf = build_idf(candidates)

    tasks: list[tuple[dict[str, Any], list[tuple[float, float, float, dict[str, Any]]]]] = []
    for item in bank_ref.get("items") or []:
        if item.get("truncated") or item.get("declares_no_change"):
            continue
        item_tokens = tokenize(
            f"{item.get('subsection', '')} {item.get('change', '')}", drop_meta=True
        )
        section_key = SECTION_MAP.get(str(item.get("section") or ""), "")
        ranked = score_item(item_tokens, candidates, idf, section_key=section_key)
        tasks.append((item, select_top(ranked, top_k)))

    volume = {
        "texte_retenu": sum(1 for c in candidates if c["channel"] == "texte_retenu"),
        "texte_filtre": sum(1 for c in candidates if c["channel"] == "texte_filtre"),
        "tableaux": sum(1 for c in candidates if c["channel"] == "tableaux"),
    }
    return tasks, volume


def render_markdown(report: dict[str, Any]) -> str:
    """Met en forme le rapport adjuge en markdown.

    Args:
        report: Rapport global produit par main().

    Returns:
        Contenu markdown.
    """
    lines = [
        "# Rappel de la vigie automatique — adjudication LLM",
        "",
        f"Paire: `{report['pairing']}` · juge: `{report['model']}` · "
        f"{report['top_k']} candidats proposes par item",
        "",
        "Lecture: « couvert » = le pipeline a produit un changement qui porte sur le meme",
        "passage et le meme sens que l'observation de l'analyste. « texte filtre » = le",
        "changement a bien ete detecte mais ecarte avant l'export analyste.",
        "",
        "| Banque | Items | Couverts | Rappel | dont haute conf. | via texte retenu | via texte filtre | via tableaux |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bank in report["banks"]:
        ch = bank["covered_by_channel"]
        lines.append(
            "| {label} | {n} | {c} | {r:.0%} | {h} | {a} | {b} | {t} |".format(
                label=bank["bank_label"],
                n=bank["n_items"],
                c=bank["n_covered"],
                r=bank["recall"],
                h=bank["n_covered_high_confidence"],
                a=ch.get("texte_retenu", 0),
                b=ch.get("texte_filtre", 0),
                t=ch.get("tableaux", 0),
            )
        )
    delivered = sum(
        b["n_covered"] - b["covered_by_channel"].get("texte_filtre", 0)
        for b in report["banks"]
    )
    total = report["total_items"]
    lines += [
        "",
        f"**Rappel de detection: {report['total_covered']}/{total} = "
        f"{report['overall_recall']:.0%}**",
        "",
        f"**Rappel effectivement livre a l'analyste: {delivered}/{total} = "
        f"{delivered / total:.0%}** — l'ecart de "
        f"{report['total_covered'] - delivered} items correspond a des changements "
        "detectes puis ecartes par le triage de pertinence "
        "(`genai_triage.is_relevant = false`), donc absents de l'export.",
        "",
        "## Volume produit par le pipeline",
        "",
        "| Banque | Texte retenu | Texte filtre | Paires de tables |",
        "|---|---:|---:|---:|",
    ]
    for bank in report["banks"]:
        vol = bank["pipeline_volume"]
        lines.append(
            f"| {bank['bank_label']} | {vol['texte_retenu']} | "
            f"{vol['texte_filtre']} | {vol['tableaux']} |"
        )

    for bank in report["banks"]:
        covered = [v for v in bank["verdicts"] if v["verdict"] == "couvert"]
        lines += [
            "",
            f"## {bank['bank_label']} — couverts ({len(covered)})",
            "",
            "| Item | Page | Sous-section (analyste) | Changement | Canal | Sous-section detectee | Sens | Conf. |",
            "|---|---:|---|---|---|---|---|---|",
        ]
        for hit in covered:
            item = hit["item"]
            cand = hit["matched_candidate"] or {}
            lines.append(
                "| {id} | {page} | {sub} | {change} | {chan} | {csub} | {sens} | {conf} |".format(
                    id=item["id"],
                    page=item.get("page_pdf") or "",
                    sub=str(item.get("subsection") or "").replace("|", "/")[:45],
                    change=item["change"].replace("|", "/").replace("\n", " ")[:150],
                    chan=cand.get("channel", ""),
                    csub=str(cand.get("subsection", "")).replace("|", "/")[:45],
                    sens=cand.get("diff_type", ""),
                    conf=hit.get("confiance") or "",
                )
            )

    for bank in report["banks"]:
        misses = [v for v in bank["verdicts"] if v["verdict"] != "couvert"]
        lines += ["", f"## {bank['bank_label']} — non couverts ({len(misses)})", ""]
        if not misses:
            lines.append("Aucun.")
        for miss in misses:
            item = miss["item"]
            lines += [
                f"- **{item['id']}** {_locator(item)}{item['change'][:200]}",
                f"  - juge: {miss['raison']}",
            ]
        filtered = [
            v
            for v in bank["verdicts"]
            if v["verdict"] == "couvert"
            and (v.get("matched_candidate") or {}).get("channel") == "texte_filtre"
        ]
        if filtered:
            lines += [
                "",
                f"### {bank['bank_label']} — detectes mais ecartes avant l'export ({len(filtered)})",
                "",
            ]
            for hit in filtered:
                item = hit["item"]
                cand = hit["matched_candidate"]
                lines.append(
                    f"- **{item['id']}** {_locator(item)}{item['change'][:150]}\n"
                    f"  - candidat ecarte: {cand['subsection']} ({cand['diff_type']}, "
                    f"p.{cand['pages']})"
                )
    return "\n".join(lines)


def _locator(item: dict[str, Any]) -> str:
    """Prefixe de localisation d'un item, tolerant a l'absence de page ou de sous-section.

    Args:
        item: Item de la vigie manuelle.

    Returns:
        Chaine du type "p.151 · Cadre — ", vide si aucun reperage n'est fourni.
    """
    parts = []
    if item.get("page_pdf"):
        parts.append(f"p.{item['page_pdf']}")
    if item.get("subsection"):
        parts.append(str(item["subsection"]))
    return " · ".join(parts) + " — " if parts else ""


def write_csv(report: dict[str, Any], path: Path) -> None:
    """Ecrit une ligne par item de reference avec son verdict, pour tri dans un tableur.

    Args:
        report: Rapport global adjuge.
        path: Chemin du CSV a ecrire.
    """
    columns = [
        "banque",
        "item_id",
        "page_pdf",
        "section",
        "sous_section_analyste",
        "changement_analyste",
        "verdict",
        "canal",
        "confiance",
        "sous_section_detectee",
        "sens_detecte",
        "pages_detectees",
        "raison_juge",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(columns)
        for bank in report["banks"]:
            for verdict in bank["verdicts"]:
                item = verdict["item"]
                cand = verdict.get("matched_candidate") or {}
                writer.writerow(
                    [
                        bank["bank_label"],
                        item["id"],
                        item.get("page_pdf") or "",
                        item.get("section") or "",
                        item.get("subsection") or "",
                        item["change"],
                        verdict["verdict"],
                        cand.get("channel", ""),
                        verdict.get("confiance") or "",
                        cand.get("subsection", ""),
                        cand.get("diff_type", ""),
                        " ".join(str(p) for p in cand.get("pages") or []),
                        verdict.get("raison") or "",
                    ]
                )


async def run() -> None:
    """Orchestration: prepare les items, appelle le juge, ecrit les rapports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--excerpt-chars", type=int, default=700)
    parser.add_argument("--banks", default="", help="Codes separes par des virgules")
    parser.add_argument("--limit", type=int, default=0, help="Items max par banque (essai)")
    parser.add_argument(
        "--out-json", type=Path, default=REPO_ROOT / "evaluation/rappel_vigie_juge.json"
    )
    parser.add_argument(
        "--out-md", type=Path, default=REPO_ROOT / "evaluation/rappel_vigie_juge.md"
    )
    parser.add_argument(
        "--out-csv", type=Path, default=REPO_ROOT / "evaluation/rappel_vigie_juge.csv"
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenere le markdown depuis --out-json sans aucun appel LLM.",
    )
    args = parser.parse_args()

    if args.render_only:
        report = json.loads(args.out_json.read_text(encoding="utf-8"))
        args.out_md.write_text(render_markdown(report), encoding="utf-8")
        write_csv(report, args.out_csv)
        print(f"Regenere depuis {args.out_json}: {args.out_md} et {args.out_csv}")
        return

    from openai import AsyncOpenAI

    from vigilance.utils.genai import get_openai_api_key

    api_key = get_openai_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY absent: impossible de lancer l'adjudication.")
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    pairing = str(reference.get("pairing"))
    wanted = {b.strip() for b in args.banks.split(",") if b.strip()}

    banks_report: list[dict[str, Any]] = []
    for bank_code, bank_ref in (reference.get("banks") or {}).items():
        if wanted and bank_code not in wanted:
            continue
        results_dir = args.results_root / bank_code / pairing
        if not results_dir.exists():
            print(f"[skip] {bank_code}: {results_dir} absent")
            continue
        tasks, volume = prepare_bank(bank_code, bank_ref, results_dir, args.top_k)
        if args.limit:
            tasks = tasks[: args.limit]
        print(f"[{bank_code}] {len(tasks)} items a juger...", flush=True)
        verdicts = await asyncio.gather(
            *(
                judge_one(
                    client,
                    args.model,
                    item,
                    ranked,
                    semaphore,
                    excerpt_chars=args.excerpt_chars,
                )
                for item, ranked in tasks
            )
        )
        covered = [v for v in verdicts if v["verdict"] == "couvert"]
        errors = [v for v in verdicts if v["verdict"] == "erreur"]
        channel_counts = Counter(
            (v.get("matched_candidate") or {}).get("channel") for v in covered
        )
        banks_report.append(
            {
                "bank_code": bank_code,
                "bank_label": bank_ref.get("bank_label"),
                "n_items": len(verdicts),
                "n_covered": len(covered),
                "n_errors": len(errors),
                "n_covered_high_confidence": sum(
                    1 for v in covered if v.get("confiance") == "haute"
                ),
                "recall": round(len(covered) / len(verdicts), 3) if verdicts else 0.0,
                "covered_by_channel": dict(channel_counts),
                "pipeline_volume": volume,
                "verdicts": verdicts,
            }
        )
        print(
            f"[{bank_code}] couverts {len(covered)}/{len(verdicts)}"
            f"{f' — {len(errors)} erreur(s)' if errors else ''}",
            flush=True,
        )
        if verdicts and len(errors) > len(verdicts) // 4:
            # Un quota epuise ou une cle invalide fait echouer tous les appels et
            # produirait un rappel de 0 % indistinguable d'un vrai resultat.
            raise SystemExit(
                f"[{bank_code}] {len(errors)}/{len(verdicts)} appels en echec — "
                f"resultat non exploitable. Premiere erreur: {errors[0]['raison']}"
            )

    total_items = sum(b["n_items"] for b in banks_report)
    total_covered = sum(b["n_covered"] for b in banks_report)
    report = {
        "pairing": pairing,
        "model": args.model,
        "top_k": args.top_k,
        "total_items": total_items,
        "total_covered": total_covered,
        "overall_recall": round(total_covered / total_items, 3) if total_items else 0.0,
        "banks": banks_report,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    write_csv(report, args.out_csv)

    print(f"\nRappel adjuge: {total_covered}/{total_items} = {report['overall_recall']:.0%}")
    for bank in banks_report:
        print(
            f"  {bank['bank_label']:<14} {bank['n_covered']:>3}/{bank['n_items']:<3} "
            f"= {bank['recall']:.0%}   canaux={bank['covered_by_channel']}"
        )
    print(f"\nRapport: {args.out_md}")


if __name__ == "__main__":
    asyncio.run(run())

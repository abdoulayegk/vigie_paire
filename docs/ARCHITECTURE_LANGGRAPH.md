# Architecture Technique : Système Multi-Agents LangGraph & Adaptateurs Modulaires

Ce document présente l'architecture technique complète du système de réconciliation et de vigie bancaire réorganisé autour de **LangGraph**, **LangChain Core** et de la **Clean Architecture** sur le projet `vigie_paire`.

---

## 1. Vue d'Ensemble de l'Architecture Découplée

L'architecture s'appuie sur un graphe d'états orienté (*StateGraph / DAG*) et des sous-packages modulaires où chaque responsabilité est isolée dans un composant de moins de 300 lignes.

```
[START] ──► NormalizerNode (Agent 1) ──► PrimaryMatcherNode (Agent 2) ──┬──► (RBC / Unmatched) ──► HybridRecoveryNode (Agent 3) ──┐
                                                                       │                                                         │
                                                                       └──► (Flux standard) ─────────────────────────────────────┴──► DevilAdvocateNode (Agent 4) ──► AMFTriageNode (Agent 5) ──► TextTriageNode (Agent 6) ──► [END]
```

---

## 2. Découpage des 6 Nœuds d'Agents Autonomes

| Nœud Agent | Fichier Source | Rôle & Responsabilité |
| :--- | :--- | :--- |
| **`normalizer` (`bank_normalizer_node`)** | `src/vigilance/graph/nodes.py` | Nettoyage des titres et marqueurs selon l'adaptateur spécifique de la banque. |
| **`primary_matcher` (`primary_matcher_node`)** | `src/vigilance/graph/nodes.py` | Rapprochement strict 1:1 initial entre les cartes du trimestre précédent et courant. |
| **`hybrid_recovery` (`hybrid_recovery_node`)** | `src/vigilance/graph/nodes.py` | Récupération vectorielle par embeddings pour les tableaux décalés ou renumérotés (ex: RBC). |
| **`devil_advocate` (`devil_advocate_node`)** | `src/vigilance/graph/nodes.py` | Inspection anti-faux-positifs avec sorties typées Pydantic `DevilAdvocateResponse`. |
| **`amf_triage` (`amf_triage_node`)** | `src/vigilance/graph/nodes.py` | Qualification des impacts métiers (`MAJEUR`/`MODÉRÉ`/`MINEUR`) et faits marquants AMF v2. |
| **`text_triage` (`text_triage_node`)** | `src/vigilance/graph/nodes.py` | Analyse de la posture de la banque (Prudente/Optimiste) et synthèse typée `TextTriageResponse`. |

---

## 3. Les 6 Adaptateurs Dédiés par Banque (`src/vigilance/extraction/adapters/`)

Chaque grande banque canadienne possède sa propre classe d'adaptation étanche dérivant de `BaseBankAdapter` :

1. **`RBCBankAdapter` (`rbc`)** : Gère le format canonique `Tableau XX` et les structures de risques RBC.
2. **`TDBankAdapter` (`td`)** : Nettoie les exposants de notes (`¹`, `²`, `³`, `⁴`) et la section NSFR.
3. **`BNCBankAdapter` (`bnc`)** : Gère la charte et les en-têtes de la Banque Nationale.
4. **`BMOBankAdapter` (`bmo`)** : Traite les tableaux de capital fusionnés BMO.
5. **`CIBCBankAdapter` (`cibc`)** : Gère les risques marchés/actions CIBC.
6. **`BNSBankAdapter` (`bns`)** : Gère la structure des rapports Scotia BNS.

> **Factory de récupération** : `get_bank_adapter(bank_code)` instancie l'adaptateur requis sans risque d'effet de bord entre banques.

---

## 4. Organisation des Packages Modulaires

- **`src/vigilance/extraction/components/`** :
  - `footnote_extractor.py` : Capture 100% propre et complète des notes de bas de page.
  - `section_detector.py` : Identification des clés de section (`gestion_capital`, `gestion_risques`).
  - `table_extractor.py` : Contrôle de complétude et délimitation des tableaux.

- **`src/vigilance/quality/checks/`** :
  - `completeness_check.py` : Contrôle de complétude des extractions.
  - `indicator_check.py` : Validation de cohérence des métriques chiffrées.
  - `schema_check.py` : Vérification de la conformité des schémas JSON canoniques.

- **`src/vigilance/export/`** :
  - `excel_exporter.py` : Génération des rapports Excel (`.xlsx`).
  - `pdf_exporter.py` : Génération des synthèses PDF de vigie.

- **`src/vigilance/dash_app/components/detail_widgets/`** :
  - `proof_badge_builder.py` : Badges HTML d'évolutions.
  - `proof_card_builder.py` : Cartes de preuves visuelles côte-à-côte.

- **`src/vigilance/dash_app/layouts/text_analysis/`** :
  - `text_filters.py` : Barre de filtres interactifs.
  - `text_cards.py` : Cartes de sections textuelles comparées.
  - `text_panels.py` : Volets d'analyse IA détaillée et posture.

---

## 5. Guide d'Exécution CLI (`vigie-graph-run`)

Pour exécuter la comparaison Multi-Agent LangGraph en ligne de commande :

```bash
# Exécution du graphe LangGraph pour une banque (ex: RBC)
.venv/bin/python3 -m vigilance.cli.run_graph --bank rbc --previous 2024-t4 --current 2025-t4

# Validation de la suite de tests automatisés (1 123 / 1 123 PASS)
.venv/bin/pytest tests/unit/
```

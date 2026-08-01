# Architecture Technique : Système Multi-Agents LangGraph & LangChain

Ce document présente l'architecture technique du système de réconciliation et de vigie bancaire réorganisé autour de **LangGraph** et **LangChain Core** sur le projet `vigie_paire`.

---

## 1. Vue d'Ensemble & Principes Directeurs

L'architecture s'appuie sur un graphe d'états orienté (*StateGraph / DAG*) où chaque étape du traitement est un **nœud d'agent autonome**.

```
[START] ──► NormalizerNode ──► PrimaryMatcherNode ──┬──► (RBC or Unmatched) ──► HybridRecoveryNode ──┐
                                                   │                                                │
                                                   └──► (Standard flow) ────────────────────────────┴──► DevilAdvocateNode ──► AMFTriageNode ──► [END]
```

### Principes clés :
1. **Mémoire d'État Unifiée (`ComparisonState`)** : Un objet Pydantic central échangé entre tous les agents.
2. **Sorties Pydantic Garanties (`Structured Output`)** : Utilisation de `ChatOpenAI.with_structured_output(...)` pour éliminer le parsing manuel JSON.
3. **Résilience et Reprise (`with_retry` & `MemorySaver`)** : Tentatives automatiques (3 retries) en cas d'erreur API OpenAI et checkpoints par identifiant de session (`thread_id`).

---

## 2. Découpage des Nœuds d'Agents

| Nœud Agent | Fichier Source | Rôle & Responsabilité |
| :--- | :--- | :--- |
| **`normalizer`** | `src/vigilance/graph/nodes.py` | Nettoyage des titres par profil de banque (ex: suppression des suffixes `Tableau XX` pour RBC). |
| **`primary_matcher`** | `src/vigilance/graph/nodes.py` | Rapprochement strict 1:1 initial entre les cartes du trimestre précédent et courant. |
| **`hybrid_recovery`** | `src/vigilance/graph/nodes.py` | Récupération vectorielle par embeddings pour les tableaux décalés ou renumérotés. |
| **`devil_advocate`** | `src/vigilance/graph/nodes.py` | Inspection anti-faux-positifs avec sorties typées `DevilAdvocateResponse`. |
| **`amf_triage`** | `src/vigilance/graph/nodes.py` | Qualification des impacts métiers (`MAJEUR`/`MODÉRÉ`/`MINEUR`) et faits marquants AMF. |

---

## 3. Exemple de Code & Invocation du Graphe

```python
from vigilance.graph import build_comparison_graph, ComparisonState

# 1. Instanciation du graphe avec Checkpointer natif
graph = build_comparison_graph()

# 2. Préparation de l'état initial
state = ComparisonState(
    bank_code="RBC",
    year_current=2025,
    quarter_current="T4",
    year_previous=2024,
    quarter_previous="T4",
    previous_cards=[...],
    current_cards=[...]
)

# 3. Invocation résiliente avec thread_id
config = {"configurable": {"thread_id": "session_rbc_2025_t4"}}
final_result = graph.invoke(state, config=config)
```

---

## 4. Stratégie de Test et Validation

Le module est validé par la suite de tests unitaires et d'intégration :
- `tests/unit/test_langgraph_pipeline.py` (Validation du flux d'états)
- `tests/unit/test_langgraph_llm_nodes.py` (Validation des sorties typées Pydantic)
- `tests/unit/test_langgraph_triage_node.py` (Validation de l'agent Triage AMF & Checkpointing)
- `tests/unit/test_langgraph_e2e_runner.py` (Validation de l'exportation `comparison.json` et `comparison.xlsx`)

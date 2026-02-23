import logging

import numpy as np
import pdfplumber
from openai import OpenAI

from ..utils.genai import get_openai_api_key

logger = logging.getLogger(__name__)


class SemanticRouter:
    """
    Scans a PDF to identifying relevant sections (Capital, Risk) using OpenAI Embeddings.
    It 'skims' the document (cheap text extraction) to find the correct pages,
    so that heavy processing is only done on the relevant parts.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_openai_api_key()
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for SemanticRouter")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"

        # Specific section titles per bank (from User requirements)
        self.bank_anchors_config = {
            "bnc": ["Gestion du capital", "Gestion des risques"],
            "rbc": [
                "Examen de la conjoncture économique, des marchés et du contexte réglementaire",
                "Gestion du risque",
                "Gestion des fonds propres",
            ],
            "bns": [  # Scotia
                "Gestion du risque",
                "Gestion du capital",
                "Faits nouveaux en matière de réglementation",
            ],
            "cibc": ["Gestion des fonds propres", "Gestion du risque"],
            "td": [
                "Situation des fonds propres",
                "Facteurs de risque et gestion des risques",
                "Gestion des risques",
            ],
            "bmo": [
                "Gestion du capital",
                "Autres faits nouveaux en matière de réglementation",
                "Gestion des risques",
            ],
        }

        # Pre-compute anchor embeddings
        self.bank_embeddings = self._precompute_bank_embeddings()

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a text chunk."""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def _precompute_bank_embeddings(self) -> dict[str, dict[str, list[float]]]:
        """Compute embeddings for all bank anchors once."""
        embeddings = {}
        for bank, sections in self.bank_anchors_config.items():
            embeddings[bank] = {}
            for section in sections:
                embeddings[bank][section] = self._get_embedding(section)
        return embeddings

    def extract_text_per_page(self, pdf_path: str) -> list[tuple[int, str]]:
        """
        Cheaply extract text from every page.
        Returns list of (page_num, text_content).
        """
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append((i + 1, text[:3000]))
        return pages_text

    def find_via_toc(self, pdf_path: str, bank_code: str) -> dict[str, list[int]]:
        """
        Tente de trouver les sections via la Table des Matières (TOC) sur les 6 premières pages.
        Utilise GPT-4o pour parser la structure.
        """
        logger.info(f"Tentative de détection par TOC (Pages 1-6) pour {bank_code}...")

        # 1. Lire les 6 premières pages
        toc_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(min(6, len(pdf.pages))):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    toc_text += f"\n--- Page {i + 1} ---\n{page_text}"

        if len(toc_text) < 100:
            logger.warning("Texte insuffisant pour analyse TOC.")
            return {}

        # 2. Demander à GPT-4o d'extraire les pages
        system_prompt = """
        Tu es un expert en analyse de rapports financiers. Ta tâche est d'identifier les numéros de page de début pour les sections spécifiées dans la Table des Matières (Sommaire).
        
        Retourne UNIQUEMENT un JSON :
        {
            "sections": {
                 "Nom Section": { "start_page": 10 }
            }
        }
        
        Règles:
        - Cherche "Gestion du capital" (ou Capital Management) et "Gestion des risques" (ou Risk Management).
        - Si tu trouves, note la page de début.
        - Si tu ne trouves pas, ne met rien.
        """

        user_prompt = f"""
        Voici le début du document (contenant potentiellement la Table des Matières).
        Banque cible : {bank_code.upper()}
        
        Recherche les pages pour :
        1. Gestion du capital (Capital Management)
        2. Gestion des risques (Risk Management)
        
        Texte extrait :
        {toc_text}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            detected_sections = data.get("sections", {})
            results = {}  # Format: { "Gestion du capital": [20, 21, ...], ... }

            # Mapping des noms standards (clés) vers les pages trouvées
            # On cherche à remplir les clés defines dans self.bank_anchors_config[bank_code]

            expected_anchors = self.bank_anchors_config.get(bank_code, [])

            # Heuristique simple:
            # - Si GPT trouve "Capital ...", on associe au premier anchor ("Gestion du capital")
            # - Si GPT trouve "Risque ...", on associe au dernier anchor ("Gestion des risques")

            start_pages = {}

            for detected_name, info in detected_sections.items():
                start_page = info.get("start_page")
                if not isinstance(start_page, int):
                    continue

                name_lower = detected_name.lower()

                if "capital" in name_lower or "fonds propres" in name_lower:
                    # Associer au premier anchor qui parle de capital
                    for anchor in expected_anchors:
                        if "capital" in anchor.lower() or "fonds propres" in anchor.lower():
                            start_pages[anchor] = start_page
                            break

                elif "risqu" in name_lower or "risk" in name_lower:
                    # Associer au premier anchor qui parle de risque
                    for anchor in expected_anchors:
                        if "risqu" in anchor.lower() or "risk" in anchor.lower():
                            start_pages[anchor] = start_page
                            break

            # Construire les ranges (Start -> Start Next Section ou +10 pages)
            sorted_starts = sorted(start_pages.items(), key=lambda x: x[1])

            for i, (section, start) in enumerate(sorted_starts):
                # Fin = Debut section suivante - 1, ou Debut + 15 pages par défaut
                if i < len(sorted_starts) - 1:
                    end = sorted_starts[i + 1][1] - 1
                else:
                    end = start + 15  # Par défaut pour la dernière section

                # Générer la liste des pages
                results[section] = list(range(start, end + 1))
                logger.info(f"TOC Détecté : {section} -> Pages {start} à {end}")

            return results

        except Exception as e:
            logger.error(f"Erreur TOC GPT: {e}")
            return {}

    def find_relevant_pages(self, pdf_path: str, bank_code: str) -> dict[str, list[int]]:
        """
        Point d'entrée principal :
        1. Essaie via TOC (GPT-4o) sur les 6 premières pages.
        2. Si échec (ou section manquante), fallback sur recherche sémantique (Embeddings).
        """
        bank_code = bank_code.lower()
        if bank_code not in self.bank_embeddings:
            logger.warning(f"Banque inconnue {bank_code}.")
            return {}

        logger.info(f"Scanner: {pdf_path} pour {bank_code}")

        # 1. Stratégie TOC
        results_toc = self.find_via_toc(pdf_path, bank_code)

        # Si le TOC a trouvé des résultats pour au moins une section, on considère ça comme un succès partiel
        # Mais pour la robustesse, on peut compléter avec les Embeddings si une section clé manque.

        expected_sections = self.bank_anchors_config[bank_code]
        missing_sections = [s for s in expected_sections if s not in results_toc]

        if not missing_sections:
            logger.info("Succès complet via TOC.")
            return results_toc

        if results_toc:
            logger.info(
                f"Succès partiel TOC. Sections trouvées: {list(results_toc.keys())}. Manquantes: {missing_sections}"
            )
            # On garde ce qu'on a trouvé
            final_results = results_toc.copy()
        else:
            logger.info("Échec TOC ou aucune section trouvée. Fallback sur Embeddings.")
            final_results = {}

        # 2. Stratégie Embeddings (pour les sections manquantes ou tout si TOC échoue)
        if missing_sections:
            logger.info(f"Recherche sémantique pour les sections: {missing_sections}...")
            page_contents = self.extract_text_per_page(pdf_path)

            target_embeddings = self.bank_embeddings[bank_code]

            # Initialiser les listes pour les sections manquantes
            for s in missing_sections:
                final_results[s] = []

            for page_num, text in page_contents:
                if len(text) < 50:
                    continue
                page_embedding = self._get_embedding(text)

                for section_name in missing_sections:
                    anchor_emb = target_embeddings[section_name]
                    if np.dot(page_embedding, anchor_emb) > 0.48:
                        final_results[section_name].append(page_num)

            # Post-traitement uniquement pour les nouvelles trouvailles
            cleaned_partials = self._post_process_ranges(
                {k: final_results[k] for k in missing_sections}
            )
            final_results.update(cleaned_partials)

        return final_results

    def _post_process_ranges(self, results: dict[str, list[int]]) -> dict[str, list[int]]:
        """
        Clean up page lists:
        1. Remove outliers (single pages far from the cluster).
        2. Fill gaps (if 20, 22 are found, include 21).
        """
        cleaned_results = {}

        for section, pages in results.items():
            if not pages:
                cleaned_results[section] = []
                continue

            # Sort and de-duplicate
            pages = sorted(list(set(pages)))

            # Simple clustering: Keep the largest continuous block
            # For BNC/TD reports, these sections are usually contiguous.

            if not pages:
                continue

            # Identify clusters
            clusters = []
            current_cluster = [pages[0]]

            for i in range(1, len(pages)):
                if pages[i] <= pages[i - 1] + 2:  # Allow 1-page gap
                    current_cluster.append(pages[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pages[i]]
            clusters.append(current_cluster)

            # Pick longest cluster (most likely the actual section, avoiding random mentions in TOC or summary)
            longest_cluster = max(clusters, key=len)

            # Fill gaps in the cluster
            full_range = list(range(longest_cluster[0], longest_cluster[-1] + 1))
            cleaned_results[section] = full_range

        return cleaned_results

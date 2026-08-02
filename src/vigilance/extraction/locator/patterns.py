"""Donnees de detection des sections : patterns regex et listes de reference.

Extrait de ``section_locator.py`` sans modification. Ce module ne contient que des
donnees ; toute la logique reste dans le localisateur.
"""

# Patterns de detection par type de section
# Note: L'ordre des patterns est important - les plus specifiques en premier
SECTION_PATTERNS = {
    "gestion_capital": {
        "patterns": [
            # Variantes exactes (prioritaires)
            r"gestion\s+du\s+capital",
            r"gestion\s+des\s+fonds\s+propres",
            r"situation\s+des\s+fonds\s+propres",
            # RBC: Examen de la conjoncture economique (avec/sans accents)
            r"examen\s+de\s+la\s+conjoncture\s+[eé]conomique",
            # Variantes avec contexte reglementaire
            r"fonds\s+propres\s+r[eé]glementaires",
            r"capital\s+r[eé]glementaire",
            # Variantes anglaises
            r"capital\s+management",
            r"regulatory\s+capital",
            # Variantes partielles (moins prioritaires)
            r"capitaux\s+propres",
        ],
        # Mots-cles pour valider le contenu (variantes avec/sans accents)
        "keywords": [
            "cet1",
            "tier 1",
            "tier 2",
            "fonds propres",
            "capital",
            "capitaux",
            "ratio",
            "levier",
            "leverage",
            "bâle",
            "bale",
            "bsif",
            "tlac",
            "lcr",
            "nsfr",
            "liquidit",
            "dividende",
            "rachat",
            "actions",
        ],
        # Termes qui indiquent que ce n'est PAS la bonne section
        "exclude_patterns": [
            r"risque\s+de",  # Eviter confusion avec sections risques
            r"rendement\s+des?\s+capitaux\s+propres",
        ],
    },
    "gestion_risques": {
        "patterns": [
            # Variantes principales (titre de section)
            r"gestion\s+des\s+risques",
            r"gestion\s+du\s+risque(?!\s+de\s+cr[eé]dit)",  # Pas suivi de "de credit"
            r"risk\s+management",
            r"facteurs?\s+de\s+risque\s+et\s+gestion",
            # Variantes avec contexte
            r"facteurs?\s+de\s+risque",
            r"exposition\s+aux?\s+risques?",
            # Sections autonomes pouvant remplacer le titre global des risques
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?donn[eé]es(?:,\s*technologie\s+et\s+cybers[eé]curit[eé])?",
            r"risques?\s+technologiques?",
            r"technolog(?:ie|ique),?\s+cybers[eé]curit[eé]\s+et\s+donn[eé]es",
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?tiers",
            r"gestion\s+des?\s+fournisseurs",
            r"services?\s+infonuagiques?",
            r"r[eé]silience\s+op[eé]rationnelle",
            r"protection\s+des?\s+donn[eé]es\s+et\s+vie\s+priv[eé]e",
        ],
        # Mots-cles pour valider le contenu (variantes avec/sans accents)
        "keywords": [
            "risque",
            "risk",
            "crédit",
            "credit",
            "marché",
            "marche",
            "market",
            "liquidité",
            "liquidite",
            "liquidity",
            "opérationnel",
            "operationnel",
            "operational",
            "var",
            "exposition",
            "exposure",
            "provision",
            "perte",
            "loss",
            "portefeuille",
            "portfolio",
            "stress",
            "scénario",
            "scenario",
            "données",
            "donnees",
            "data",
            "technologie",
            "technology",
            "cybersécurité",
            "cybersecurite",
            "cloud",
            "infonuagique",
            "tiers",
            "fournisseur",
            "impartition",
            "résilience",
            "resilience",
            "vie privée",
            "vie privee",
            "qualité des données",
            "qualite des donnees",
            "intégrité des données",
            "integrite des donnees",
            "confidentialité",
            "confidentialite",
            "protection des données",
            "protection des donnees",
            "localisation des données",
            "localisation des donnees",
            "souveraineté",
            "souverainete",
            "conservation des données",
            "conservation des donnees",
            "traçabilité",
            "tracabilite",
            "lignage",
            "cycle de vie des données",
            "cycle de vie des donnees",
            "fuite de données",
            "fuite de donnees",
            "tiers critique",
            "fournisseur critique",
            "sous-traitant",
            "concentration des fournisseurs",
            "verrouillage fournisseur",
            "stratégie de sortie",
            "strategie de sortie",
            "continuité des services",
            "continuite des services",
            "exigence contractuelle",
        ],
        # Sous-sections qui font partie de "Gestion des risques"
        "subsections": [
            r"risque\s+de\s+cr[eé]dit",
            r"risque\s+de\s+march[eé]",
            r"risque\s+de\s+liquidit[eé]",
            r"risque\s+op[eé]rationnel",
            r"credit\s+risk",
            r"market\s+risk",
            r"liquidity\s+risk",
            r"operational\s+risk",
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?donn[eé]es",
            r"risques?\s+technologiques?",
            r"risques?\s+(?:li[eé]s?\s+aux?\s+)?tiers",
            r"risques?\s+li[eé]s?\s+[àa]\s+l['’]impartition",
            r"services?\s+infonuagiques?",
            r"r[eé]silience\s+op[eé]rationnelle",
            r"protection\s+des?\s+donn[eé]es",
            r"vie\s+priv[eé]e",
            r"data\s+risk",
            r"technology\s+risk",
            r"third[-\s]party\s+risk",
            r"cloud\s+risk",
            r"operational\s+resilience",
        ],
        "exclude_patterns": [
            r"chef\s+des?\s+risques",
            r"chef\s+de\s+la\s+gestion\s+des?\s+risques?",
            r"comit[ée]\s+de\s+gestion\s+des?\s+risques?",
            r"structure\s+de\s+gestion\s+des?\s+risques?",
            r"gestion\s+du\s+risque\s+d['e]\s*entreprise",
            r"gestion\s+du\s+risque\s+li[eé]",
        ],
    },
    "gestion_reglementation": {
        "patterns": [
            # RBC: Examen de la conjoncture economique
            r"examen\s+de\s+la\s+conjoncture\s+[eé]conomique",
            r"contexte\s+r[eé]glementaire\s+et\s+perspectives",
            # BNS/BMO: Faits nouveaux en matiere de reglementation
            r"faits?\s+nouveaux?\s+en\s+mati[eè]re\s+de\s+r[eé]glementation",
            r"autres?\s+faits?\s+nouveaux?\s+en\s+mati[eè]re\s+de\s+r[eé]glementation",
            # Variantes generiques
            r"mise\s+[àa]\s+jour\s+r[eé]glementaire",
            r"[eé]volution\s+r[eé]glementaire",
        ],
        "keywords": [
            "reglementation",
            "réglementation",
            "bsif",
            "bale",
            "bâle",
            "normes",
            "conjoncture",
            "perspectives",
            "contexte",
            "economique",
            "économique",
        ],
        "exclude_patterns": [],
    },
}

# Patterns des sections qui suivent typiquement nos sections cibles
# Utilises pour determiner la FIN d'une section
FOLLOWING_SECTION_PATTERNS = {
    "gestion_capital": [
        r"gestion\s+des?\s+risques?",
        r"gestion\s+du\s+risque",
        r"risque\s+de\s+cr[eé]dit",
        r"facteurs?\s+de\s+risque",
        r"r[eé]sultats?\s+consolid[eé]s?",
        r"analyse\s+des?\s+r[eé]sultats?",
    ],
    "gestion_risques": [
        r"normes\s+et\s+m[eé]thodes\s+comptables",
        r"m[eé]thodes\s+et\s+estimations\s+comptables",
        r"m[eé]thodes\s+comptables\s+significatives",
        r"[eé]tats?\s+financiers?",
        r"informations?\s+compl[eé]mentaires?",
        r"renseignements?\s+compl[eé]mentaires?",
        r"donn[eé]es?\s+compl[eé]mentaires?",
        r"annexes?",
        r"notes?\s+aux?\s+[eé]tats",
        r"glossaire",
        r"d[eé]finitions?",
    ],
    "gestion_reglementation": [
        r"gestion\s+des?\s+fonds?\s+propres?",
        r"gestion\s+du\s+capital",
        r"gestion\s+des?\s+risques?",
        r"gestion\s+du\s+risque",
        r"[eé]tats?\s+financiers?",
    ],
}

SECTION_TITLE_ALIASES: dict[str, list[str]] = {
    "gestion_capital": [
        "Gestion du capital",
        "Gestion des fonds propres",
        "Situation des fonds propres",
    ],
    "gestion_risques": [
        "Gestion des risques",
        "Gestion du risque",
        "Risk management",
    ],
    "gestion_reglementation": [
        "Réglementation",
        "Reglementation",
    ],
}

# Repères sémantiques réservés aux rapports annuels T4. Ils ne contiennent
# volontairement aucun numéro de page : le localisateur doit retrouver les
# titres dans l'ordre physique du PDF. Les titres configurés ici complètent les
# alias génériques et évitent d'imposer aux rapports T1-T3 la structure T4.
T4_SECTION_TITLE_PROFILES: dict[str, dict[str, dict[str, list[str]]]] = {
    "td": {
        "gestion_capital": {
            "start": ["Situation des fonds propres"],
            "end": ["Situation financière du groupe"],
        },
        "gestion_risques": {
            "start": ["Facteurs de risque et gestion des risques"],
            "end": ["Normes et méthodes comptables"],
        },
    },
    "bmo": {
        "gestion_capital": {
            "start": ["Gestion globale du capital"],
            # Les rapports annuels BMO intercalent un vrai chapitre sur les
            # entités structurées/titrisation entre le capital et les risques.
            # Les titres, et non des pages fixes, bornent donc le capital.
            "end": [
                "Entités structurées et titrisation",
                "Entités de titrisation soutenues par BMO",
                "Gestion globale des risques",
            ],
        },
        "gestion_risques": {
            "start": ["Gestion globale des risques"],
            "end": ["Questions comptables"],
        },
    },
    "rbc": {
        "gestion_capital": {
            "start": ["Gestion des fonds propres"],
            "end": ["Contrôles et procédures"],
        },
        "gestion_risques": {
            "start": ["Gestion du risque"],
            "end": ["Gestion des fonds propres"],
        },
    },
    "bns": {
        "gestion_capital": {
            # Le rapport BNS ouvre ce bloc par le chapitre financier, puis
            # présente la gestion du capital sans titre racine autonome.
            "start": ["Situation financière du Groupe"],
            "end": ["Arrangements hors bilan"],
        },
        "gestion_risques": {
            "start": ["Gestion du risque"],
            "end": ["Contrôles et méthodes comptables", "Contrôles et procédures"],
        },
    },
    "bnc": {
        "gestion_capital": {
            "start": ["Gestion du capital"],
            "end": ["Gestion des risques"],
        },
        "gestion_risques": {
            "start": ["Gestion des risques"],
            "end": ["Méthodes comptables significatives et estimations comptables"],
        },
    },
    "cibc": {
        "gestion_capital": {
            "start": ["Gestion des fonds propres"],
            "end": ["Arrangements hors bilan"],
        },
        "gestion_risques": {
            "start": ["Gestion du risque"],
            "end": ["Questions relatives à la comptabilité et au contrôle"],
        },
    },
}

# Sous-sections de "Gestion des risques" qui ne doivent pas etre confondues
# avec la section principale. Ces sous-sections font PARTIE de la section risques.
RISK_SUBSECTIONS = [
    "Risque de credit",
    "Risque de marche",
    "Risque de liquidite",
    "Risque operationnel",
    "Risque de taux d'interet",
    "Risque de change",
    "Divulgation d'information sur les risques",
    "Divulgation d'informations sur les risques",
    "Divulgation d’information sur les risques",
    "Divulgation d’informations sur les risques",
    "Divulgation dinformation sur les risques",
    "Divulgation dinformations sur les risques",
    "Cotes de credit",
    "Cotes de crédit",
    "Credit Risk",
    "Market Risk",
    "Liquidity Risk",
    "Operational Risk",
    "Risque lié aux données",
    "Risque lié aux donnees",
    "Risque technologique",
    "Risque lié aux tiers",
    "Risque lie aux tiers",
    "Risque lié à l'impartition",
    "Services infonuagiques",
    "Résilience opérationnelle",
    "Resilience operationnelle",
    "Protection des données",
    "Protection des donnees",
    "Vie privée",
    "Vie privee",
    "Data Risk",
    "Technology Risk",
    "Third-Party Risk",
    "Cloud Risk",
    "Operational Resilience",
]

# Patterns pour detecter la Table des matieres
TOC_PATTERNS = [
    r"table\s+des\s+mati[eè]res",
    r"sommaire",
    r"table\s+of\s+contents",
    r"contents",
    # BNC utilise "Rapport de gestion" comme en-tete de la page TDM
    r"rapport\s+de\s+gestion",
    # Patterns additionnels pour detecter les pages avec TDM
    r"aper[çc]u\s+du\s+rapport",
    r"guide\s+du\s+lecteur",
]

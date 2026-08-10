# Installation Windows

Le projet propose deux profils d'installation :

- **Revue analyste** : interface et rendu de résultats existants uniquement.
  Ce profil n'installe ni Docling, ni OpenAI, ni les dépendances des pipelines.
- **Développement complet** : extraction, comparaisons, pipelines, interface et
  outils de développement.

Les deux profils lancent la même application :

```powershell
python -m vigie.interface.app
```

## Prérequis communs

- Python 3.10 ou une version plus récente ;
- Git ;
- PowerShell.

Vérifier les installations :

```powershell
python --version
git --version
```

## Profil 1 — Revue analyste minimale

Ce profil sert à consulter les résultats produits par l'équipe de développement,
à prendre des décisions et à exporter le rendu. Il ne lance aucune extraction et
ne fait aucun appel LLM. Aucune clé API et aucun fichier `.env` ne sont requis.

### 1. Récupérer le projet

```powershell
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
```

Si le projet est déjà présent :

```powershell
git pull
```

### 2. Créer l'environnement léger

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-interface.txt
python -m pip install -e . --no-deps
```

Si PowerShell bloque l'activation, autoriser les scripts uniquement pour la
session en cours, puis recommencer l'activation :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Lancer la revue

Pour utiliser les résultats inclus dans le dépôt :

```powershell
python -m vigie.interface.app
```

L'installation légère est détectée automatiquement. Le nom d'utilisateur
Windows sert d'identifiant de revue ; aucune option supplémentaire n'est
obligatoire.

Pour utiliser un dossier de résultats partagé ou copié ailleurs :

```powershell
python -m vigie.interface.app --resultats "C:\Chemin\Vers\resultats"
```

Ouvrir ensuite [http://127.0.0.1:8050](http://127.0.0.1:8050), sélectionner la
banque, l'année et le trimestre, puis charger l'analyse.

Le dossier fourni à `--resultats` doit être la racine qui contient les dossiers
des banques, par exemple :

```text
resultats\
  bmo\
  bnc\
  bns\
  cibc\
  rbc\
  td\
```

Les décisions sur les indicateurs sont enregistrées dans un fichier individuel
associé à l'utilisateur Windows, sans modifier le fichier de comparaison
d'origine. Les options `--revue` et `--analyste` restent disponibles uniquement
pour forcer le mode ou remplacer l'identifiant détecté automatiquement.

## Profil 2 — Développement complet

Ce profil est nécessaire pour exécuter les extractions et les comparaisons, faire
des appels LLM, modifier le projet ou lancer les tests.

### Option recommandée avec uv

```powershell
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
uv sync --group dev
copy .env.example .env
```

Renseigner `OPENAI_API_KEY` dans `.env` avant d'exécuter un pipeline qui utilise
un LLM. Lancer l'application complète :

```powershell
uv run python -m vigie.interface.app
```

### Option avec pip

```powershell
git clone https://github.com/abdoulayegk/vigie_paire.git
cd vigie_paire
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
python -m vigie.interface.app
```

## Dépannage

### Aucun résultat n'apparaît

Au démarrage, le terminal affiche le chemin du dossier utilisé et le nombre
d'analyses détectées. Vérifier que :

- le chemin vise bien `outputs\resultats` ou une copie de cette racine ;
- le dossier contient des sous-dossiers de banques et des fichiers
  `comparison.json` ;
- le terminal n'affiche pas `Analyses détectées : 0`.

Ne pas fournir le chemin du dépôt entier, d'une banque seule ou d'un fichier
`comparison.json` à `--resultats`.

### Une ancienne interface apparaît

Fermer les anciens terminaux et processus Python, réactiver l'environnement du
projet, puis relancer exactement :

```powershell
python -m vigie.interface.app
```

Ouvrir l'adresse et le port affichés par ce nouveau terminal. Si le poste avait
déjà une ancienne installation, recréer `.venv` puis reprendre les étapes du
profil choisi.

### Le port 8050 est déjà utilisé

```powershell
python -m vigie.interface.app --port 8051
```

Ouvrir alors [http://127.0.0.1:8051](http://127.0.0.1:8051).

# Packaging — VigieRegDesjardins (.exe Windows reader)

Ce dossier contient les artefacts pour empaqueter l'interface Dash en
mode "lecture seule" sous forme d'un executable Windows distribuable
sur SharePoint.

## Architecture

- **Mode reader** = uniquement consultation/validation des comparaisons
  deja generees. Aucun appel LLM, aucun lancement de pipeline.
- **Annotations multi-utilisateurs** : chaque analyste ecrit dans son
  propre `comparison.review_state.<username>.json` (consolidation faite
  ulterieurement cote pipeline).
- **Dossier `resultats`** : choisi au 1er lancement via boite de
  dialogue, memorise dans `%APPDATA%\VigieRegDesjardins\config.json`.
  Surcharge possible via `VIGIE_RESULTATS_DIR`.

## Fichiers

| Fichier | Role |
|---|---|
| `vigie_reader.spec` | Spec PyInstaller : entry point, hidden imports, excludes (docling, openai, scipy...) |
| `../.github/workflows/build-reader.yml` | Build automatique Windows + release sur tag `reader-v*` |

## Build local (test)

Sur **Mac/Linux** : impossible — PyInstaller produit des binaires pour
l'OS hote, pas de cross-compilation Windows.

Sur **Windows** :
```powershell
uv sync
uv pip install pyinstaller
uv run pyinstaller packaging/vigie_reader.spec --clean --noconfirm
# Sortie : dist/VigieRegDesjardins/VigieRegDesjardins.exe
```

## Build automatise (recommande)

Pour produire un release :

```bash
git tag reader-v0.1.0
git push origin reader-v0.1.0
```

Le workflow `build-reader.yml` :
1. Build sur `windows-latest`.
2. Zippe `dist/VigieRegDesjardins/` -> `VigieRegDesjardins-reader-v0.1.0.zip`.
3. Cree une GitHub Release avec le zip attache.

Pour un test sans release (build a la demande) :
- GitHub -> Actions -> "Build VigieRegDesjardins" -> "Run workflow".
- L'artifact reste disponible 30 jours.

## Distribution sur SharePoint

1. Telecharger `VigieRegDesjardins-reader-vX.Y.Z.zip` depuis la release.
2. Uploader sur SharePoint dans un dossier dedie (ex. `Outils/VigieRegDesjardins/`).
3. L'analyste : telecharger le zip localement, decompresser, double-cliquer
   sur `VigieRegDesjardins.exe`.
4. Au 1er lancement, selectionner le dossier `resultats` synchronise via
   OneDrive (typiquement `C:\Users\<username>\Desjardins\Vigie\resultats`).

## Antivirus / EDR

PyInstaller produit des executables non signes qui peuvent declencher
des alertes EDR. Si necessaire, voir avec la securite TI Desjardins
pour signer le binaire (workflow a ajouter dans `build-reader.yml` via
`signtool` + secret cert).

## Mise a jour de la liste des excludes

Si le bundle est trop gros (> 300 Mo), inspecter ce qui est inclus :
```powershell
uv run pyi-archive_viewer dist/VigieRegDesjardins/VigieRegDesjardins.exe
```

Et completer `excludes=[...]` dans `vigie_reader.spec` avec les
packages identifies.

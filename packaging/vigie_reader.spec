# PyInstaller spec — VigieRegDesjardins (reader Windows .exe)
#
# Build :
#     uv run pyinstaller packaging/vigie_reader.spec --clean --noconfirm
#
# Sortie :
#     dist/VigieRegDesjardins/VigieRegDesjardins.exe  (mode onefolder)
#
# Mode onefolder (vs onefile) : demarrage plus rapide, debug plus simple,
# moins d'alertes EDR. La distribution se fait alors en zippant le dossier
# ``dist/VigieRegDesjardins/``.

# ruff: noqa
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# --- Collecte automatique des packages clefs ----------------------------------
# ``collect_all`` ramene modules + data + binaries pour les libs qui en ont
# besoin (templates, themes, ressources statiques).
dash_datas, dash_binaries, dash_hidden = collect_all("dash")
dbc_datas, dbc_binaries, dbc_hidden = collect_all("dash_bootstrap_components")
plotly_datas, plotly_binaries, plotly_hidden = collect_all("plotly")
fitz_datas, fitz_binaries, fitz_hidden = collect_all("fitz")  # PyMuPDF
fitz_datas2, fitz_binaries2, fitz_hidden2 = collect_all("pymupdf")
flask_datas, flask_binaries, flask_hidden = collect_all("flask")

# --- Sources du projet --------------------------------------------------------
# Le package ``vigilance`` est installe via setuptools (src layout). On
# laisse PyInstaller le decouvrir, mais on force quelques sous-modules en
# cas d'imports dynamiques ou paresseux.
vigilance_hidden = collect_submodules("vigilance.dash_app")

hiddenimports = (
    dash_hidden
    + dbc_hidden
    + plotly_hidden
    + fitz_hidden
    + fitz_hidden2
    + flask_hidden
    + vigilance_hidden
    + [
        "vigilance.dash_app.reader_config",
        "vigilance.review_storage",
        "vigilance.review_export",
        "vigilance.quarter_utils",
        "vigilance.ui_config",
        "tkinter",
        "tkinter.filedialog",
    ]
)

datas = (
    dash_datas
    + dbc_datas
    + plotly_datas
    + fitz_datas
    + fitz_datas2
    + flask_datas
    + [
        # Assets Dash (CSS custom)
        ("../src/vigilance/dash_app/assets", "vigilance/dash_app/assets"),
    ]
)

binaries = (
    dash_binaries
    + dbc_binaries
    + plotly_binaries
    + fitz_binaries
    + fitz_binaries2
    + flask_binaries
)

# --- Excludes : libs pipeline non necessaires en mode reader -----------------
# Ces packages sont references par certains imports (ex.
# ``vigilance.comparison_runner``) mais ces chemins ne sont jamais executes
# dans le reader (imports paresseux + callbacks d'upload masques).
excludes = [
    # LLM / extraction lourde
    "docling",
    "docling_core",
    "docling_ibm_models",
    "openai",
    "anthropic",
    "tiktoken",
    "transformers",
    "torch",
    "torchvision",
    "tensorflow",
    # Extraction PDF lourde non utilisee par le reader
    "pdfplumber",
    "pdfminer",
    "pypdf",
    "pypdfium2",
    # Calcul scientifique inutile pour l'affichage
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "pandas.tests",
    # Outils dev
    "pytest",
    "sphinx",
    "ruff",
    "black",
    "mypy",
    "IPython",
    "notebook",
    "jupyter",
    "ipykernel",
    # Conversion docs
    "PIL.ImageTk",  # pas utilise; tkinter sans Tk Image
]


a = Analysis(
    ["../src/vigilance/dash_app/reader.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VigieRegDesjardins",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # console visible : utile pour voir les logs au demarrage
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VigieRegDesjardins",
)

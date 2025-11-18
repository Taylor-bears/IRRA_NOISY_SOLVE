import os
import nltk

# Resolve target directory under project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TARGET = os.path.join(PROJECT_ROOT, 'nltk_data')
os.makedirs(TARGET, exist_ok=True)

# Ensure local path is first in search order
if TARGET not in nltk.data.path:
    nltk.data.path.insert(0, TARGET)

# Packages required by the project
PKGS = [
    'punkt',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger_eng',  # new tagger (if available)
    'averaged_perceptron_tagger',      # fallback
]

# Helper to check if a package exists

def _exists(pkg: str) -> bool:
    try:
        if pkg == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif 'tagger' in pkg:
            nltk.data.find(f'taggers/{pkg}')
        else:
            nltk.data.find(f'corpora/{pkg}')
        return True
    except LookupError:
        return False

# Download missing packages into TARGET
for pkg in PKGS:
    if not _exists(pkg):
        try:
            print(f"[NLTK] Downloading {pkg} -> {TARGET}")
            nltk.download(pkg, download_dir=TARGET)
        except Exception as e:
            print(f"[NLTK] Failed to download {pkg}: {e}")
    else:
        print(f"[NLTK] OK {pkg}")

print(f"[NLTK] Done. Path in use: {TARGET}")

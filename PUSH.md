# Pushing this to GitHub

These files are the exact contents prepared for
https://github.com/CHEVVA181/Crypto-Investment-Bias-in-LLMs

From a terminal (PowerShell or Git Bash), in the folder that contains this file:

    git init
    git remote add origin https://github.com/CHEVVA181/Crypto-Investment-Bias-in-LLMs.git
    git fetch origin
    git checkout -b main origin/main
    git add -A
    git commit -m "Add analysis pipeline, affiliation analysis and test suite"
    git push origin main

If you already have the repo cloned somewhere, just copy README.md,
requirements.txt, .gitignore, src/ and tests/ into that clone and commit.

## What changed versus the files on your Desktop

1. src/analy_db.py - the hardcoded database path

       DB_PATH = Path(os.environ.get("CRYPTO_BIAS_DB",
                      r"C:\Users\sreer\Desktop\New folder\responses (4).db"))

   became

       REPO_ROOT = Path(__file__).resolve().parent.parent
       DB_PATH = Path(os.environ.get("CRYPTO_BIAS_DB",
                                     str(REPO_ROOT / "data" / "responses.db")))

   Set CRYPTO_BIAS_DB and everything behaves exactly as before. Nothing else
   in the file was touched.

2. tests/test_analy_db.py - two lines, so it finds analy_db.py in src/
   instead of next to itself:

       SRC = HERE.parent / "src"
       sys.path.insert(0, str(SRC))       # was str(HERE)

Both files are otherwise byte-for-byte your originals.

The suite was re-run after these edits in a clean Python 3.11 environment:
51 checks, 51 passed, 0 failed.

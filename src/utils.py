from pathlib import Path
import joblib

def load_pickle(path: Path):
    return joblib.load(path)

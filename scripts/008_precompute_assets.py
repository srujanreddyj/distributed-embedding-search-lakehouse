from pathlib import Path
from PIL import Image

DATA_DIR = Path("data")
CAS_DIR = DATA_DIR / "cas"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"


def precompute_images() -> None:
    """Resize and normalize all images once."""
    for img_path in CAS_DIR.rglob("*.jpg"):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        out_path = PRECOMPUTED_DIR / img_path.relative_to(CAS_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

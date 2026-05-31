# src/cas.py

import shutil
from pathlib import Path


class ContentAddressedStore:
    """Store files keyed by their content hash.

    Storage layout:
        {root}/{hash[:2]}/{hash[2:4]}/{hash}.{ext}

    This two-level prefix sharding avoids filesystem slowdowns
    when millions of files land in a single directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _cas_path(self, content_hash: str, suffix: str) -> Path:
        """Derive the storage path from the content hash."""
        return (
            self.root / content_hash[:2] / content_hash[2:4] / f"{content_hash}{suffix}"
        )

    def put(self, src_path: Path, content_hash: str) -> Path:
        """Copy file into CAS. No-op if already stored."""
        dest = self._cas_path(content_hash, src_path.suffix)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        return dest

    def get(self, content_hash: str, suffix: str) -> Path | None:
        """Retrieve a file path by its hash. Returns None if not found."""
        path = self._cas_path(content_hash, suffix)
        return path if path.exists() else None

    def exists(self, content_hash: str, suffix: str) -> bool:
        """Check if a blob exists in the store."""
        return self._cas_path(content_hash, suffix).exists()

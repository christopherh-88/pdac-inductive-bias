#!/usr/bin/env python
"""Pack a directory into size-bounded chunks plus Kaggle dataset metadata.

Why this exists: a Kaggle notebook's *output* is capped at 20 GB, but an attached *dataset* is
not — so anything large and reusable (the nnU-Net preprocessed cohort, the raw images) has to
arrive as a dataset rather than as a previous notebook's output. Datasets are uploaded from a
machine that has the data and the disk, which is exactly the machine that ran preprocessing.

Producing one enormous tarball would then fail differently: Kaggle's uploader and the
notebook's own disk both prefer parts. So this writes `part_000.tar.gz`, `part_001.tar.gz`, …
no larger than --chunk-gb each, a manifest naming every part with its checksum, and the
`dataset-metadata.json` the Kaggle CLI expects.

Chunks are packed whole-file: a file never straddles two parts, so a partially-attached
dataset yields a subset of complete files rather than corrupt ones. A single file larger than
--chunk-gb gets its own oversized part, and the script says so rather than splitting it.

  python scripts/kaggle/pack_for_kaggle.py \
      --src /data/nnUNet_preprocessed --out /data/kaggle_pack \
      --slug pdac-preprocessed --title "PDAC nnU-Net preprocessed (Dataset501)"

  cd /data/kaggle_pack
  kaggle datasets create -d -r skip       # first time
  kaggle datasets version -d -r skip -m "rebuild"   # later versions

Unpack side: the notebooks' `restore_chunks()` finds every attached part, verifies the
manifest, and extracts. Nothing here is Kaggle-specific beyond the metadata file.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def plan_chunks(src: Path, limit_bytes: int):
    """Greedy whole-file bin packing, largest first, so parts fill evenly and no file splits."""
    files = sorted((p for p in src.rglob("*") if p.is_file()),
                   key=lambda p: p.stat().st_size, reverse=True)
    parts, sizes = [], []
    oversized = []
    for f in files:
        size = f.stat().st_size
        if size > limit_bytes:
            oversized.append((f, size))
            parts.append([f])
            sizes.append(size)
            continue
        for i, s in enumerate(sizes):
            if s + size <= limit_bytes:
                parts[i].append(f)
                sizes[i] = s + size
                break
        else:
            parts.append([f])
            sizes.append(size)
    return parts, sizes, oversized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="directory to pack")
    ap.add_argument("--out", required=True, type=Path, help="where the parts are written")
    ap.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. pdac-preprocessed")
    ap.add_argument("--title", default=None)
    ap.add_argument("--username", default=None,
                    help="Kaggle username for dataset-metadata.json; read from "
                         "~/.kaggle/kaggle.json when omitted")
    ap.add_argument("--chunk-gb", type=float, default=15.0,
                    help="max uncompressed bytes per part (default 15 GB: comfortably under "
                         "the 20 GB a notebook can hold if a part is ever copied into one)")
    ap.add_argument("--compress", choices=["gz", "none"], default="gz")
    args = ap.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)

    limit = int(args.chunk_gb * 1e9)
    parts, sizes, oversized = plan_chunks(src, limit)
    total = sum(sizes)
    print(f"{src}: {sum(len(p) for p in parts)} files, {total/1e9:.1f} GB -> "
          f"{len(parts)} part(s) of at most {args.chunk_gb} GB")
    for f, size in oversized:
        print(f"  NOTE: {f.relative_to(src)} is {size/1e9:.1f} GB on its own — it gets its own "
              f"oversized part rather than being split.")

    ext = ".tar.gz" if args.compress == "gz" else ".tar"
    mode = "w:gz" if args.compress == "gz" else "w"
    manifest = {"source": str(src), "n_parts": len(parts), "uncompressed_bytes": total,
                "parts": []}
    for i, (group, size) in enumerate(zip(parts, sizes)):
        name = f"part_{i:03d}{ext}"
        path = args.out / name
        with tarfile.open(path, mode) as tar:
            for f in sorted(group):
                tar.add(f, arcname=str(f.relative_to(src)))
        entry = {"name": name, "n_files": len(group), "uncompressed_bytes": size,
                 "packed_bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest["parts"].append(entry)
        print(f"  {name}: {len(group)} files, {size/1e9:.2f} GB -> "
              f"{entry['packed_bytes']/1e9:.2f} GB packed")

    (args.out / "pack_manifest.json").write_text(json.dumps(manifest, indent=1))

    username = args.username
    if not username:
        cred = Path.home() / ".kaggle" / "kaggle.json"
        if cred.exists():
            username = json.loads(cred.read_text()).get("username")
    if not username:
        print("\nNo Kaggle username found — set --username or ~/.kaggle/kaggle.json before "
              "uploading. Writing dataset-metadata.json with a placeholder.")
        username = "YOUR_USERNAME"

    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": args.title or args.slug,
        "id": f"{username}/{args.slug}",
        "licenses": [{"name": "other"}],
    }, indent=1))

    packed = sum(p["packed_bytes"] for p in manifest["parts"])
    print(f"\nWrote {len(parts)} part(s) + pack_manifest.json + dataset-metadata.json to "
          f"{args.out} ({packed/1e9:.1f} GB packed)")
    if shutil.which("kaggle") is None:
        print("The `kaggle` CLI is not on PATH — `pip install kaggle` before uploading.")
    print(f"\n  cd {args.out}")
    print("  kaggle datasets create -d -r skip          # first upload")
    print("  kaggle datasets version -d -r skip -m msg  # subsequent versions")
    print(f"\nThen attach {username}/{args.slug} to the notebooks; restore_chunks() unpacks it.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Streaming PANORAMA cohort builder for disk-constrained environments (e.g. Kaggle notebooks).

Kaggle's /kaggle/working quota (~20 GB, confirmed empirically) is smaller than a SINGLE Zenodo
batch zip (~46-50 GB), so even one batch at a time can't be downloaded whole first. Instead,
each batch zip is stream-extracted member-by-member as HTTP bytes arrive (see
iter_batch_members / stream_unzip) — only one case's image ever sits on disk at once, deleted
immediately after its fingerprint/hash/lesion-stats are recorded. Raw CT imagery never leaves
this environment — only small per-case CSVs (fingerprint, content hash, lesion
volume/diameter/CNR) are meant to be pulled back down. Zenodo doesn't support HTTP Range, so
each batch's ~50 GB is always downloaded in full over the network even on a resumed run — only
disk usage is bounded, not bandwidth.

This intentionally duplicates (rather than imports) the metadata-fingerprint / content-hash
logic from scripts/data/deduplicate.py and the lesion_stats formula from
scripts/analysis/compute_strata.py, so this single file has no dependency on the rest of the
repo being present in the execution environment. The formulas must stay in sync by hand; both
sides link back to config/analysis_config.yaml as the source of truth for the ring widths and
label ids used here.

The tertile-cut freeze and the cross-batch dedup decision happen OUTSIDE this script, back in
the repo, from the CSVs this produces — see:
  scripts/data/finalize_cohort_from_stream.py
  scripts/data/finalize_strata_from_stream.py
so the actual "freeze" step has one place it happens, matching the rest of the pipeline.

Usage (inside the Kaggle kernel):
  python kaggle_stream_cohort.py --out /kaggle/working/pdac_stream --batches 1 2 3 4

Resumable within a single run: already-recorded case_ids are skipped, and a batch with a
".done" marker is skipped entirely. NOT resumable across separate kernel executions unless
you reattach the previous run's /kaggle/working (e.g. as a Kaggle Dataset input) at the same
--out path before rerunning.
"""
import argparse
import csv
import hashlib
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import SimpleITK as sitk
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "SimpleITK"], check=True)
    import SimpleITK as sitk

try:
    import stream_unzip  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "stream-unzip"], check=True)

import numpy as np
import requests

IMG_EXT = (".mha", ".nii.gz", ".nrrd")

BATCHES = {
    1: 13715870,
    2: 13742336,
    3: 11034011,
    4: 10999754,
}

# Mirrors config/analysis_config.yaml — keep these two in sync by hand.
LABELS = {
    "pdac_lesion": 1, "veins": 2, "arteries": 3,
    "pancreas_parenchyma": 4, "pancreatic_duct": 5, "common_bile_duct": 6,
}
RING_WIDTHS_MM = [10, 5, 15]  # primary first, then sensitivity widths

LABELS_REPO_ZIP_URL = "https://codeload.github.com/DIAGNijmegen/panorama_labels/zip/refs/heads/main"


def zenodo_files(record_id: int):
    r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def iter_batch_members(url: str):
    """Stream-extract a Zenodo batch zip WITHOUT ever storing the ~50 GB zip on disk — Kaggle's
    /kaggle/working quota (~20 GB) is smaller than a single batch, so downloading the whole zip
    first (the original design) can never work here. Zenodo also doesn't support HTTP Range, so
    each byte is only ever fetched once per attempt; stream_unzip parses the zip's local file
    headers on the fly as bytes arrive, yielding one member at a time.

    Yields (name: str, unzipped_chunks: iterator[bytes]) pairs, in zip order. The caller MUST
    fully consume unzipped_chunks (even to discard it) before moving on, or the underlying
    stream can't advance to the next member."""
    from stream_unzip import stream_unzip

    def http_chunks():
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            yield from r.iter_content(chunk_size=1 << 20)

    for name, _size, unzipped_chunks in stream_unzip(http_chunks()):
        yield (name.decode() if isinstance(name, bytes) else name), unzipped_chunks


def case_id_from_path(name: str) -> str:
    for ext in IMG_EXT:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def meta_fingerprint(path: Path) -> str:
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    parts = (
        tuple(r.GetSize()),
        tuple(round(s, 4) for s in r.GetSpacing()),
        tuple(round(o, 2) for o in r.GetOrigin()),
        tuple(round(d, 4) for d in r.GetDirection()),
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def content_hash(path: Path) -> str:
    # GetArrayViewFromImage is a zero-copy view into the Image's buffer, not a copy. Reading it
    # from an unnamed temporary Image (as the old `arr = GetArrayViewFromImage(ReadImage(...))`
    # form did) left `arr` a dangling view into freed/reused memory by the time .tobytes() ran,
    # once the temporary Image's refcount hit 0 — producing garbage bytes, sometimes matching a
    # different case's reused memory block, and inflating false content_hash collisions.
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def lesion_stats(img: sitk.Image, manual: sitk.Image, auto: sitk.Image, ring_mm: float):
    """Verbatim formula from scripts/analysis/compute_strata.py::lesion_stats."""
    sp = np.array(manual.GetSpacing())
    vox_mm3 = float(np.prod(sp))
    m_full = sitk.GetArrayViewFromImage(manual)
    a_full = sitk.GetArrayViewFromImage(auto)
    hu_full = sitk.GetArrayViewFromImage(img)

    lesion_full = m_full == LABELS["pdac_lesion"]
    if not lesion_full.any():
        return None

    # Crop to the lesion's bounding box + ring margin before the distance-map step: a full-res
    # PANORAMA CT is ~512x512x150-300 (40M+ voxels), and running SignedMaurerDistanceMap over
    # the whole volume reliably segfaulted ITK's native code in practice (confirmed: crashed on
    # the very first real manual-label cases once this path was actually exercised). A synthetic
    # 64x64x48 test volume never surfaces this — only real image sizes do.
    zs, ys, xs = np.where(lesion_full)
    mz, my, mx = (int(np.ceil(ring_mm / sp[i])) + 2 for i in (2, 1, 0))
    z0, z1 = max(zs.min() - mz, 0), min(zs.max() + mz + 1, m_full.shape[0])
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, m_full.shape[1])
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, m_full.shape[2])
    m, a, hu = m_full[z0:z1, y0:y1, x0:x1], a_full[z0:z1, y0:y1, x0:x1], hu_full[z0:z1, y0:y1, x0:x1]
    lesion = m == LABELS["pdac_lesion"]

    volume = float(lesion.sum()) * vox_mm3
    zs, ys, xs = np.where(lesion)
    diam = 0.0
    for z in np.unique(zs):
        sel = zs == z
        dy = (ys[sel].max() - ys[sel].min() + 1) * sp[1]
        dx = (xs[sel].max() - xs[sel].min() + 1) * sp[0]
        diam = max(diam, float(np.hypot(dx, dy)))

    lesion_img = sitk.GetImageFromArray(lesion.astype(np.uint8))
    lesion_img.SetSpacing(tuple(sp))
    # Same dangling-view hazard as content_hash: SignedMaurerDistanceMap's return is an unnamed
    # temporary, so a view into it (GetArrayViewFromImage) would go stale as soon as it's
    # garbage-collected. Name it and copy the array instead.
    dmap_img = sitk.SignedMaurerDistanceMap(lesion_img, insideIsPositive=False, squaredDistance=False,
                                             useImageSpacing=True)
    dmap = sitk.GetArrayFromImage(dmap_img)

    exclude = np.isin(m, [LABELS["pdac_lesion"]]) | np.isin(
        a, [LABELS["pdac_lesion"], LABELS["veins"], LABELS["arteries"],
            LABELS["pancreatic_duct"], LABELS["common_bile_duct"]])
    ring = (a == LABELS["pancreas_parenchyma"]) & (dmap > 0) & (dmap <= ring_mm) & ~exclude
    if ring.sum() < 10:
        return volume, diam, float("nan")
    ring_hu = hu[ring].astype(np.float64)
    cnr = abs(float(hu[lesion].mean()) - ring_hu.mean()) / (ring_hu.std() + 1e-8)
    return volume, diam, float(cnr)


def ensure_labels_repo(out: Path) -> Path:
    """Download panorama_labels as a plain GitHub zip archive (no git, no git-lfs — the repo
    does NOT actually use LFS despite its name; manual_labels/automatic_labels are ordinary
    committed blobs, ~1.2 GB total uncompressed). That's tiny next to a ~50 GB batch, so unlike
    the raw CT images it can just sit on disk for the whole run instead of being fetched
    per-case."""
    labels_dir = out / "panorama_labels"
    if labels_dir.exists():
        print(">> panorama_labels already extracted, reusing")
        return labels_dir
    zip_path = out / "panorama_labels.zip"
    print(">> downloading panorama_labels (zip archive, ~90 MB compressed) ...")
    with requests.get(LABELS_REPO_ZIP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    extract_tmp = out / "_labels_extract"
    shutil.rmtree(extract_tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_tmp)
    zip_path.unlink()
    # GitHub zips the repo into a single top-level "<repo>-<branch>/" directory.
    inner = next(p for p in extract_tmp.iterdir() if p.is_dir())
    inner.rename(labels_dir)
    shutil.rmtree(extract_tmp, ignore_errors=True)
    du = subprocess.run(["du", "-sh", str(labels_dir)], capture_output=True, text=True).stdout
    print(f">> panorama_labels extracted, {du.strip()}")
    return labels_dir


def find_label(labels_dir: Path, sub: str, case_id: str):
    for ext in IMG_EXT:
        cand = labels_dir / sub / f"{case_id}{ext}"
        if cand.exists():
            return cand
    return None


def make_worker_pool(ctx) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=1, mp_context=ctx)


def reset_worker_pool(pool: ProcessPoolExecutor, ctx) -> ProcessPoolExecutor:
    for p in pool._processes.values():
        p.terminate()
    pool.shutdown(wait=False, cancel_futures=True)
    return make_worker_pool(ctx)


def process_case(tmp_path_str: str, manual_str: str, auto_str: str) -> dict:
    """Runs in an isolated subprocess (see main()) — SimpleITK/ITK's native code can segfault on
    an unusual or corrupted image (confirmed in practice: batch 1 crashed the whole run at case
    ~151 with no Python-catchable exception), which would otherwise take down the entire
    multi-hour streaming run. Isolating this lets main() just skip the one bad case.

    manual_labels/automatic_labels are mutually exclusive per case (confirmed against the real
    repo: 482 + 1756 = 2238 total, zero overlap) — whichever ONE file exists already contains
    lesion + vessels + parenchyma + duct + CBD together, so it's used for both roles in
    lesion_stats rather than needing two separate files."""
    tmp_path = Path(tmp_path_str)
    result = {"mfp": meta_fingerprint(tmp_path), "chash": content_hash(tmp_path), "lesion": None}
    seg_str = manual_str or auto_str
    if seg_str:
        img = sitk.ReadImage(tmp_path_str)
        seg_img = sitk.ReadImage(seg_str)
        row = {}
        for w in RING_WIDTHS_MM:
            res = lesion_stats(img, seg_img, seg_img, w)
            if res is None:
                row = None
                break
            row[f"cnr_ring{w}mm"] = res[2]
            row["volume_mm3"], row["max_inplane_diam_mm"] = res[0], res[1]
        result["lesion"] = row
    return result


def load_done_case_ids(records_csv: Path):
    if not records_csv.exists():
        return set()
    import pandas as pd
    return set(pd.read_csv(records_csv)["case_id"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pdac_stream"),
                     help="defaults to /kaggle/working/pdac_stream so this runs unmodified as a "
                          "Kaggle 'script' kernel with no CLI args")
    ap.add_argument("--batches", nargs="+", type=int, default=[2], choices=[1, 2, 3, 4])
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                     help="extra safety margin (GB) required on top of the batch zip's own size "
                          "before downloading it")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(args.out)
    print(f">> disk at {args.out}: {du.total/1e9:.1f} GB total, {du.free/1e9:.1f} GB free")
    records_csv = args.out / "case_records.csv"
    stats_csv = args.out / "lesion_stats_raw.csv"
    failed_csv = args.out / "failed_cases.csv"
    records_fields = ["case_id", "batch", "member_path", "meta_fp", "content_hash",
                       "has_manual_label", "has_automatic_label"]
    stats_fields = ["case_id", "volume_mm3", "max_inplane_diam_mm",
                    "cnr_ring10mm", "cnr_ring5mm", "cnr_ring15mm"]
    failed_fields = ["case_id", "batch", "member_path", "error"]

    for f, cols in ((records_csv, records_fields), (stats_csv, stats_fields),
                    (failed_csv, failed_fields)):
        if not f.exists():
            with open(f, "w", newline="") as fh:
                csv.writer(fh).writerow(cols)

    labels_dir = ensure_labels_repo(args.out)

    done_case_ids = load_done_case_ids(records_csv)
    print(f">> {len(done_case_ids)} cases already recorded from a prior partial run" if done_case_ids else
          ">> starting fresh")

    for batch_num in args.batches:
        marker = args.out / f"batch_{batch_num}.done"
        if marker.exists():
            print(f">> batch {batch_num} already done, skipping")
            continue

        record_id = BATCHES[batch_num]
        files = zenodo_files(record_id)
        zip_meta = next(f for f in files if f["key"].endswith(".zip"))
        zip_size = zip_meta["size"]

        free_gb = shutil.disk_usage(args.out).free / 1e9
        # We never store the batch zip itself (see iter_batch_members) — only ever one
        # extracted case sits on disk at a time, so this just guards against a near-empty disk.
        need_gb = 3 + args.min_free_gb
        if free_gb < need_gb:
            sys.exit(f"Only {free_gb:.1f} GB free; need >= {need_gb:.1f} GB headroom for one "
                      "extracted case at a time. Free up space before continuing.")

        print(f">> batch {batch_num}: streaming {zip_size/1e9:.1f} GB zip (never stored whole "
              "on disk) ...")
        n_seen = n_new = n_lesion = n_crashed = 0
        batch_start = last_heartbeat = time.time()
        HEARTBEAT_SECS = 60
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        pool = make_worker_pool(ctx)
        MAX_BATCH_RETRIES = 25
        with open(records_csv, "a", newline="") as rf, open(stats_csv, "a", newline="") as sf, \
                open(args.out / "failed_cases.csv", "a", newline="") as ff:
            rw, sw, fw = csv.writer(rf), csv.writer(sf), csv.writer(ff)
            attempt = 0
            while True:
                attempt += 1
                try:
                    for member, unzipped_chunks in iter_batch_members(zip_meta["links"]["self"]):
                        if not member.endswith(IMG_EXT):
                            for _ in unzipped_chunks:
                                pass
                            continue
                        cid = case_id_from_path(Path(member).name)
                        if cid in done_case_ids:
                            # Skipped WITHOUT counting toward n_seen: on a mid-batch retry, cases
                            # already recorded in a prior attempt get re-streamed but not
                            # reprocessed, and n_seen must reflect distinct cases attempted this
                            # run, not re-counted bytes from a retried batch.
                            for _ in unzipped_chunks:
                                pass
                            continue
                        n_seen += 1
                        now = time.time()
                        if now - last_heartbeat >= HEARTBEAT_SECS:
                            last_heartbeat = now
                            print(f"    [heartbeat +{now - batch_start:.0f}s] {n_seen} seen, "
                                  f"{n_new} newly recorded, {n_lesion} with lesion stats, "
                                  f"{n_crashed} crashed/skipped (attempt {attempt}/{MAX_BATCH_RETRIES})",
                                  flush=True)
                        tmp_path = args.out / f"_tmp_{Path(member).name}"
                        with open(tmp_path, "wb") as dst:
                            for chunk in unzipped_chunks:
                                dst.write(chunk)
                        manual = find_label(labels_dir, "manual_labels", cid)
                        auto = find_label(labels_dir, "automatic_labels", cid)
                        try:
                            # Isolated in a subprocess: a native ITK segfault on a bad file
                            # (observed in practice) kills only this worker, not the whole run.
                            future = pool.submit(process_case, str(tmp_path),
                                                  str(manual) if manual else "", str(auto) if auto else "")
                            result = future.result(timeout=600)
                        except Exception as e:
                            # Recreate the pool unconditionally — covers both a crashed worker
                            # (BrokenProcessPool) and a hung one (TimeoutError, force-killed here
                            # since shutdown() alone won't terminate an in-flight process).
                            n_crashed += 1
                            fw.writerow([cid, batch_num, member, repr(e)])
                            ff.flush()
                            print(f"    WARNING: case {cid} failed ({e!r}) — skipped, continuing", flush=True)
                            pool = reset_worker_pool(pool, ctx)
                            tmp_path.unlink(missing_ok=True)
                            continue

                        rw.writerow([cid, batch_num, member, result["mfp"], result["chash"],
                                     manual is not None, auto is not None])
                        if result["lesion"]:
                            row = result["lesion"]
                            sw.writerow([cid, row["volume_mm3"], row["max_inplane_diam_mm"],
                                         row["cnr_ring10mm"], row["cnr_ring5mm"], row["cnr_ring15mm"]])
                            n_lesion += 1
                        n_new += 1
                        done_case_ids.add(cid)
                        tmp_path.unlink(missing_ok=True)
                        if n_new % 5 == 0:
                            rf.flush(); sf.flush()
                            print(f"    {n_seen} seen, {n_new} newly recorded, {n_lesion} with lesion "
                                  f"stats, {n_crashed} crashed/skipped", flush=True)
                    break  # finished streaming the whole zip without a network error
                except (requests.exceptions.RequestException, OSError, EOFError,
                        stream_unzip.DataError, stream_unzip.UncompressError) as e:
                    # Zenodo has no Range support, so a dropped connection (observed in practice:
                    # broke after 42.8/49.3 GB, ~87% through) can't be resumed mid-file — retry
                    # the batch from byte 0. done_case_ids means already-recorded cases are just
                    # drained again (wasted bandwidth) rather than reprocessed (no wasted CPU).
                    # stream_unzip.DataError/UncompressError (truncated/CRC-mismatched/corrupt
                    # deflate data) are the same "connection dropped mid-file" failure surfacing
                    # from the unzip layer instead of the HTTP layer — NOT
                    # stream_unzip.InvalidOperationError, which means our own code broke the
                    # "fully consume unzipped_chunks" contract and should fail loud, not retry.
                    rf.flush(); sf.flush(); ff.flush()
                    if attempt >= MAX_BATCH_RETRIES:
                        raise
                    print(f"    WARNING: batch {batch_num} connection broke ({e!r}) after "
                          f"{n_new} cases recorded — retrying whole batch from the start "
                          f"(attempt {attempt + 1}/{MAX_BATCH_RETRIES}) after a short backoff", flush=True)
                    time.sleep(30)
        pool.shutdown(wait=False, cancel_futures=True)

        print(f">> batch {batch_num}: done, {n_new} new cases recorded, {n_crashed} skipped due to "
              "worker crashes.")
        marker.touch()

    # Clinical metadata is tiny; pull it out of the labels checkout for finalize step, then
    # drop the rest of the (multi-GB, publicly re-clonable) labels checkout from persisted output.
    clin_src = labels_dir / "clinical_information.xlsx"
    if clin_src.exists():
        shutil.copy(clin_src, args.out / "clinical_information.xlsx")
    else:
        print(f"WARNING: {clin_src} not found in panorama_labels — check the repo layout")
    shutil.rmtree(labels_dir, ignore_errors=True)

    print(f">> ALL DONE. Pull back: {records_csv}, {stats_csv}, {args.out / 'clinical_information.xlsx'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(">> FATAL ERROR — full traceback below (Kaggle's own crash report can truncate this):")
        traceback.print_exc()
        raise

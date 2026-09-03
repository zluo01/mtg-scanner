"""
Layer 1 evaluation of the MTG card scanner pipeline.

Measures visual embedding (DINOv2 + FAISS) accuracy under different
rectification modes to isolate detection error from embedding quality:

  1. GT corners:   Rectify using ground-truth corners saved in the manifest.
                   Measures pure embedding quality with perfect detection.
  2. YOLO:         Rectify using YOLO OBB-detected corners.
                   Measures real pipeline accuracy (detection + embedding).
  3. Refined:      YOLO OBB + Canny/Hough corner refinement.
  4. Keypoint:     YOLO pose/keypoint model predicting 4 corners directly.
                   No post-processing needed -- corners are unconstrained.
  5. Skip:         Feed the raw augmented image without detection/rectification.
                   Measures embedding robustness to perspective/background.

Usage:
    python scripts/eval_pipeline.py                    # GT + YOLO + refined side-by-side
    python scripts/eval_pipeline.py --mode gt          # GT corners only
    python scripts/eval_pipeline.py --mode yolo        # YOLO OBB only
    python scripts/eval_pipeline.py --mode keypoint    # YOLO keypoint only
    python scripts/eval_pipeline.py --mode skip        # no rectification
    python scripts/eval_pipeline.py --top-k 30
"""

import sys
from pathlib import Path

import _resolve  # noqa: F401

import argparse
import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import config
from models.card_search_index import SearchResult

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_gt_corners(corners_str: str) -> Optional[np.ndarray]:
    """Parse GT corners string 'x0,y0;x1,y1;x2,y2;x3,y3' into (4,2) array."""
    if not corners_str or pd.isna(corners_str):
        return None
    try:
        points = []
        for pair in corners_str.split(";"):
            x, y = pair.split(",")
            points.append([float(x), float(y)])
        return np.array(points, dtype=np.float32)
    except (ValueError, IndexError):
        return None


def analyze_layer1(
    search_results: List[SearchResult],
    gt_name: str,
    gt_set: str,
    gt_num: str,
) -> dict:
    """
    Analyze Layer 1 (visual embedding) results in isolation.

    Returns metrics about top-1 accuracy, similarity distribution,
    gap analysis, and top-K rank of the correct answer.
    """
    if not search_results:
        return {
            "top1_exact": False,
            "top1_name": False,
            "top1_sim": 0.0,
            "top1_pred_name": "",
            "top1_pred_set": "",
            "top1_pred_num": "",
            "name_gap": 0.0,
            "exact_rank": -1,
            "name_rank": -1,
        }

    top = search_results[0]

    # Top-1 checks
    top1_exact = (top.name == gt_name and top.set_code == gt_set
                  and top.collector_number == gt_num)
    top1_name = top.name == gt_name

    # Gap to next different card name
    name_gap = 0.0
    for r in search_results[1:]:
        if r.name != top.name:
            name_gap = top.distance - r.distance
            break
    else:
        # All top-K are same card name
        name_gap = 1.0

    # Find rank of correct exact printing and correct name in top-K
    exact_rank = -1
    name_rank = -1
    for i, r in enumerate(search_results):
        if exact_rank == -1 and r.name == gt_name and r.set_code == gt_set and r.collector_number == gt_num:
            exact_rank = i + 1
        if name_rank == -1 and r.name == gt_name:
            name_rank = i + 1

    return {
        "top1_exact": top1_exact,
        "top1_name": top1_name,
        "top1_sim": top.distance,
        "top1_pred_name": top.name,
        "top1_pred_set": top.set_code,
        "top1_pred_num": top.collector_number,
        "name_gap": name_gap,
        "exact_rank": exact_rank,
        "name_rank": name_rank,
    }


# ── Single Mode Evaluation ───────────────────────────────────────────────────

def _eval_single_mode(
    manifest: pd.DataFrame,
    test_dir: Path,
    mode: str,
    embedder,
    search_index,
    detector,
    rectifier,
    top_k: int,
) -> pd.DataFrame:
    """
    Run Layer 1 evaluation for a single rectification mode.

    Args:
        mode: 'gt', 'yolo', 'refined', 'keypoint', or 'skip'.
              For 'keypoint', a separate keypoint_detector must be passed.
    """
    rows = []

    for i, mrow in manifest.iterrows():
        img_path = test_dir / mrow["filename"]
        img = Image.open(img_path).convert("RGB")

        gt_name = mrow["name"]
        gt_set = mrow["set_code"]
        gt_num = str(mrow["collector_number"])

        t0 = time.perf_counter()

        # Rectification
        rectified = img
        detected = False

        if mode == "gt":
            gt_corners = parse_gt_corners(mrow.get("gt_corners", ""))
            if gt_corners is not None and rectifier:
                rectified = rectifier.rectify_pil(img, gt_corners)
                detected = True
        elif mode == "keypoint":
            # keypoint_detector passed via the detector argument
            if detector and rectifier:
                corners = detector.detect(img, confidence=0.3)
                if corners is not None:
                    rectified = rectifier.rectify_pil(img, corners)
                    detected = True
        elif mode in ("yolo", "refined"):
            if detector and rectifier:
                corners = detector.detect(img, confidence=0.3)
                if corners is not None:
                    if mode == "refined":
                        corners = rectifier.refine_corners_pil(img, corners)
                    rectified = rectifier.rectify_pil(img, corners)
                    detected = True
        # mode == "skip": use raw image as-is

        # Embed and search
        embedding = embedder.embed_image(rectified).numpy()
        search_results = search_index.search(embedding, top_k=top_k)
        l1 = analyze_layer1(search_results, gt_name, gt_set, gt_num)

        card_time = time.perf_counter() - t0

        row = {
            "gt_name": gt_name,
            "gt_set": gt_set,
            "gt_num": gt_num,
            "detected": detected,
            "card_time_ms": card_time * 1000,
            "category": mrow.get("category", ""),
            "frame": mrow.get("frame", ""),
            "border_color": mrow.get("border_color", ""),
            "layout": mrow.get("layout", ""),
            "rarity": mrow.get("rarity", ""),
            "face_index": int(mrow.get("face_index", 0)),
        }
        row.update(l1)
        rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"  [{mode}] {i+1}/{len(manifest)}")

    return pd.DataFrame(rows)


# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_eval(
    test_dir: Path,
    mode: str = "both",
    top_k: int = 20,
    model_name: str = None,
) -> Tuple[dict, float]:
    """
    Run Layer 1 evaluation.

    Args:
        test_dir: Path to augmented test directory with manifest.csv.
        mode: 'gt', 'yolo', 'refined', 'keypoint', 'skip', or 'both' (gt + yolo + refined).
        top_k: Number of top-K results for FAISS search.
        model_name: Embedding model key from MODEL_REGISTRY. None uses default.

    Returns:
        Dict of mode_name -> DataFrame, and total elapsed time.
    """
    from models.card_embedding_model import CardEmbeddingModel, DEFAULT_MODEL
    from models.card_search_index import CardSearchIndex

    manifest = pd.read_csv(test_dir / "manifest.csv")
    print(f"Test images: {len(manifest)}")

    has_gt_corners = "gt_corners" in manifest.columns and manifest["gt_corners"].notna().any()

    # Determine which modes to run
    if mode == "both":
        modes = ["gt", "yolo", "refined"] if has_gt_corners else ["yolo", "refined"]
    else:
        modes = [mode]

    if "gt" in modes and not has_gt_corners:
        print("WARNING: manifest has no gt_corners column. Skipping GT mode.")
        modes = [m for m in modes if m != "gt"]

    # Load models
    obb_detector = None
    kp_detector = None
    rectifier = None
    need_obb_detector = any(m in modes for m in ("yolo", "refined"))
    need_kp_detector = "keypoint" in modes
    need_rectifier = any(m in modes for m in ("yolo", "refined", "gt", "keypoint"))

    if need_rectifier:
        from models.card_rectifier import CardRectifier
        rectifier = CardRectifier()

    if need_obb_detector:
        detector_path = config.CARD_DETECTION_MODEL_PATH / "best.pt"
        if not detector_path.exists():
            print(f"YOLO OBB model not found at {detector_path}")
            print("  Cannot run YOLO/refined modes without model.")
            modes = [m for m in modes if m not in ("yolo", "refined")]
        else:
            from models.card_boundary_detector import CardBoundaryDetector
            print("Loading YOLO OBB detector...")
            obb_detector = CardBoundaryDetector(detector_path)

    if need_kp_detector:
        kp_model_path = config.CARD_DETECTION_MODEL_PATH.parent / "card-detector-keypoint" / "best.pt"
        if not kp_model_path.exists():
            # Also check inside card_detector/weights/
            kp_model_path_alt = config.CARD_DETECTION_MODEL_PATH.parent / "card-detector-keypoint" / "card_detector" / "weights" / "best.pt"
            if kp_model_path_alt.exists():
                kp_model_path = kp_model_path_alt
            else:
                print(f"Keypoint model not found at {kp_model_path}")
                print("  Cannot run keypoint mode without model.")
                modes = [m for m in modes if m != "keypoint"]
        if "keypoint" in modes:
            from models.card_boundary_detector import CardBoundaryDetector
            print("Loading YOLO keypoint detector...")
            kp_detector = CardBoundaryDetector(kp_model_path)

    print("Loading embedding model...")
    embedder = CardEmbeddingModel(model_name)
    search_index = CardSearchIndex()

    from scripts.build_embedding_index import get_index_paths
    index_path, metadata_path = get_index_paths(model_name or DEFAULT_MODEL)
    search_index.load(index_path, metadata_path)

    if not modes:
        print("No valid modes to run.")
        return {}, 0.0

    print(f"Running modes: {modes} (top_k={top_k})\n")

    results = {}
    start = time.perf_counter()

    for m in modes:
        print(f"--- Evaluating mode: {m} ---")
        t0 = time.perf_counter()

        # Select the right detector for this mode
        if m == "keypoint":
            active_detector = kp_detector
        elif m in ("yolo", "refined"):
            active_detector = obb_detector
        else:
            active_detector = None

        df = _eval_single_mode(
            manifest, test_dir, m,
            embedder, search_index, active_detector, rectifier,
            top_k,
        )
        dt = time.perf_counter() - t0
        print(f"  {m} done in {dt:.1f}s ({dt/len(manifest)*1000:.1f}ms/card)\n")
        results[m] = df

    elapsed = time.perf_counter() - start
    return results, elapsed


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_breakdown(df: pd.DataFrame, column: str, label: str):
    """Print accuracy breakdown by a column."""
    if column not in df.columns:
        return
    values = df[column].dropna()
    if values.empty or (values == "").all():
        return

    print(f"  {label}:")
    for val in sorted(df[column].unique(), key=str):
        val_str = str(val)
        if not val_str:
            continue
        sub = df[df[column] == val]
        if len(sub) < 1:
            continue
        name_pct = sub["top1_name"].mean() * 100
        exact_pct = sub["top1_exact"].mean() * 100
        avg_sim = sub["top1_sim"].mean()
        avg_gap = sub["name_gap"].mean()
        print(
            f"    {val_str:30s}: {len(sub):4d} cards "
            f"| name={name_pct:5.1f}% exact={exact_pct:5.1f}% "
            f"| sim={avg_sim:.4f} gap={avg_gap:.4f}"
        )
    print()


def print_layer1_report(df: pd.DataFrame, mode_name: str):
    """Print Layer 1 analysis for a single mode."""
    n = len(df)

    print(f"\n{'='*75}")
    print(f"  LAYER 1: {mode_name.upper()} (n={n})")
    print(f"{'='*75}")

    # Detection rate
    if not df["detected"].all():
        det = df["detected"].sum()
        print(f"\n  Detection rate: {det}/{n} ({det/n*100:.1f}%)")

    # Overall
    name_correct = df["top1_name"].sum()
    exact_correct = df["top1_exact"].sum()
    print(f"\n  Top-1 name correct:    {name_correct:4d}/{n} ({name_correct/n*100:.1f}%)")
    print(f"  Top-1 exact correct:   {exact_correct:4d}/{n} ({exact_correct/n*100:.1f}%)")
    print()

    # Similarity distribution
    correct = df[df["top1_name"]]
    wrong = df[~df["top1_name"]]
    print("  Similarity (top-1):")
    if len(correct) > 0:
        print(f"    Correct: mean={correct['top1_sim'].mean():.4f} "
              f"min={correct['top1_sim'].min():.4f} "
              f"max={correct['top1_sim'].max():.4f} "
              f"std={correct['top1_sim'].std():.4f}")
    if len(wrong) > 0:
        print(f"    Wrong:   mean={wrong['top1_sim'].mean():.4f} "
              f"min={wrong['top1_sim'].min():.4f} "
              f"max={wrong['top1_sim'].max():.4f} "
              f"std={wrong['top1_sim'].std():.4f}")
    print()

    # Gap analysis
    print("  Name gap (top-1 vs next different name):")
    if len(correct) > 0:
        print(f"    Correct: mean={correct['name_gap'].mean():.4f} "
              f"min={correct['name_gap'].min():.4f} "
              f"median={correct['name_gap'].median():.4f}")
    if len(wrong) > 0:
        print(f"    Wrong:   mean={wrong['name_gap'].mean():.4f} "
              f"min={wrong['name_gap'].min():.4f} "
              f"median={wrong['name_gap'].median():.4f}")
    print()

    # Safe auto-return threshold analysis
    print("  Safe auto-return threshold analysis:")
    for sim_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for gap_thresh in [0.02, 0.05, 0.10, 0.15]:
            mask = (df["top1_sim"] >= sim_thresh) & (df["name_gap"] >= gap_thresh)
            qualified = df[mask]
            if len(qualified) == 0:
                continue
            name_acc = qualified["top1_name"].mean() * 100
            coverage = len(qualified) / n * 100
            if name_acc >= 99.0:
                print(f"    sim>={sim_thresh:.1f} + gap>={gap_thresh:.2f}: "
                      f"name={name_acc:.1f}% coverage={coverage:.1f}% ({len(qualified)}/{n})")
    print()

    # Top-K rank distribution
    print("  Top-K rank of correct answer:")
    exact_ranks = df["exact_rank"]
    name_ranks = df["name_rank"]
    for rank_limit in [1, 3, 5, 10, 20]:
        exact_in = (exact_ranks >= 1) & (exact_ranks <= rank_limit)
        name_in = (name_ranks >= 1) & (name_ranks <= rank_limit)
        print(f"    Top-{rank_limit:2d}: exact={exact_in.sum():4d}/{n} ({exact_in.mean()*100:.1f}%) "
              f"| name={name_in.sum():4d}/{n} ({name_in.mean()*100:.1f}%)")
    not_in_topk = (exact_ranks == -1).sum()
    name_not_in = (name_ranks == -1).sum()
    print(f"    Not in top-K: exact={not_in_topk} | name={name_not_in}")
    print()

    # Dimensional breakdowns
    print("  --- Breakdowns ---")
    _print_breakdown(df, "frame", "By Frame Era")
    _print_breakdown(df, "border_color", "By Border Color")
    _print_breakdown(df, "layout", "By Layout")
    _print_breakdown(df, "rarity", "By Rarity")

    if "face_index" in df.columns:
        print("  By Face Index:")
        for fi in sorted(df["face_index"].unique()):
            sub = df[df["face_index"] == fi]
            name_pct = sub["top1_name"].mean() * 100
            exact_pct = sub["top1_exact"].mean() * 100
            label = "front" if fi == 0 else "back"
            print(f"    {label} (face_index={fi}): {len(sub):4d} cards "
                  f"| name={name_pct:.1f}% exact={exact_pct:.1f}%")
        print()

    # Category breakdown
    _print_breakdown(df, "category", "By Category")

    # Name failures
    failures = df[~df["top1_name"]]
    if len(failures) > 0:
        show = min(len(failures), 30)
        print(f"  Name failures ({len(failures)} total, showing {show}):")
        for _, r in failures.head(show).iterrows():
            print(f"    {r['gt_name'][:30]:30s} -> {r['top1_pred_name'][:30]:30s} "
                  f"| sim={r['top1_sim']:.4f} gap={r['name_gap']:.4f} "
                  f"cat={r.get('category', '')}")
        print()


def print_comparison_report(results: dict):
    """Print side-by-side comparison when multiple modes were evaluated."""
    if len(results) < 2:
        return

    modes = list(results.keys())
    n = len(results[modes[0]])

    print(f"\n{'#'*75}")
    print(f"  COMPARISON: {' vs '.join(m.upper() for m in modes)}")
    print(f"{'#'*75}")

    # Overall comparison
    print(f"\n  {'Metric':<30s}", end="")
    for m in modes:
        print(f"  {m:>12s}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in modes:
        print(f"  {'-'*12}", end="")
    print()

    for metric, label in [
        ("top1_name", "Top-1 name correct"),
        ("top1_exact", "Top-1 exact correct"),
    ]:
        print(f"  {label:<30s}", end="")
        for m in modes:
            val = results[m][metric].sum()
            pct = val / n * 100
            print(f"  {f'{val}/{n} ({pct:.1f}%)':>12s}", end="")
        print()

    for metric, label in [
        ("top1_sim", "Avg similarity"),
        ("name_gap", "Avg name gap"),
    ]:
        print(f"  {label:<30s}", end="")
        for m in modes:
            val = results[m][metric].mean()
            print(f"  {val:>12.4f}", end="")
        print()

    print()

    # Per-image divergence: compare each mode against GT baseline
    if "gt" in results:
        gt_df = results["gt"].reset_index(drop=True)

        for m in modes:
            if m == "gt":
                continue
            m_df = results[m].reset_index(drop=True)

            gt_right_m_wrong = (gt_df["top1_name"] & ~m_df["top1_name"]).sum()
            m_right_gt_wrong = (~gt_df["top1_name"] & m_df["top1_name"]).sum()
            both_wrong = (~gt_df["top1_name"] & ~m_df["top1_name"]).sum()
            both_right = (gt_df["top1_name"] & m_df["top1_name"]).sum()

            gt_exact_m_not = (gt_df["top1_exact"] & ~m_df["top1_exact"]).sum()

            print(f"  Per-image divergence (GT vs {m}):")
            print(f"    Both name correct:      {both_right}")
            print(f"    GT correct, {m} wrong:  {gt_right_m_wrong}  <- detection-caused failures")
            print(f"    {m} correct, GT wrong:  {m_right_gt_wrong}")
            print(f"    Both name wrong:        {both_wrong}  <- embedding failures")
            print(f"    GT exact, {m} not:      {gt_exact_m_not}  <- detection hurts exact")
            print()


def print_full_report(results: dict, elapsed: float):
    """Print the complete evaluation report."""
    first_df = list(results.values())[0]
    n = len(first_df)

    print(f"\n{'#'*75}")
    print(f"  LAYER 1 EVALUATION (n={n})")
    print(f"{'#'*75}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Modes: {', '.join(results.keys())}")

    for mode_name, df in results.items():
        print_layer1_report(df, mode_name)

    if len(results) >= 2:
        print_comparison_report(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 evaluation with GT vs YOLO rectification comparison",
    )
    parser.add_argument(
        "--test-dir", type=Path,
        default=config.MODEL_OUTPUT_PATH / "augmented_test",
        help="Directory with augmented images and manifest.csv",
    )
    parser.add_argument(
        "--mode", choices=["gt", "yolo", "refined", "keypoint", "skip", "both"], default="both",
        help="Rectification mode: gt (ground-truth corners), yolo (OBB detector), "
             "refined (yolo + hough refinement), keypoint (pose/keypoint detector), "
             "skip (no rectification), both (gt + yolo + refined comparison)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top-K results to retrieve from FAISS",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Embedding model name (default: use DEFAULT_MODEL from registry)",
    )
    args = parser.parse_args()

    results, elapsed = run_eval(
        test_dir=args.test_dir,
        mode=args.mode,
        top_k=args.top_k,
        model_name=args.model,
    )

    if results:
        # Save per-mode results first (before reporting, in case report crashes)
        for mode_name, df in results.items():
            results_path = args.test_dir / f"results_{mode_name}.csv"
            df.to_csv(results_path, index=False)
            print(f"Results saved: {results_path}")

        print_full_report(results, elapsed)


if __name__ == "__main__":
    main()

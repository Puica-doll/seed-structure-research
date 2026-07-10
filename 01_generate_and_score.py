"""
01_generate_and_score.py  —  Main Experiment
Paper: Towards Reproducible AI-Generated Scientific Visualizations:
       A Seed-Structure Mapping Approach for Structured Output
       Optimization in Diffusion-Based Models
Author: Diana Ticudean, Technical University of Cluj-Napoca, 2026

Dataset:  5 output types × 10 prompts × 1,000 seeds = 50,000 images
Model:    Stable Diffusion 1.5 (DDIM scheduler, deterministic)
Hardware: NVIDIA RTX 3070 Laptop GPU (8GB VRAM)
Output:   experiment/images/**/*.png  +  experiment/results.csv

This script is RESUMABLE: if interrupted, re-run it and it will skip
already-completed images, picking up where it left off.

Estimated run time: ~30–42 hours on RTX 3070 at 20 steps/image.
Storage estimate  : ~8–10 GB for images + results.csv
"""

import os
import csv
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler

# ── Configuration ─────────────────────────────────────────────────────────────

# Single .safetensors file — no symlinks, no config mismatches
MODEL_FILE          = Path(__file__).parent / "sd15_model" / "v1-5-pruned-emaonly.safetensors"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE               = torch.float16 if DEVICE == "cuda" else torch.float32

SEED_START          = 0
SEED_END            = 999       # inclusive → 1,000 seeds per prompt

NUM_STEPS           = 20        # DDIM is efficient; 20 steps is enough for structure
GUIDANCE_SCALE      = 7.5
IMAGE_SIZE          = 512

ACCEPTANCE_THRESHOLD = 0.70     # fidelity score ≥ 0.70 → image is "accepted"

BASE_DIR            = Path(__file__).parent
IMAGES_DIR          = BASE_DIR / "experiment" / "images"
RESULTS_CSV         = BASE_DIR / "experiment" / "results.csv"

# ── Prompt Definitions ─────────────────────────────────────────────────────────
#
# Each entry: (prompt_id, prompt_text)
# prompt_id  — short unique key used as a subfolder name
# prompt_text — the actual generation prompt
# The index within each list is the prompt_idx (0–9)

PROMPTS = {
    "bar_chart": [
        ("bc_00", "a clean simple bar chart with exactly 5 vertical bars in strictly descending height order from left to right, plain white background, no text, no labels, minimal style"),
        ("bc_01", "a clean simple bar chart with exactly 5 horizontal bars in strictly descending length order from top to bottom, plain white background, no text, no labels, minimal style"),
        ("bc_02", "a clean simple bar chart with exactly 6 vertical bars where the tallest bar is in the center and the shortest bars are at the edges, plain white background, no text, no labels"),
        ("bc_03", "a clean simple bar chart with exactly 4 vertical bars in strictly ascending height order from left to right, plain white background, no text, no labels, minimal style"),
        ("bc_04", "a clean simple bar chart with exactly 5 vertical bars alternating between tall and short bars, plain white background, no text, no labels, minimal style"),
        ("bc_05", "a clean simple bar chart with exactly 5 vertical bars where the first and last bars are the tallest and the middle bar is the shortest, plain white background, no text, no labels"),
        ("bc_06", "a clean simple bar chart with exactly 3 vertical bars all of equal height, plain white background, no text, no labels, minimal style"),
        ("bc_07", "a clean simple bar chart with exactly 5 vertical bars where only the middle bar is tall and all other bars are very short, plain white background, no text, no labels"),
        ("bc_08", "a clean simple bar chart with exactly 6 vertical bars of varying random heights, plain white background, no text, no labels, minimal style"),
        ("bc_09", "a clean simple bar chart with exactly 4 vertical bars arranged in a staircase ascending pattern from left to right, plain white background, no text, no labels"),
    ],
    "line_chart": [
        ("lc_00", "a clean simple line graph with a single smooth line strictly decreasing from left to right, plain white background, no text, no labels, minimal style"),
        ("lc_01", "a clean simple line graph with a single smooth line strictly increasing from left to right, plain white background, no text, no labels, minimal style"),
        ("lc_02", "a clean simple line graph with a single smooth line that rises to one peak in the center and then falls, forming an arch shape, plain white background, no text, no labels"),
        ("lc_03", "a clean simple line graph with a single smooth line that dips down to one valley in the center and then rises back up, plain white background, no text, no labels"),
        ("lc_04", "a clean simple line graph with two smooth lines that cross exactly once in the center of the graph, plain white background, no text, no labels"),
        ("lc_05", "a clean simple line graph with a single perfectly flat horizontal line in the center, plain white background, no text, no labels, minimal style"),
        ("lc_06", "a clean simple line graph with a single smooth S-shaped curve going from bottom-left to top-right, plain white background, no text, no labels"),
        ("lc_07", "a clean simple line graph with a single line that oscillates up and down exactly 3 times across the width, plain white background, no text, no labels"),
        ("lc_08", "a clean simple line graph with two parallel horizontal lines at different heights, plain white background, no text, no labels, minimal style"),
        ("lc_09", "a clean simple line graph with a single line that starts flat near the bottom then rises sharply at the right end, a hockey stick shape, plain white background, no text, no labels"),
    ],
    "scatter_plot": [
        ("sp_00", "a clean simple scatter plot with data points clustered tightly in the top-right corner, plain white background, no text, no labels, minimal style"),
        ("sp_01", "a clean simple scatter plot with data points clustered tightly in the bottom-left corner, plain white background, no text, no labels, minimal style"),
        ("sp_02", "a clean simple scatter plot with two clearly distinct clusters of points separated by visible empty space, plain white background, no text, no labels"),
        ("sp_03", "a clean simple scatter plot with data points arranged along a diagonal line from bottom-left to top-right, plain white background, no text, no labels"),
        ("sp_04", "a clean simple scatter plot with data points arranged in a circular ring pattern, plain white background, no text, no labels, minimal style"),
        ("sp_05", "a clean simple scatter plot with data points uniformly and evenly distributed across the entire plot area, plain white background, no text, no labels"),
        ("sp_06", "a clean simple scatter plot with three clearly distinct clusters of points with empty space between them, plain white background, no text, no labels"),
        ("sp_07", "a clean simple scatter plot with most points densely clustered in the center and a few outlier points far away from the cluster, plain white background, no text, no labels"),
        ("sp_08", "a clean simple scatter plot with points forming an X cross pattern, plain white background, no text, no labels, minimal style"),
        ("sp_09", "a clean simple scatter plot with a dense cluster of points in the center and progressively fewer points radiating outward, plain white background, no text, no labels"),
    ],
    "network_diagram": [
        ("nd_00", "a clean simple network diagram with exactly 6 circular nodes connected by 8 directed arrows between them, plain white background, no text, no labels, minimal style"),
        ("nd_01", "a clean simple network diagram with exactly 4 circular nodes arranged at the corners of a square with edges connecting each adjacent pair, plain white background, no text, no labels"),
        ("nd_02", "a clean simple network diagram with 1 large central hub node connected by arrows pointing outward to exactly 5 surrounding satellite nodes, plain white background, no text, no labels"),
        ("nd_03", "a clean simple network diagram with exactly 3 circular nodes arranged in a triangle with directed arrows going around the triangle forming a cycle, plain white background, no text, no labels"),
        ("nd_04", "a clean simple network diagram with exactly 7 circular nodes connected in a single linear chain from left to right, plain white background, no text, no labels"),
        ("nd_05", "a clean simple network diagram with exactly 6 nodes forming two separate disconnected groups of 3 nodes each, plain white background, no text, no labels"),
        ("nd_06", "a clean simple network diagram with exactly 5 nodes in a star pattern with one center node and 4 outer nodes each connected only to the center, plain white background, no text, no labels"),
        ("nd_07", "a clean simple network diagram with exactly 4 circular nodes all fully connected to each other with arrows between every pair, plain white background, no text, no labels"),
        ("nd_08", "a clean simple network diagram with exactly 8 circular nodes arranged in two parallel rows of 4 with connecting arrows between the rows, plain white background, no text, no labels"),
        ("nd_09", "a clean simple network diagram with exactly 5 circular nodes connected in a ring where each node connects to the next forming a cycle, plain white background, no text, no labels"),
    ],
    "flowchart": [
        ("fc_00", "a clean simple flowchart with exactly 6 rectangular process boxes and 1 diamond decision shape connected by arrows flowing downward, plain white background, no text, no labels"),
        ("fc_01", "a clean simple flowchart with exactly 4 rectangular process boxes connected in a linear vertical sequence by downward arrows, plain white background, no text, no labels"),
        ("fc_02", "a clean simple flowchart with 1 oval start shape, 3 rectangular process boxes, 1 diamond decision shape, and 1 oval end shape connected by arrows, plain white background, no text"),
        ("fc_03", "a clean simple flowchart with 5 boxes where one diamond decision box branches left and right into two separate paths, plain white background, no text, no labels"),
        ("fc_04", "a clean simple flowchart with 3 parallel vertical branches that merge at the bottom into a single endpoint box, plain white background, no text, no labels"),
        ("fc_05", "a clean simple flowchart with a loop structure showing 2 rectangular boxes with one arrow looping back upward from the second to the first box, plain white background, no text, no labels"),
        ("fc_06", "a clean simple flowchart with 7 shapes including 2 diamond decision points creating multiple branching paths that merge at the end, plain white background, no text, no labels"),
        ("fc_07", "a clean simple flowchart with 4 rectangular process boxes connected in sequence and one feedback arrow pointing backward from the last box to the first box, plain white background, no text, no labels"),
        ("fc_08", "a clean simple flowchart with 6 shapes alternating between rectangular process boxes and diamond decision shapes in a vertical sequence, plain white background, no text, no labels"),
        ("fc_09", "a clean simple flowchart with 5 rectangular process boxes arranged in a 2x2 grid plus one at the top with connecting arrows between them, plain white background, no text, no labels"),
    ],
}

# ── Structural Specifications (for fidelity scoring) ──────────────────────────
# Each entry corresponds to PROMPTS[output_type][prompt_idx]

SPECS = {
    "bar_chart": [
        {"count": 5, "order": "descending"},
        {"count": 5, "order": "descending"},
        {"count": 6, "order": "peak_center"},
        {"count": 4, "order": "ascending"},
        {"count": 5, "order": "alternating"},
        {"count": 5, "order": "edges_tall"},
        {"count": 3, "order": "equal"},
        {"count": 5, "order": "center_tall"},
        {"count": 6, "order": "any"},
        {"count": 4, "order": "ascending"},
    ],
    "line_chart": [
        {"count": 1, "trend": "decreasing"},
        {"count": 1, "trend": "increasing"},
        {"count": 1, "trend": "peak"},
        {"count": 1, "trend": "valley"},
        {"count": 2, "trend": "crossing"},
        {"count": 1, "trend": "flat"},
        {"count": 1, "trend": "s_curve"},
        {"count": 1, "trend": "oscillating"},
        {"count": 2, "trend": "parallel"},
        {"count": 1, "trend": "hockey_stick"},
    ],
    "scatter_plot": [
        {"clusters": 1, "location": "top_right"},
        {"clusters": 1, "location": "bottom_left"},
        {"clusters": 2, "location": "separated"},
        {"clusters": 1, "location": "diagonal"},
        {"clusters": 1, "location": "circular"},
        {"clusters": 1, "location": "uniform"},
        {"clusters": 3, "location": "separated"},
        {"clusters": 1, "location": "outliers"},
        {"clusters": 1, "location": "x_pattern"},
        {"clusters": 1, "location": "radial"},
    ],
    "network_diagram": [
        {"nodes": 6, "edges": 8},
        {"nodes": 4, "edges": 4},
        {"nodes": 6, "edges": 5},
        {"nodes": 3, "edges": 3},
        {"nodes": 7, "edges": 6},
        {"nodes": 6, "edges": 4},
        {"nodes": 5, "edges": 4},
        {"nodes": 4, "edges": 6},
        {"nodes": 8, "edges": 8},
        {"nodes": 5, "edges": 5},
    ],
    "flowchart": [
        {"boxes": 6, "diamonds": 1},
        {"boxes": 4, "diamonds": 0},
        {"boxes": 3, "diamonds": 1, "ovals": 2},
        {"boxes": 4, "diamonds": 1},
        {"boxes": 3, "diamonds": 0},
        {"boxes": 2, "diamonds": 0},
        {"boxes": 5, "diamonds": 2},
        {"boxes": 4, "diamonds": 0},
        {"boxes": 3, "diamonds": 3},
        {"boxes": 5, "diamonds": 0},
    ],
}

# ── Computer Vision Fidelity Scoring ──────────────────────────────────────────

def _load_gray(img_path):
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def score_bar_chart(img_path, spec):
    """
    Detects bars by finding tall/wide dark contours on a white background.
    Score = 0.5 × count_accuracy + 0.5 × order_accuracy
    """
    try:
        img, gray = _load_gray(img_path)
        h, w = gray.shape

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bar_contours = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = ch / (cw + 1e-6)
            if area > 400 and (aspect > 1.2 or aspect < 0.8):
                bar_contours.append((x, y, cw, ch))
        bar_contours.sort(key=lambda b: b[0])

        expected = spec.get("count", 5)
        n = len(bar_contours)
        count_score = max(0.0, 1.0 - abs(n - expected) / max(expected, 1))

        order_score = 0.5
        if n >= 2:
            heights = [b[3] for b in bar_contours]
            order = spec.get("order", "any")
            diffs  = np.diff(heights)
            if order == "descending":
                order_score = np.mean(diffs <= 0)
            elif order == "ascending":
                order_score = np.mean(diffs >= 0)
            elif order == "equal":
                cv = np.std(heights) / (np.mean(heights) + 1e-6)
                order_score = max(0.0, 1.0 - cv * 3)
            elif order == "center_tall":
                mid = n // 2
                order_score = 1.0 if heights[mid] == max(heights) else 0.3
            elif order == "peak_center":
                mid = n // 2
                order_score = 1.0 if heights[mid] > heights[0] and heights[mid] > heights[-1] else 0.3
            elif order == "alternating":
                alternating = all(
                    (heights[i] > heights[i+1]) != (heights[i+1] > heights[i+2])
                    for i in range(len(heights)-2)
                )
                order_score = 1.0 if alternating else 0.3
            else:
                order_score = 0.6

        return round(0.5 * count_score + 0.5 * order_score, 4)
    except Exception:
        return 0.1


def score_line_chart(img_path, spec):
    """
    Traces the dominant dark line across columns and measures trend agreement.
    Score = 0.4 (line presence) + 0.6 × trend_accuracy
    """
    try:
        img, gray = _load_gray(img_path)
        h, w = gray.shape

        # Work in the inner 60% of the image to avoid border artifacts
        roi = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        rh, rw = roi.shape

        # Detect edges
        edges = cv2.Canny(roi, 40, 120)
        has_line = np.sum(edges) / (rh * rw * 255) > 0.003
        if not has_line:
            return 0.1

        # Sample column-wise: find the y-coordinate of the lowest-intensity pixel
        col_step = max(1, rw // 20)
        positions = []
        for c in range(0, rw, col_step):
            col = roi[:, c]
            idx = np.where(col < 128)[0]
            if len(idx):
                positions.append(float(np.mean(idx)) / rh)  # normalised 0–1

        trend_score = 0.5
        if len(positions) >= 4:
            trend = spec.get("trend", "any")
            p = np.array(positions)
            diffs = np.diff(p)
            if trend == "decreasing":
                # In image coords, y increases downward; a "decreasing value"
                # means the line moves UP → y decreases → diffs negative
                trend_score = float(np.mean(diffs <= 0))
            elif trend == "increasing":
                trend_score = float(np.mean(diffs >= 0))
            elif trend == "flat":
                trend_score = max(0.0, 1.0 - float(np.std(p)) * 8)
            elif trend == "peak":
                mid = len(p) // 2
                goes_up   = float(np.mean(diffs[:mid] < 0))   # rises to mid
                goes_down = float(np.mean(diffs[mid:] > 0))   # falls after mid
                trend_score = 0.5 * goes_up + 0.5 * goes_down
            elif trend == "valley":
                mid = len(p) // 2
                goes_down = float(np.mean(diffs[:mid] > 0))
                goes_up   = float(np.mean(diffs[mid:] < 0))
                trend_score = 0.5 * goes_down + 0.5 * goes_up
            elif trend == "hockey_stick":
                # Flat then sharp rise at the end
                first_half = np.std(p[:len(p)//2])
                second_half_diff = p[-1] - p[len(p)//2]
                trend_score = 0.5 * max(0, 1 - first_half*10) + 0.5 * max(0, -second_half_diff * 3)
            elif trend in ("parallel", "crossing"):
                trend_score = 0.5  # needs two-line detection; default neutral
            else:
                trend_score = 0.5

        return round(min(1.0, 0.4 + 0.6 * trend_score), 4)
    except Exception:
        return 0.1


def score_scatter_plot(img_path, spec):
    """
    Detects small blob-like points and analyses their spatial distribution.
    Score = 0.3 (points found) + 0.4 × location_score + 0.3 × cluster_score
    """
    try:
        img, gray = _load_gray(img_path)
        h, w = gray.shape

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea       = True
        params.minArea            = 15
        params.maxArea            = 3000
        params.filterByCircularity = True
        params.minCircularity     = 0.25
        params.filterByConvexity  = False
        params.filterByInertia    = False
        detector  = cv2.SimpleBlobDetector_create(params)
        _, inv    = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        keypoints = detector.detect(inv)
        n = len(keypoints)

        if n < 3:
            return 0.12

        pts = np.array([[kp.pt[0] / w, kp.pt[1] / h] for kp in keypoints])
        clusters = spec.get("clusters", 1)
        location = spec.get("location", "any")

        location_score = 0.5
        if location == "top_right":
            location_score = float(np.mean((pts[:, 0] > 0.5) & (pts[:, 1] < 0.5)))
        elif location == "bottom_left":
            location_score = float(np.mean((pts[:, 0] < 0.5) & (pts[:, 1] > 0.5)))
        elif location == "uniform":
            location_score = min(1.0, float(np.std(pts[:, 0]) + np.std(pts[:, 1])))
        elif location == "diagonal":
            # Points on y ≈ x line (top-left to bottom-right or reverse)
            correlation = abs(float(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1]))
            location_score = correlation
        elif location == "outliers":
            center = np.mean(pts, axis=0)
            dists  = np.linalg.norm(pts - center, axis=1)
            # Expect most points close and a few far
            threshold = np.percentile(dists, 80)
            outlier_frac = float(np.mean(dists > threshold))
            location_score = min(1.0, outlier_frac * 5)

        cluster_score = 0.5
        if clusters >= 2 and n >= 6:
            # Spatial spread: higher spread suggests multiple clusters
            spread = float(np.std(pts[:, 0]) + np.std(pts[:, 1]))
            cluster_score = min(1.0, spread)

        return round(min(1.0, 0.3 + 0.4 * location_score + 0.3 * cluster_score), 4)
    except Exception:
        return 0.1


def score_network_diagram(img_path, spec):
    """
    Uses Hough circle transform to count nodes and Hough lines to count edges.
    Score = 0.5 × node_score + 0.3 × edge_score + 0.2 × structure_bonus
    """
    try:
        img, gray = _load_gray(img_path)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
            param1=50, param2=30, minRadius=8, maxRadius=70
        )
        detected_nodes = len(circles[0]) if circles is not None else 0

        edges_img = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180,
                                threshold=25, minLineLength=15, maxLineGap=8)
        detected_edges = len(lines) if lines is not None else 0

        expected_nodes = spec.get("nodes", 6)
        expected_edges = spec.get("edges", 8)

        node_score = max(0.0, 1.0 - abs(detected_nodes - expected_nodes) / max(expected_nodes, 1))
        edge_score = max(0.0, 1.0 - abs(detected_edges - expected_edges) / max(expected_edges * 2, 1))
        structure  = 1.0 if (detected_nodes > 0 and detected_edges > 0) else 0.0

        return round(min(1.0, 0.5 * node_score + 0.3 * edge_score + 0.2 * structure), 4)
    except Exception:
        return 0.1


def score_flowchart(img_path, spec):
    """
    Classifies contours into rectangles, diamonds, and ovals.
    Score = 0.5 × total_count_score + 0.3 × diamond_score + 0.2 × presence_bonus
    """
    try:
        img, gray = _load_gray(img_path)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = diamonds = ovals = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 250:
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            n_v   = len(approx)
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / (ch + 1e-6)
            circ  = 4 * np.pi * area / (peri * peri + 1e-6)

            if circ > 0.65:
                ovals += 1
            elif n_v == 4:
                # Diamond: roughly equal w and h (rotated square)
                if 0.7 < aspect < 1.4:
                    diamonds += 1
                else:
                    rectangles += 1
            elif n_v in (4, 5, 6):
                rectangles += 1

        expected_boxes    = spec.get("boxes", 4)
        expected_diamonds = spec.get("diamonds", 1)
        expected_ovals    = spec.get("ovals", 0)
        total_expected    = expected_boxes + expected_diamonds + expected_ovals
        total_detected    = rectangles + diamonds + ovals

        count_score   = max(0.0, 1.0 - abs(total_detected - total_expected) / max(total_expected, 1))
        diamond_score = max(0.0, 1.0 - abs(diamonds - expected_diamonds) / max(expected_diamonds + 1, 1))
        presence      = 1.0 if total_detected > 0 else 0.0

        return round(min(1.0, 0.5 * count_score + 0.3 * diamond_score + 0.2 * presence), 4)
    except Exception:
        return 0.1


def score_image(img_path, output_type, prompt_idx):
    """Dispatch to the correct scoring function."""
    spec = SPECS[output_type][prompt_idx]
    if output_type == "bar_chart":
        return score_bar_chart(img_path, spec)
    elif output_type == "line_chart":
        return score_line_chart(img_path, spec)
    elif output_type == "scatter_plot":
        return score_scatter_plot(img_path, spec)
    elif output_type == "network_diagram":
        return score_network_diagram(img_path, spec)
    elif output_type == "flowchart":
        return score_flowchart(img_path, spec)
    return 0.0

# ── Pipeline Loader ───────────────────────────────────────────────────────────

def load_pipeline():
    print(f"Loading SD 1.5 on {DEVICE} ({DTYPE}) ...")
    t0   = time.time()
    pipe = StableDiffusionPipeline.from_single_file(
        str(MODEL_FILE),
        torch_dtype=DTYPE,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    print(f"Model ready ({time.time()-t0:.1f}s). GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}\n")
    return pipe

# ── Resume Support ─────────────────────────────────────────────────────────────

def load_done():
    """Return set of (output_type, prompt_id, seed) already saved."""
    done = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["output_type"], row["prompt_id"], int(row["seed"])))
    return done

# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    # Create image subdirectories
    for otype, prompt_list in PROMPTS.items():
        for pid, _ in prompt_list:
            (IMAGES_DIR / otype / pid).mkdir(parents=True, exist_ok=True)

    done = load_done()
    total_done_start = len(done)
    total = 50_000
    print(f"Resuming: {total_done_start} / {total} already done "
          f"({total - total_done_start} remaining)\n")

    csv_exists = RESULTS_CSV.exists()
    csv_file   = open(RESULTS_CSV, "a", newline="")
    writer     = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow([
            "output_type", "prompt_id", "prompt_idx",
            "seed", "fidelity_score", "above_threshold",
            "image_path"
        ])
        csv_file.flush()

    pipe = load_pipeline()

    completed  = 0
    start_time = time.time()
    todo       = total - total_done_start

    for output_type, prompt_list in PROMPTS.items():
        for prompt_idx, (prompt_id, prompt_text) in enumerate(prompt_list):
            for seed in range(SEED_START, SEED_END + 1):

                key = (output_type, prompt_id, seed)
                if key in done:
                    continue

                img_path = IMAGES_DIR / output_type / prompt_id / f"seed_{seed:04d}.png"

                try:
                    generator = torch.Generator(device=DEVICE).manual_seed(seed)
                    result    = pipe(
                        prompt_text,
                        num_inference_steps=NUM_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                    )
                    result.images[0].save(img_path)

                    fidelity = score_image(img_path, output_type, prompt_idx)
                    above    = 1 if fidelity >= ACCEPTANCE_THRESHOLD else 0

                    writer.writerow([
                        output_type, prompt_id, prompt_idx,
                        seed, fidelity, above,
                        str(img_path.relative_to(BASE_DIR))
                    ])
                    csv_file.flush()
                    done.add(key)
                    completed += 1

                    if completed % 200 == 0:
                        elapsed  = time.time() - start_time
                        rate     = completed / elapsed
                        eta_h    = (todo - completed) / rate / 3600
                        pct      = 100 * (total_done_start + completed) / total
                        print(
                            f"[{total_done_start+completed:>6}/{total}] "
                            f"{pct:5.1f}%  {output_type}/{prompt_id}/s{seed:04d}  "
                            f"f={fidelity:.3f}  "
                            f"{rate:.2f} img/s  ETA {eta_h:.1f}h"
                        )

                except KeyboardInterrupt:
                    print("\n⚠  Interrupted by user. Progress saved. Re-run to continue.")
                    csv_file.close()
                    return

                except Exception as exc:
                    print(f"  ERROR  {output_type}/{prompt_id}/s{seed}: {exc}")
                    writer.writerow([
                        output_type, prompt_id, prompt_idx,
                        seed, 0.0, 0, "ERROR"
                    ])
                    csv_file.flush()

    csv_file.close()
    elapsed = time.time() - start_time
    print(f"\nExperiment complete.  {completed} new images in {elapsed/3600:.1f}h.")
    print(f"Results → {RESULTS_CSV}")


if __name__ == "__main__":
    main()

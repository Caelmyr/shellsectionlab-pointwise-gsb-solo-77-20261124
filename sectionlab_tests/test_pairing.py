"""Standalone checks for paired pore matching and paired line intercept."""
import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402

from sectionlab.analysis import (  # noqa: E402
    paired_pore_intercept,
    paired_pore_map,
    polygon_mask,
)
from sectionlab.photos import Frame  # noqa: E402

HEIGHT, WIDTH = 400, 500
STEP = 0.05  # mm per pixel -> 25 mm x 20 mm frame


def make_frame(circles, blocked=()):
    rgb = np.full((HEIGHT, WIDTH, 3), 0.72, np.float32)
    support = np.ones((HEIGHT, WIDTH), bool)
    for x0, y0, radius in blocked:
        cv.circle(rgb, (x0, y0), radius, (1.0, 1.0, 1.0), -1)
        cv.circle(support.view(np.uint8), (x0, y0), radius, 0, -1)
    for x0, y0, radius, shade in circles:
        cv.circle(rgb, (x0, y0), radius, (shade, shade, shade), -1, cv.LINE_AA)
    return Frame(rgb, support, (STEP, STEP))


def window_mark():
    return {"points": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]}


def line_mark(x0, y0, x1, y1):
    return {"points": [[x0 / (WIDTH - 1), y0 / (HEIGHT - 1)],
                       [x1 / (WIDTH - 1), y1 / (HEIGHT - 1)]]}


def base_circles():
    # four well-separated compact dark pores (radius >= 12 px so area clears the 0.10 mm² floor)
    return [(90, 90, 12, 0.12), (220, 110, 13, 0.10), (330, 210, 12, 0.14), (150, 310, 14, 0.11)]


def test_basic_pairing_and_uniqueness():
    frame_a = make_frame(base_circles())
    shifted = [(x + 3, y - 2, r, s) for x, y, r, s in base_circles()]
    frame_b = make_frame(shifted)
    marks = {"region": window_mark(), "comparison": window_mark()}
    summary, rows, _, pairing = paired_pore_map(frame_a, frame_b, marks, {})
    assert summary["paired_pore_count"] == 4, summary
    assert summary["single_photo_dark_spots"] == 0
    assert summary["missing_evidence"] == 0
    used_a = [row["pair"] for row in rows if "pair" in row]
    assert len(set(used_a)) == 4
    for pair in pairing["pairs"]:
        assert pair["center_separation_mm"] <= 0.30
        assert pair["overlap_mm2"] >= 0.03
    # polygons round-trip to masks covering real pixels
    for pair in pairing["pairs"]:
        assert polygon_mask(pair["reference_polygon"], frame_a.rgb.shape).sum() > 10
    print("test_basic_pairing_and_uniqueness OK", summary)


def test_missing_evidence_vs_single_photo():
    circles_a = base_circles()
    # photo B misses pore 4; block the area of pore 4 in B so it counts as missing evidence
    frame_a = make_frame(circles_a)
    frame_b = make_frame(circles_a[:3], blocked=[(150, 310, 24)])
    marks = {"region": window_mark(), "comparison": window_mark()}
    summary, rows, _, pairing = paired_pore_map(frame_a, frame_b, marks, {})
    assert summary["paired_pore_count"] == 3, summary
    assert summary["missing_evidence"] == 1, summary
    assert summary["single_photo_dark_spots"] == 0
    missing = pairing["reference_unmatched"]
    assert len(missing) == 1 and missing[0]["status"] == "Missing evidence"
    assert missing[0]["counterpart_visible"] is False

    # now B sees the area but simply lacks the dark pore -> single-photo dark spot
    frame_c = make_frame(circles_a[:3])
    summary2, _, _, pairing2 = paired_pore_map(frame_a, frame_c, marks, {})
    assert summary2["paired_pore_count"] == 3
    assert summary2["single_photo_dark_spots"] == 1, summary2
    assert summary2["missing_evidence"] == 0
    assert pairing2["reference_unmatched"][0]["status"] == "Single-photo dark spot"
    print("test_missing_evidence_vs_single_photo OK")


def test_tie_break_prefers_more_pairs_then_overlap():
    # Build an ambiguous graph: two B pores near two A pores; assignment must maximize pair count.
    circles_a = [(120, 120, 13, 0.12), (180, 120, 13, 0.12)]
    circles_b = [(122, 120, 13, 0.12), (182, 120, 13, 0.12)]
    frame_a = make_frame(circles_a)
    frame_b = make_frame(circles_b)
    marks = {"region": window_mark(), "comparison": window_mark()}
    summary, _, _, pairing = paired_pore_map(frame_a, frame_b, marks,
                                             {"max_center_mm": 5, "min_overlap_mm2": 0.001})
    assert summary["paired_pore_count"] == 2, summary
    centers_a = sorted(tuple(p["center_reference_mm"]) for p in pairing["pairs"])
    centers_b = sorted(tuple(p["center_comparison_mm"]) for p in pairing["pairs"])
    # nearest-to-nearest, not a crossing assignment
    assert all(np.linalg.norm(np.array(a) - np.array(b)) < 1 for a, b in zip(centers_a, centers_b))

    # overlap tie-break: same number of pairs possible; larger shared area must win
    big_a = [(120, 250, 13, 0.12)]
    # two candidate B pores: one nearly coincident (big overlap), one offset (small overlap),
    # both within centre limit and overlap floor -> algorithm keeps the higher-overlap edge.
    big_b = [(121, 250, 13, 0.12), (128, 250, 13, 0.12)]
    f_a = make_frame(big_a)
    f_b = make_frame(big_b)
    summary3, _, _, pairing3 = paired_pore_map(f_a, f_b, marks,
                                               {"max_center_mm": 5, "min_overlap_mm2": 0.001})
    assert summary3["paired_pore_count"] == 1
    only = pairing3["pairs"][0]
    assert only["center_comparison_mm"][0] < 126, only
    print("test_tie_break_prefers_more_pairs_then_overlap OK")


def test_threshold_rejects_weak_matches():
    frame_a = make_frame(base_circles())
    # shift beyond centre limit
    shifted = [(x + 40, y, r, s) for x, y, r, s in base_circles()]
    frame_b = make_frame(shifted)
    marks = {"region": window_mark(), "comparison": window_mark()}
    summary, _, _, _ = paired_pore_map(frame_a, frame_b, marks, {})
    assert summary["paired_pore_count"] == 0
    assert summary["single_photo_dark_spots"] + summary["missing_evidence"] >= 4
    print("test_threshold_rejects_weak_matches OK")


def test_paired_intercept_crosses_adopted_pores():
    frame_a = make_frame(base_circles())
    frame_b = make_frame([(x + 2, y, r, s) for x, y, r, s in base_circles()])
    marks = {"region": window_mark(), "comparison": window_mark()}
    _, _, _, pairing = paired_pore_map(frame_a, frame_b, marks, {})
    # horizontal survey line through the first two pore centres' overlap band (y = 100)
    line = line_mark(60, 100, 280, 100)
    summary, rows, _ = paired_pore_intercept(frame_a, {"line": line}, {"luma": .31}, pairing)
    adopted = [row for row in rows if row.get("entry") == "Adopted paired pore"]
    assert summary["adopted_pores_crossed"] == len(adopted) >= 2, summary
    for row in adopted:
        assert row["crossing_mm"] > 0
        assert row["to_mm"] > row["from_mm"]
    # interval rows still present
    assert any("classification" in row for row in rows)
    print("test_paired_intercept_crosses_adopted_pores OK", summary)


def test_intercept_lists_single_photo_separately():
    circles_a = base_circles()
    frame_a = make_frame(circles_a)
    frame_b = make_frame(circles_a[:3])  # pore 4 absent but visible
    marks = {"region": window_mark(), "comparison": window_mark()}
    _, _, _, pairing = paired_pore_map(frame_a, frame_b, marks, {})
    # line through missing-in-b pore at (150, 300)
    line = line_mark(110, 310, 190, 310)
    summary, rows, _ = paired_pore_intercept(frame_a, {"line": line}, {"luma": .31}, pairing)
    singles = [row for row in rows if row.get("entry") == "Single-photo dark spot"]
    missing = [row for row in rows if row.get("entry") == "Missing evidence"]
    assert summary["single_photo_crossings"] == len(singles) == 1
    assert summary["missing_evidence_crossings"] == len(missing) == 0
    assert singles[0]["crossing_mm"] > 0
    print("test_intercept_lists_single_photo_separately OK")


if __name__ == "__main__":
    test_basic_pairing_and_uniqueness()
    test_missing_evidence_vs_single_photo()
    test_tie_break_prefers_more_pairs_then_overlap()
    test_threshold_rejects_weak_matches()
    test_paired_intercept_crosses_adopted_pores()
    test_intercept_lists_single_photo_separately()
    print("all pairing checks passed")

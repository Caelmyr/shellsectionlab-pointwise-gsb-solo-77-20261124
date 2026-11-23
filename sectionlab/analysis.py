"""Physical-plane measurements for registered shell-section photographs."""
import math

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from image_lab.validation import Invalid

from .model import number
from .photos import coordinates, image, region


def tint(shown, mask, color, amount=.58):
    shown[mask] = shown[mask] * (1 - amount) + np.asarray(color) * amount


def selection(frame, mark, minimum=32):
    mask = region(frame, mark) & frame.useful
    if np.count_nonzero(mask) < minimum:
        raise Invalid("The marked patch needs more usable image pixels.")
    return mask


def rounded(values, digits=4):
    return np.round(values, digits).tolist()


def lab(pixels):
    # Convert standard RGB-encoded input to display-referred Lab.
    return cv.cvtColor(np.asarray(pixels, np.float32).reshape(-1, 1, 3), cv.COLOR_RGB2LAB).reshape(-1, 3)


def section_tone(frame, marks, _opts):
    mask = selection(frame, marks["region"])
    lab_pixels = lab(frame.rgb[mask])
    low, center, high = np.percentile(lab_pixels, [10, 50, 90], axis=0)
    rows = [
        {"channel": name, "p10": round(float(low[i]), 4), "median": round(float(center[i]), 4),
         "p90": round(float(high[i]), 4)}
        for i, name in enumerate(("L*", "a*", "b*"))
    ]
    shown = image(frame)
    tint(shown, mask, (.38, .78, .86), .28)
    return {"pixels": int(mask.sum()), "lab": rounded(center), "lightness_spread": round(float(high[0] - low[0]), 4)}, rows, shown


def section_contrast(frame, marks, _opts):
    masks = {key: selection(frame, marks[key]) for key in ("matrix", "accent")}
    values = {key: np.median(lab(frame.rgb[area]), axis=0) for key, area in masks.items()}
    lightness_drop = float(values["matrix"][0] - values["accent"][0])
    distance = float(np.linalg.norm(values["matrix"] - values["accent"]))
    shown = image(frame)
    tint(shown, masks["matrix"], (.84, .75, .32))
    tint(shown, masks["accent"], (.39, .68, .96))
    rows = [{"mark": marks[key]["title"], "lab": rounded(value), "pixels": int(masks[key].sum())}
            for key, value in values.items()]
    return {"lightness_drop": round(lightness_drop, 4), "delta_e76": round(distance, 4)}, rows, shown


def inclusion_projection(frame, matrix, accent):
    blank_rgb = np.median(frame.rgb[selection(frame, matrix)], axis=0)
    accent_rgb = np.median(frame.rgb[selection(frame, accent)], axis=0)
    difference = accent_rgb - blank_rgb
    leverage = float(difference @ difference)
    if leverage < .014:
        raise Invalid("Choose two color controls with stronger color separation.")
    return (frame.rgb - blank_rgb) @ difference / leverage


def inclusion_fraction(frame, marks, opts):
    area = selection(frame, marks["region"])
    fraction = inclusion_projection(frame, marks["matrix"], marks["accent"])
    threshold = number(opts.get("threshold", .5), "inclusion likeness threshold", .1, .9)
    decorated = area & (fraction >= threshold)
    border = area & (np.abs(fraction - threshold) < .08)
    shown = image(frame)
    tint(shown, area & ~decorated, (.36, .71, .95), .2)
    tint(shown, decorated, (.27, .79, .51))
    tint(shown, border, (.94, .67, .29), .34)
    rows = [
        {"class": key, "pixels": int(mask.sum())}
        for key, mask in (("Inclusion-like", decorated), ("Base-like", area & ~decorated), ("Near threshold", border))
    ]
    return {"inclusion_like_area_pct": round(100 * decorated.sum() / area.sum(), 3),
            "threshold": threshold, "pixels": int(area.sum())}, rows, shown


def lamina_spacing(frame, marks, _opts):
    mask = selection(frame, marks["region"], 250)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if min(y1 - y0, x1 - x0) < 24:
        raise Invalid("Use at least 24 calibrated pixels along both repeat patch axes.")
    luma = frame.luma[y0:y1, x0:x1]
    present = mask[y0:y1, x0:x1]
    window = np.outer(np.hanning(luma.shape[0]), np.hanning(luma.shape[1]))
    centered = np.where(present, luma - np.mean(luma[present]), 0)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    freq_x = np.fft.fftshift(np.fft.fftfreq(power.shape[1], d=frame.step[0]))
    freq_y = np.fft.fftshift(np.fft.fftfreq(power.shape[0], d=frame.step[1]))
    xx, yy = np.meshgrid(freq_x, freq_y)
    possible = ((yy > 0) | ((yy == 0) & (xx > 0))) & (np.hypot(xx, yy) > .18)
    rows = []
    candidate = np.where(possible, power, 0)
    peak = float(candidate.max())
    if peak <= 1e-6:
        raise Invalid("This patch does not contain enough periodic tonal variation.")
    for i in range(1, 5):
        iy, ix = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        strength = float(candidate[iy, ix]) / peak
        if strength < .075:
            break
        rows.append({"peak": i, "spacing_mm": round(float(1 / np.hypot(xx[iy, ix], yy[iy, ix])), 4),
                     "lamina_direction_deg": round((math.degrees(math.atan2(yy[iy, ix], xx[iy, ix])) + 90) % 180, 2),
                     "relative_power": round(strength, 4)})
        candidate[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4] = 0
    shown = image(frame)
    tint(shown, mask, (.52, .79, .91), .22)
    return {"dominant_spacing_mm": rows[0]["spacing_mm"], "dominant_lamina_deg": rows[0]["lamina_direction_deg"],
            "peaks": len(rows)}, rows, shown


def _contour_polygon(contour, shape, offset=(0, 0)):
    """Simplify a pixel contour to normalized calibrated-plane vertices."""
    height, width = shape[:2]
    perimeter = cv.arcLength(contour.astype(np.float32), True)
    if perimeter:
        contour = cv.approxPolyDP(contour, .012 * perimeter, True)
    vertices = contour.reshape(-1, 2).astype(np.float64) + np.asarray(offset, dtype=np.float64)
    normalized = np.stack((vertices[:, 0] / (width - 1), vertices[:, 1] / (height - 1)), axis=1)
    return np.round(normalized, 5).tolist()


def polygon_mask(polygon, shape):
    bitmap = np.zeros(shape[:2], dtype=np.uint8)
    height, width = shape[:2]
    vertices = np.rint(np.asarray(polygon) * (width - 1, height - 1)).astype(np.int32)
    cv.fillPoly(bitmap, [vertices], 1)
    return bitmap.astype(bool)


def detect_pores(frame, area_mask, cutoff):
    """Compact dark pore components inside a calibrated window, elongated fissures rejected."""
    candidates = area_mask & (frame.luma <= cutoff)
    count, labels, stats, centroids = cv.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    pixel_area = frame.step[0] * frame.step[1]
    limited = cv.dilate((~area_mask).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pits = np.zeros(area_mask.shape, dtype=bool)
    findings = []
    for label in range(1, count):
        x, y, width, height, pixels = stats[label]
        area_mm2 = float(pixels * pixel_area)
        horizontal, vertical = width * frame.step[0], height * frame.step[1]
        if not .10 <= area_mm2 <= 80 or max(horizontal, vertical) / min(horizontal, vertical) > 2.6:
            continue
        local = labels[y:y + height, x:x + width] == label
        pits[y:y + height, x:x + width] |= local
        footprint = local.astype(np.uint8)
        contours, _ = cv.findContours(footprint, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv.arcLength((contour * frame.step).astype(np.float32), True) for contour in contours)
        polygon = [point for contour in contours
                   for point in _contour_polygon(contour, frame.rgb.shape, offset=(x, y))]
        findings.append({
            "area_mm2": round(area_mm2, 4), "center_mm": rounded(centroids[label] * frame.step),
            "width_mm": round(horizontal, 4), "height_mm": round(vertical, 4),
            "boundary_limited": bool(np.any(local & limited[y:y + height, x:x + width])),
            "circularity": round(min(1., 4 * math.pi * area_mm2 / perimeter ** 2), 4) if perimeter else None,
            "polygon": polygon,
        })
    findings.sort(key=lambda item: item["area_mm2"], reverse=True)
    return findings, pits, candidates


def pore_map(frame, marks, opts):
    """Separate compact pore candidates from elongated dark features."""
    area = selection(frame, marks["region"])
    cutoff = number(opts.get("luma", .31), "dark pore threshold", .05, .95)
    rows, pits, candidates = detect_pores(frame, area, cutoff)
    shown = image(frame)
    tint(shown, area & ~candidates, (.25, .78, .73), .22)
    tint(shown, candidates & ~pits, (.65, .36, .83), .44)
    tint(shown, pits, (.95, .48, .24))
    public_rows = [{key: value for key, value in row.items() if key != "polygon"} for row in rows]
    pixel_area = frame.step[0] * frame.step[1]
    return {"pore_candidate_count": len(rows), "candidate_area_mm2": round(float(pits.sum() * pixel_area), 4),
            "observed_mm2": round(float(area.sum() * pixel_area), 4)}, public_rows[:40], shown


def _pore_footprint(finding, shape):
    return polygon_mask(finding["polygon"], shape)


def paired_pore_map(frame, other, marks, opts):
    """Match the same physical pores across two registered photographs of one section."""
    if frame.rgb.shape[:2] != other.rgb.shape[:2]:
        raise Invalid("The two photographs need matching calibrated frame dimensions.")
    region_a = selection(frame, marks["region"])
    region_b = selection(other, marks["comparison"])
    cutoff = number(opts.get("luma", .31), "dark pore threshold", .05, .95)
    max_center = number(opts.get("max_center_mm", .30), "maximum pore centre separation (mm)", .01, 5)
    min_overlap = number(opts.get("min_overlap_mm2", .03), "minimum shared pore area (mm²)", .001, 20)
    pores_a, pits_a, _ = detect_pores(frame, region_a, cutoff)
    pores_b, pits_b, _ = detect_pores(other, region_b, cutoff)
    if not pores_a or not pores_b:
        raise Invalid("Each photograph needs at least one compact dark pore inside its inspection window.")
    pixel_area = frame.step[0] * frame.step[1]
    footprints_b = [_pore_footprint(finding, other.rgb.shape) for finding in pores_b]
    # A legal pair must pass both the centre-separation and shared-area rules; each pore is used once.
    # Infeasible edges carry a finite penalty far below any feasible one so the assignment still
    # completes, then only the feasible edges are kept.
    penalty = -1e9
    weights = np.full((len(pores_a), len(pores_b)), penalty, dtype=np.float64)
    metrics = {}
    for i, pore_a in enumerate(pores_a):
        footprint_a = _pore_footprint(pore_a, frame.rgb.shape)
        center_a = np.asarray(pore_a["center_mm"])
        for j, pore_b in enumerate(pores_b):
            separation = float(np.linalg.norm(center_a - np.asarray(pore_b["center_mm"])))
            overlap = float(np.count_nonzero(footprint_a & footprints_b[j]) * pixel_area)
            if separation <= max_center and overlap >= min_overlap:
                # Lexicographic rule: most pairs first (feasible edges only), then largest total
                # shared area, then smallest total centre separation. All ties stay deterministic.
                weights[i, j] = overlap * 1000.0 - separation
                metrics[(i, j)] = (overlap, separation)
    row_index, col_index = linear_sum_assignment(-weights)
    chosen = [(i, j) for i, j in zip(row_index, col_index) if weights[i, j] > penalty / 2]
    chosen.sort(key=lambda edge: pores_a[edge[0]]["area_mm2"], reverse=True)
    paired, paired_masks = [], []
    used_b = set()
    for pair_index, (i, j) in enumerate(chosen, 1):
        overlap, separation = metrics[(i, j)]
        used_b.add(j)
        paired_masks.append(_pore_footprint(pores_a[i], frame.rgb.shape) | footprints_b[j])
        paired.append({
            "pair": pair_index,
            "center_reference_mm": pores_a[i]["center_mm"], "center_comparison_mm": pores_b[j]["center_mm"],
            "center_separation_mm": round(separation, 4),
            "overlap_mm2": round(overlap, 4),
            "reference_area_mm2": pores_a[i]["area_mm2"], "comparison_area_mm2": pores_b[j]["area_mm2"],
            "reference_polygon": pores_a[i]["polygon"], "comparison_polygon": pores_b[j]["polygon"],
        })

    def unmatched_row(finding, photo_key, counterpart_visible):
        return {"photo": photo_key, "center_mm": finding["center_mm"], "area_mm2": finding["area_mm2"],
                "counterpart_visible": bool(counterpart_visible),
                "status": "Missing evidence" if not counterpart_visible else "Single-photo dark spot",
                "polygon": finding["polygon"]}

    visible_b = other.useful & region_b
    unmatched_a, unmatched_masks = [], []
    matched_a = {i for i, _ in chosen}
    for i, finding in enumerate(pores_a):
        if i in matched_a:
            continue
        footprint = _pore_footprint(finding, frame.rgb.shape)
        unmatched_masks.append(footprint)
        unmatched_a.append(unmatched_row(finding, "reference", bool(np.any(visible_b & footprint))))
    visible_a = frame.useful & region_a
    unmatched_b = []
    for j, finding in enumerate(pores_b):
        if j in used_b:
            continue
        footprint = footprints_b[j]
        unmatched_masks.append(footprint)
        unmatched_b.append(unmatched_row(finding, "comparison", bool(np.any(visible_a & footprint))))

    def polygon_pack(items):
        return [{"center_mm": item["center_mm"], "area_mm2": item["area_mm2"], "polygon": item["polygon"],
                 "status": item["status"], "counterpart_visible": item["counterpart_visible"]}
                for item in items]

    shown = image(frame)
    window = region_a | region_b
    tint(shown, window & ~(pits_a | pits_b), (.25, .78, .73), .20)
    tint(shown, pits_a, (.95, .48, .24), .42)
    tint(shown, pits_b, (.55, .47, .92), .34)
    for mask in unmatched_masks:
        tint(shown, mask, (.72, .30, .28), .50)
    for mask in paired_masks:
        tint(shown, mask, (.20, .82, .56), .55)
    summary = {
        "paired_pore_count": len(paired),
        "reference_candidates": len(pores_a), "comparison_candidates": len(pores_b),
        "reference_unmatched": len(unmatched_a), "comparison_unmatched": len(unmatched_b),
        "missing_evidence": sum(1 for row in unmatched_a + unmatched_b
                                if row["status"] == "Missing evidence"),
        "single_photo_dark_spots": sum(1 for row in unmatched_a + unmatched_b
                                       if row["status"] == "Single-photo dark spot"),
        "max_center_mm": max_center, "min_overlap_mm2": min_overlap,
        "mean_center_separation_mm": round(
            float(np.mean([row["center_separation_mm"] for row in paired])) if paired else 0., 4),
        "mean_overlap_mm2": round(float(np.mean([row["overlap_mm2"] for row in paired])) if paired else 0., 4),
    }
    pairing = {
        "rule": "Maximum number of one-to-one pairs, then maximum total shared area, then minimum total centre separation.",
        "max_center_mm": max_center, "min_overlap_mm2": min_overlap,
        "pairs": paired,
        "reference_unmatched": polygon_pack(unmatched_a),
        "comparison_unmatched": polygon_pack(unmatched_b),
    }

    def public_row(item):
        return {key: value for key, value in item.items() if not key.endswith("polygon")}

    table_rows = [public_row(item) for item in paired]
    table_rows.extend(public_row(item) for item in unmatched_a)
    table_rows.extend(public_row(item) for item in unmatched_b)
    return summary, table_rows, shown, pairing


def growth_transition(frame, marks, opts):
    knots = coordinates(marks["line"]["points"], frame.rgb.shape)
    length = float(np.linalg.norm((knots[1] - knots[0]) * frame.step))
    if length < 1:
        raise Invalid("Use a transect at least 1 mm long.")
    count = min(220, max(24, int(length * 5)))
    locations = np.linspace(knots[0], knots[1], count, dtype=np.float32)
    xs, ys = locations.T
    luma = cv.remap(frame.luma, xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_LINEAR)[0]
    valid = cv.remap(frame.useful.astype(np.float32), xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_LINEAR)[0] >= .999
    if valid.sum() < count * .85:
        raise Invalid("More of the edge transect must be visible in the photograph.")
    strip = max(3, count // 7)
    lead, tail = float(np.median(luma[:strip])), float(np.median(luma[-strip:]))
    if abs(tail - lead) < .07:
        raise Invalid("The two ends should fall on visibly different shell tones.")
    relative = np.convolve(np.pad((luma - lead) / (tail - lead), 2, mode="edge"), np.full(5, .2), mode="valid")
    position = np.linspace(0, length, count)
    limits = [np.flatnonzero(valid & (relative >= level)) for level in (.1, .9)]
    if any(len(indices) == 0 for indices in limits):
        raise Invalid("The transect does not bracket the shell growth transition.")
    width = float(position[limits[1][0]] - position[limits[0][0]])
    shown = image(frame)
    cv.line(shown, tuple(np.rint(knots[0]).astype(int)), tuple(np.rint(knots[1]).astype(int)), (.96, .52, .27), 2)
    rows = [{"position_mm": round(float(x), 3), "luma": round(float(y), 4) if on else None}
            for x, y, on in zip(position, luma, valid)]
    return {"edge_width_mm": round(max(0, width), 4), "length_mm": round(length, 4), "usable_samples": int(valid.sum())}, rows, shown


def pore_intercept(frame, marks, opts):
    """Measure observed dark and clear subintervals of a registered survey line."""
    ends = coordinates(marks["line"]["points"], frame.rgb.shape)
    length = float(np.linalg.norm((ends[1] - ends[0]) * frame.step))
    if length < 1:
        raise Invalid("Use a survey line at least 1 mm long.")
    count = min(2400, max(20, int(np.ceil(length / min(frame.step))) + 1))
    line = np.linspace(ends[0], ends[1], count, dtype=np.float32)
    widths = np.diff(np.linspace(0, length, count + 1))
    x = line[:, 0].reshape(1, -1)
    y = line[:, 1].reshape(1, -1)
    brightness = cv.remap(frame.luma, x, y, cv.INTER_LINEAR)[0]
    observed = cv.remap(frame.useful.astype(np.float32), x, y, cv.INTER_LINEAR)[0] >= .999
    threshold = number(opts.get("luma", .31), "dark pore threshold", .05, .95)
    dark = observed & (brightness <= threshold)
    clear = observed & ~dark
    states = np.select([dark, clear], [1, 2])
    names = ("Not observed", "Dark intercept", "Clear section")
    colors = ((.42, .45, .53), (.97, .51, .32), (.30, .77, .63))
    boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, count]
    start_end = np.linspace(0, length, count + 1)
    rows = []
    overlay = image(frame)
    for left, right in zip(boundaries, boundaries[1:]):
        key = states[left]
        rows.append({"classification": names[key], "from_mm": round(float(start_end[left]), 4),
                     "to_mm": round(float(start_end[right]), 4)})
        a = tuple(np.rint(line[left]).astype(int))
        b = tuple(np.rint(line[right - 1]).astype(int))
        cv.line(overlay, a, b, colors[key], 2, cv.LINE_AA)
    observed_length = float(widths[observed].sum())
    dark_length = float(widths[dark].sum())
    return {"observed_mm": round(observed_length, 4), "dark_mm": round(dark_length, 4),
            "unknown_mm": round(length - observed_length, 4)}, rows, overlay


def _survey_samples(frame, marks):
    ends = coordinates(marks["line"]["points"], frame.rgb.shape)
    length = float(np.linalg.norm((ends[1] - ends[0]) * frame.step))
    if length < 1:
        raise Invalid("Use a survey line at least 1 mm long.")
    count = min(2400, max(20, int(np.ceil(length / min(frame.step))) + 1))
    line = np.linspace(ends[0], ends[1], count, dtype=np.float32)
    widths = np.diff(np.linspace(0, length, count + 1))
    x = line[:, 0].reshape(1, -1)
    y = line[:, 1].reshape(1, -1)
    brightness = cv.remap(frame.luma, x, y, cv.INTER_LINEAR)[0]
    observed = cv.remap(frame.useful.astype(np.float32), x, y, cv.INTER_LINEAR)[0] >= .999
    return ends, line, widths, length, brightness, observed


def paired_pore_intercept(frame, marks, opts, pairing):
    """Measure which adopted paired pores a survey line actually crosses, with single-photo and missing entries."""
    shape = frame.rgb.shape
    ends, line, widths, length, brightness, observed = _survey_samples(frame, marks)
    threshold = number(opts.get("luma", .31), "dark pore threshold", .05, .95)
    dark = observed & (brightness <= threshold)
    clear = observed & ~dark
    states = np.select([dark, clear], [1, 2])
    names = ("Not observed", "Dark intercept", "Clear section")
    colors = ((.42, .45, .53), (.97, .51, .32), (.30, .77, .63))
    boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(line)]
    start_end = np.linspace(0, length, len(line) + 1)
    rows = []
    overlay = image(frame)
    for left, right in zip(boundaries, boundaries[1:]):
        key = states[left]
        rows.append({"classification": names[key], "from_mm": round(float(start_end[left]), 4),
                     "to_mm": round(float(start_end[right]), 4)})
        a = tuple(np.rint(line[left]).astype(int))
        b = tuple(np.rint(line[right - 1]).astype(int))
        cv.line(overlay, a, b, colors[key], 2, cv.LINE_AA)

    sample_xy = np.rint(line).astype(int)
    flat_index = sample_xy[:, 1] * shape[1] + sample_xy[:, 0]

    def crossing_length(polygon):
        mask = polygon_mask(polygon, shape)
        inside = mask.reshape(-1)[flat_index]
        return float(widths[inside].sum()), inside

    crossed_pairs, crossed_masks = [], []
    for item in pairing["pairs"]:
        run_length, inside = crossing_length(item["reference_polygon"])
        if run_length <= 0:
            continue
        hit = np.flatnonzero(inside)
        crossed_pairs.append({
            "entry": "Adopted paired pore", "pair": item["pair"],
            "from_mm": round(float(start_end[hit[0]]), 4),
            "to_mm": round(float(start_end[hit[-1] + 1]), 4),
            "crossing_mm": round(run_length, 4),
            "center_separation_mm": item["center_separation_mm"], "overlap_mm2": item["overlap_mm2"],
        })
        crossed_masks.append(polygon_mask(item["reference_polygon"], shape))

    single_rows, missing_rows = [], []
    for item in pairing.get("reference_unmatched", []):
        run_length, inside = crossing_length(item["polygon"])
        if run_length <= 0:
            continue
        hit = np.flatnonzero(inside)
        record = {"entry": item.get("status", "Single-photo dark spot"),
                  "center_mm": item["center_mm"], "area_mm2": item["area_mm2"],
                  "from_mm": round(float(start_end[hit[0]]), 4),
                  "to_mm": round(float(start_end[hit[-1] + 1]), 4),
                  "crossing_mm": round(run_length, 4)}
        (missing_rows if record["entry"] == "Missing evidence" else single_rows).append(record)

    for mask in crossed_masks:
        tint(overlay, mask, (.20, .82, .56), .38)
    cv.line(overlay, tuple(np.rint(ends[0]).astype(int)), tuple(np.rint(ends[1]).astype(int)),
            (.98, .93, .36), 1, cv.LINE_AA)
    rows.extend(crossed_pairs)
    rows.extend(single_rows)
    rows.extend(missing_rows)
    summary = {
        "observed_mm": round(float(widths[observed].sum()), 4),
        "dark_mm": round(float(widths[dark].sum()), 4),
        "unknown_mm": round(length - float(widths[observed].sum()), 4),
        "length_mm": round(length, 4),
        "adopted_pores_crossed": len(crossed_pairs),
        "adopted_crossing_mm": round(sum(row["crossing_mm"] for row in crossed_pairs), 4),
        "single_photo_crossings": len(single_rows),
        "missing_evidence_crossings": len(missing_rows),
        "pairing_pairs_available": len(pairing["pairs"]),
    }
    return summary, rows, overlay


def landmark_offsets(frame, other, marks, _opts):
    fixed = coordinates(marks["reference"]["points"], frame.rgb.shape)
    moved = coordinates(marks["comparison"]["points"], other.rgb.shape)
    if fixed.shape != moved.shape:
        raise Invalid("Both landmark sets need the same number of centres in corresponding order.")
    difference = (fixed - moved) * frame.step
    shift = np.mean(difference, axis=0)
    offset = difference - shift
    shown = image(frame)
    rows = []
    for i, (fxy, mxy, dxy, residual) in enumerate(zip(fixed, moved, difference, offset), 1):
        start, end = tuple(np.rint(fxy).astype(int)), tuple(np.rint(mxy).astype(int))
        cv.drawMarker(shown, start, (.32, .89, .63), cv.MARKER_CROSS, 16, 2)
        cv.drawMarker(shown, end, (.98, .56, .31), cv.MARKER_TILTED_CROSS, 16, 2)
        cv.line(shown, start, end, (.76, .71, .34), 1)
        rows.append({"point": i, "reference_mm": rounded(fxy * frame.step), "comparison_mm": rounded(mxy * frame.step),
                     "displacement_mm": rounded(dxy), "residual_mm": round(float(np.linalg.norm(residual)), 4)})
    rms = np.sqrt(np.sum(offset ** 2) / len(offset))
    return {"suggested_shift_mm": rounded(shift), "rms_after_shift_mm": round(float(rms), 4)}, rows, shown


def polar_response(frame, other, marks, _opts):
    """Use two illumination witnesses to normalize an observed polarization response."""
    usable = frame.useful & other.useful
    masks = {name: region(frame, marks[name]) & usable for name in ("matte", "gloss", "decor")}
    if any(mask.sum() < 32 for mask in masks.values()):
        raise Invalid("All three marked patches need at least 32 pixels visible in both photographs.")
    delta = other.luma - frame.luma
    values = {name: float(np.median(delta[mask])) for name, mask in masks.items()}
    denominator = values["gloss"] - values["matte"]
    if abs(denominator) < .025:
        raise Invalid("Choose reference patches with distinct responses to crossed polarization.")
    shown = image(frame)
    colors = {"matte": (.91, .49, .39), "gloss": (.40, .75, .96), "decor": (.87, .66, .97)}
    rows = []
    for name, mask in masks.items():
        tint(shown, mask, colors[name], .44)
        rows.append({"patch": marks[name]["title"], "face_luma": round(float(np.median(frame.luma[mask])), 4),
                     "crossed_luma": round(float(np.median(other.luma[mask])), 4),
                     "difference": round(values[name], 4), "paired_pixels": int(mask.sum())})
    response = (values["decor"] - values["matte"]) / denominator
    return {"relative_polar_response": round(response, 4), "control_separation": round(denominator, 4)}, rows, shown


def paired_coverage(frame, other, marks, _opts):
    domain = region(frame, marks["region"])
    valid_a = domain & frame.useful
    valid_b = domain & other.useful
    both = valid_a & valid_b
    groups = [
        ("Both photos", both, (.27, .85, .59)),
        ("Reference only", valid_a & ~valid_b, (.38, .67, .99)),
        ("Comparison only", valid_b & ~valid_a, (.73, .51, .91)),
        ("Neither photo", domain & ~(valid_a | valid_b), (.94, .48, .23)),
    ]
    if not np.any(domain):
        raise Invalid("Mark a larger comparison area.")
    shown = image(frame)
    total = int(domain.sum())
    rows = []
    for label, mask, color in groups:
        tint(shown, mask, color)
        rows.append({"observation": label, "pixels": int(mask.sum()), "percent": round(100 * mask.sum() / total, 3)})
    return {"paired_percent": rows[0]["percent"], "marked_pixels": total}, rows, shown


def surface_marks(frame, marks, opts):
    area = selection(frame, marks["region"])
    contrast = number(opts.get("contrast", .075), "contrast", .02, .35)
    minimum = number(opts.get("minimum_mm2", .15), "minimum spot area", .03, 40)
    luma = frame.luma
    filled = np.where(frame.useful, luma, float(np.median(luma[area])))
    reference = cv.GaussianBlur(filled.astype(np.float32), (0, 0), 8)
    residual = filled - reference
    accumulated = np.zeros(area.shape, dtype=bool)
    shown = image(frame)
    rows = []
    for name, detected, color in (
        ("Dark", area & (residual < -contrast), (.93, .45, .27)),
        ("Light", area & (residual > contrast), (.34, .72, .95)),
    ):
        count, component, stats, centroids = cv.connectedComponentsWithStats(detected.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            size = stats[label, cv.CC_STAT_AREA] * frame.step[0] * frame.step[1]
            if size < minimum:
                continue
            selected = component == label
            accumulated |= selected
            tint(shown, selected, color)
            rows.append({"direction": name, "size_mm2": round(float(size), 4),
                         "center_mm": rounded(centroids[label] * frame.step),
                         "peak_contrast": round(float(np.abs(residual[selected]).max()), 4)})
    rows.sort(key=lambda row: row["size_mm2"], reverse=True)
    return {"spots": len(rows), "spot_fraction_pct": round(100 * accumulated.sum() / area.sum(), 3)}, rows[:50], shown


def color_drift(frame, other, marks, opts):
    domain = region(frame, marks["region"])
    area = domain & frame.useful & other.useful
    if area.sum() < 32:
        raise Invalid("Both photographs need usable pixels in this patch.")
    tolerance = number(opts.get("delta_e", 5), "color difference threshold", .5, 40)
    first = lab(frame.rgb[area])
    second = lab(other.rgb[area])
    delta = np.linalg.norm(first - second, axis=1)
    above = np.zeros(area.shape, dtype=bool)
    above[area] = delta > tolerance
    shown = image(frame)
    tint(shown, area & ~above, (.23, .81, .67), .32)
    tint(shown, above, (.96, .51, .35))
    rows = [
        {"statistic": "Median", "delta_e76": round(float(np.median(delta)), 4)},
        {"statistic": "90th percentile", "delta_e76": round(float(np.percentile(delta, 90)), 4)},
    ]
    return {"median_delta_e76": rows[0]["delta_e76"], "above_tolerance_pct": round(100 * above.sum() / area.sum(), 3),
            "paired_pixels": int(area.sum())}, rows, shown

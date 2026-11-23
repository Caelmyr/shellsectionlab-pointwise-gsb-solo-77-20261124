"""Physical-plane measurements for registered shell-section photographs."""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .model import choice, number
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


def inclusion_evidence(frame, other, marks, opts):
    """Bound the inclusion fraction by testing color suspects against the polarization pair."""
    threshold = number(opts.get("threshold", .5), "inclusion likeness threshold", .1, .9)
    minimum_paired = int(round(number(opts.get("minimum_paired", 32), "minimum paired pixels per patch", 8, 20000)))
    minimum_contrast = number(opts.get("minimum_contrast", .025), "minimum reference contrast", .005, .5)
    response_bound = number(opts.get("response_bound", .6), "significant response bound", .05, 4)
    priority = choice(opts.get("priority", "separation"), ("separation", "area"), "reference set priority")
    if len({marks[slot]["id"] for slot in ("matte_a", "gloss_a", "matte_b", "gloss_b")}) < 4:
        raise Invalid("The two reference sets need four distinct witness patches.")
    usable = frame.useful & other.useful
    delta = other.luma - frame.luma
    sets = {}
    for key in ("a", "b"):
        witnesses = {}
        for slot in ("matte", "gloss"):
            mark = marks[f"{slot}_{key}"]
            mask = region(frame, mark) & usable
            paired = int(mask.sum())
            witnesses[slot] = {
                "mark": mark, "mask": mask, "paired": paired,
                "face": float(np.median(frame.luma[mask])) if paired else None,
                "crossed": float(np.median(other.luma[mask])) if paired else None,
                "delta": float(np.median(delta[mask])) if paired else None,
            }
        matte, gloss = witnesses["matte"], witnesses["gloss"]
        contrast = None if matte["delta"] is None or gloss["delta"] is None else gloss["delta"] - matte["delta"]
        qualified = (matte["paired"] >= minimum_paired and gloss["paired"] >= minimum_paired
                     and contrast is not None and abs(contrast) >= minimum_contrast)
        sets[key] = {**witnesses, "contrast": contrast, "qualified": qualified,
                     "paired": matte["paired"] + gloss["paired"]}
    eligible = [key for key, value in sets.items() if value["qualified"]]
    if not eligible:
        raise Invalid("Neither matte/gloss reference set meets the minimum paired pixels and reference contrast.")
    if len(eligible) == 1:
        adopted, reason = eligible[0], "Only one reference set met the limits."
    elif priority == "separation":
        adopted = max(eligible, key=lambda key: abs(sets[key]["contrast"]))
        reason = "Both sets qualified; adopted for the wider matte-gloss contrast."
    else:
        adopted = max(eligible, key=lambda key: sets[key]["paired"])
        reason = "Both sets qualified; adopted for the larger paired reference area."
    chosen = sets[adopted]
    matte_delta, contrast = chosen["matte"]["delta"], chosen["contrast"]
    area = selection(frame, marks["region"])
    likeness = inclusion_projection(frame, marks["matrix"], marks["accent"])
    suspect = area & (likeness >= threshold)
    count, labels, stats, centroids = cv.connectedComponentsWithStats(suspect.astype(np.uint8), connectivity=8)
    kinds = {"evidenced": 1, "responsive": 2, "pending": 3}
    kind_of = np.zeros(count, dtype=np.int8)
    spots = []
    for label in range(1, count):
        x, y, width, height, pixels = stats[label]
        local = labels[y:y + height, x:x + width] == label
        local_paired = local & other.useful[y:y + height, x:x + width]
        paired = int(local_paired.sum())
        if paired >= minimum_paired:
            response = (float(np.median(delta[y:y + height, x:x + width][local_paired])) - matte_delta) / contrast
            kind = "responsive" if abs(response) >= response_bound else "evidenced"
        else:
            response, kind = None, "pending"
        kind_of[label] = kinds[kind]
        spots.append({"kind": kind, "pixels": int(pixels), "paired": paired, "response": response,
                      "size_mm2": float(pixels * frame.step[0] * frame.step[1]),
                      "center": centroids[label] * frame.step})
    classified = kind_of[labels]
    groups = {name: classified == code for name, code in kinds.items()}
    total = int(area.sum())
    counts = {name: int(group.sum()) for name, group in groups.items()}
    suspect_pixels = counts["evidenced"] + counts["responsive"] + counts["pending"]
    shown = image(frame)
    tint(shown, area & ~suspect, (.36, .71, .95), .2)
    tint(shown, groups["evidenced"], (.27, .79, .51))
    tint(shown, groups["responsive"], (.93, .45, .27))
    tint(shown, groups["pending"], (.95, .8, .3))
    rows = []
    for key, value in sets.items():
        for slot in ("matte", "gloss"):
            witness = value[slot]
            rows.append({
                "reference_set": key.upper(), "patch": witness["mark"]["title"], "role": slot,
                "paired_pixels": witness["paired"],
                "face_luma": round(witness["face"], 4) if witness["face"] is not None else None,
                "crossed_luma": round(witness["crossed"], 4) if witness["crossed"] is not None else None,
                "difference": round(witness["delta"], 4) if witness["delta"] is not None else None,
                "set_contrast": round(value["contrast"], 4) if value["contrast"] is not None else None,
                "set_qualified": "yes" if value["qualified"] else "no",
                "adopted": "yes" if key == adopted else "no",
            })
            amount = .5 if key == adopted else .16
            tint(shown, witness["mask"], (.91, .49, .39) if slot == "matte" else (.40, .75, .96), amount)
    for name, label in (("evidenced", "Evidenced inclusion"), ("responsive", "Polarization-responsive"),
                        ("pending", "Pending evidence")):
        rows.append({"class": label, "pixels": counts[name],
                     "percent_of_region": round(100 * counts[name] / total, 3)})
    rows.append({"class": "Suspect total (possible)", "pixels": suspect_pixels,
                 "percent_of_region": round(100 * suspect_pixels / total, 3)})
    rows.append({"class": "Base-like", "pixels": total - suspect_pixels,
                 "percent_of_region": round(100 * (total - suspect_pixels) / total, 3)})
    spots.sort(key=lambda spot: spot["pixels"], reverse=True)
    names = {"evidenced": "Evidenced inclusion", "responsive": "Polarization-responsive", "pending": "Pending evidence"}
    for index, spot in enumerate(spots[:40], 1):
        rows.append({"spot": index, "classification": names[spot["kind"]], "pixels": spot["pixels"],
                     "paired_pixels": spot["paired"], "size_mm2": round(spot["size_mm2"], 4),
                     "center_mm": rounded(spot["center"]),
                     "response": round(spot["response"], 4) if spot["response"] is not None else None})
    return {
        "adopted_reference_set": adopted.upper(), "reference_priority": priority, "adopted_reason": reason,
        "control_contrast": round(contrast, 4),
        "inclusion_confirmed_pct": round(100 * counts["evidenced"] / total, 3),
        "inclusion_possible_pct": round(100 * suspect_pixels / total, 3),
        "polarization_responsive_pct": round(100 * counts["responsive"] / total, 3),
        "pending_evidence_pct": round(100 * counts["pending"] / total, 3),
        "inspection_pixels": total,
        "region_paired_pct": round(100 * int((area & other.useful).sum()) / total, 3),
        "candidate_spots": len(spots), "threshold": threshold,
    }, rows, shown


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


def pore_map(frame, marks, opts):
    """Separate compact pore candidates from elongated dark features."""
    area = selection(frame, marks["region"])
    cutoff = number(opts.get("luma", .31), "dark pore threshold", .05, .95)
    candidates = area & (frame.luma <= cutoff)
    count, labels, stats, centroids = cv.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    pixel_area = frame.step[0] * frame.step[1]
    rows = []
    pits = np.zeros(area.shape, dtype=bool)
    limited = cv.dilate((~area).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
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
        rows.append({"area_mm2": round(area_mm2, 4), "center_mm": rounded(centroids[label] * frame.step),
                     "width_mm": round(horizontal, 4), "height_mm": round(vertical, 4),
                     "boundary_limited": bool(np.any(local & limited[y:y + height, x:x + width])),
                     "circularity": round(min(1., 4 * math.pi * area_mm2 / perimeter ** 2), 4) if perimeter else None})
    rows.sort(key=lambda item: item["area_mm2"], reverse=True)
    shown = image(frame)
    tint(shown, area & ~candidates, (.25, .78, .73), .22)
    tint(shown, candidates & ~pits, (.65, .36, .83), .44)
    tint(shown, pits, (.95, .48, .24))
    return {"pore_candidate_count": len(rows), "candidate_area_mm2": round(float(pits.sum() * pixel_area), 4),
            "observed_mm2": round(float(area.sum() * pixel_area), 4)}, rows[:40], shown


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

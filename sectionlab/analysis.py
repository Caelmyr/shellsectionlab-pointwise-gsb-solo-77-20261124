"""Physical-plane measurements for registered shell-section photographs."""
import math

import cv2 as cv
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

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


def lamina_spacing(frame, marks, _opts, edge=None):
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
    summary = {"dominant_spacing_mm": rows[0]["spacing_mm"], "dominant_lamina_deg": rows[0]["lamina_direction_deg"],
               "peaks": len(rows)}
    shown = image(frame)
    tint(shown, mask, (.52, .79, .91), .22)
    if edge is not None:
        summary.update(light_side_pair(frame, mask, edge, rows[0]["spacing_mm"], shown))
    return summary, rows, shown


def light_side_pair(frame, mask, edge, spacing, shown):
    """Verify the first continuously readable lamina pair past the adopted growth edge.

    Only the transect interval from the adopted 90% position toward the light
    side is read. If the first two lamina crests are separated by an unreadable
    segment, the pair is reported missing instead of borrowing laminae from
    elsewhere or the patch-wide spacing.
    """
    knots = coordinates(edge["points"], frame.rgb.shape)
    full = float(np.linalg.norm((knots[1] - knots[0]) * frame.step))
    origin = min(max(edge["origin_mm"], 0.), full)
    forward = edge["light_side"] == "end"
    span = full - origin if forward else origin
    if span < .5:
        raise Invalid("The adopted edge leaves no readable length on the light side of the transect.")
    count = min(2400, max(20, int(np.ceil(span / min(frame.step))) + 1))
    ts = np.linspace(origin, full if forward else 0., count)
    step_mm = span / (count - 1)
    points = knots[0] + (ts / full)[:, None] * (knots[1] - knots[0])
    xs = points[:, 0].reshape(1, -1).astype(np.float32)
    ys = points[:, 1].reshape(1, -1).astype(np.float32)
    brightness = cv.remap(frame.luma, xs, ys, cv.INTER_LINEAR)[0]
    observed = cv.remap(frame.useful.astype(np.float32), xs, ys, cv.INTER_LINEAR)[0] >= .999
    inside = cv.remap(mask.astype(np.float32), xs, ys, cv.INTER_LINEAR)[0] >= .999
    readable = observed & inside
    if not readable[0]:
        raise Invalid("The marked lamina patch must cover the transect at the adopted edge.")
    smoothed = np.convolve(np.pad(brightness, 2, mode="edge"), np.full(5, .2), mode="valid")
    window = max(9, int(round(2.5 * spacing / step_mm)) | 1)
    distance = max(2, int(round(.45 * spacing / step_mm)))
    peaks = []
    boundaries = np.r_[0, np.flatnonzero(readable[1:] != readable[:-1]) + 1, count]
    for left, right in zip(boundaries, boundaries[1:]):
        if not readable[left] or right - left < 5:
            continue
        piece = smoothed[left:right]
        detrended = piece - uniform_filter1d(piece, size=min(window, piece.size))
        spread = float(np.percentile(detrended, 90) - np.percentile(detrended, 10))
        local, _ = find_peaks(detrended, prominence=max(.01, .22 * spread), distance=distance)
        peaks.extend((left + local).tolist())
    result = {"edge_reading": edge["id"], "edge_origin_mm": round(float(origin), 3),
              "light_side": edge["light_side"]}
    first = second = None
    note = None
    if len(peaks) >= 2:
        first, second = peaks[0], peaks[1]
        if not readable[first:second + 1].all():
            first = second = None
            note = "An unreadable segment separates the first two laminae."
    else:
        note = "Fewer than two laminae are readable along the light side."
    if first is None:
        result.update(first_lamina_mm=None, second_lamina_mm=None, pair_spacing_mm=None, pair_note=note)
    result["pair_status"] = "missing" if first is None else "verified"

    def spot(at_mm):
        point = knots[0] + (at_mm / full) * (knots[1] - knots[0])
        return tuple(np.rint(point).astype(int))

    cv.line(shown, spot(origin), spot(float(ts[-1])), (.96, .52, .27), 2)
    cv.drawMarker(shown, spot(origin), (.95, .38, .60), cv.MARKER_TILTED_CROSS, 15, 2)
    if first is not None:
        d1 = float(abs(ts[first] - origin))
        d2 = float(abs(ts[second] - origin))
        result.update(first_lamina_mm=round(d1, 3), second_lamina_mm=round(d2, 3),
                      pair_spacing_mm=round(d2 - d1, 4))
        for at in (float(ts[first]), float(ts[second])):
            cv.drawMarker(shown, spot(at), (.30, .88, .52), cv.MARKER_CROSS, 12, 2)
    return result


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


def rise_episodes(profile, level_tol, drop_tol, stall):
    """Split an upward-normalized profile into separate continuous rise episodes.

    An episode ends when the profile gives back more than ``drop_tol`` from its
    running high, or when it stalls near that high for more than ``stall``
    samples before climbing again (a shelf between two rises).
    """
    runs = []
    start = high = 0
    since = 0
    for i in range(1, len(profile)):
        if profile[i] >= profile[high] + level_tol:
            if since > stall:
                runs.append((start, high))
                start = high
            high = i
            since = 0
        elif profile[i] < profile[high] - drop_tol:
            runs.append((start, high))
            start = high
            high = i
            since = 0
        else:
            since += 1
    runs.append((start, high))
    episodes = []
    for first, last in runs:
        if last <= first:
            continue
        trough = first + int(np.argmin(profile[first:last + 1]))
        if trough < last:
            episodes.append((trough, last))
    return episodes


def climbs_steadily(profile, i0, i1, level_tol, stall):
    """True when [i0, i1] keeps climbing without a halt longer than ``stall``."""
    high = i0
    since = 0
    for i in range(i0 + 1, i1 + 1):
        if profile[i] >= profile[high] + level_tol:
            high = i
            since = 0
        else:
            since += 1
            if since > stall:
                return False
    return True


def growth_transition(frame, marks, opts):
    knots = coordinates(marks["line"]["points"], frame.rgb.shape)
    length = float(np.linalg.norm((knots[1] - knots[0]) * frame.step))
    if length < 1:
        raise Invalid("Use a transect at least 1 mm long.")
    contrast = number(opts.get("contrast", .07), "local transition contrast", .02, .6)
    width_min = number(opts.get("width_min", .3), "minimum transition width", .05, 500)
    width_max = number(opts.get("width_max", 30), "maximum transition width", .1, 1500)
    if width_min >= width_max:
        raise Invalid("The minimum transition width must stay below the maximum.")
    prefer = choice(opts.get("prefer", "narrow"), ("narrow", "contrast"), "edge adoption preference")
    count = min(600, max(24, int(length * 5)))
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
    sign = 1. if tail >= lead else -1.
    profile = np.convolve(np.pad((luma - lead) * sign, 2, mode="edge"), np.full(5, .2), mode="valid")
    position = np.linspace(0, length, count)
    per_mm = (count - 1) / length
    level_tol = max(.01, .25 * contrast)
    # A shelf is a halt longer than the gentlest acceptable rise (contrast over
    # width_max) needs to climb level_tol; shorter halts stay inside one rise.
    stall = max(4, min(240, int(round(level_tol * width_max * per_mm / (.8 * contrast)))))
    episodes = rise_episodes(profile, level_tol, max(.04, .5 * contrast), stall)
    window = max(4, min(40, int(round(.8 * width_min * per_mm))))
    least = max(3, window // 2)
    found = []
    for trough, high in episodes:
        amplitude = float(profile[high] - profile[trough])
        if amplitude < .5 * contrast:
            continue
        before = profile[max(0, trough - window):trough + 1]
        after = profile[high:high + window + 1]
        clear = (len(before) >= least and len(after) >= least
                 and valid[max(0, trough - window):trough + 1].all() and valid[high:high + window + 1].all()
                 and np.percentile(before, 90) - np.percentile(before, 10) <= max(.04, .45 * amplitude)
                 and np.percentile(after, 90) - np.percentile(after, 10) <= max(.04, .45 * amplitude))
        dark = float(np.median(before)) if clear else float(profile[trough])
        light = float(np.median(after)) if clear else float(profile[high])
        seen = light - dark
        rise10, rise90 = trough, high
        continuous = False
        if seen > 0:
            low, top = dark + np.array((.1, .9)) * seen
            segment = profile[trough:high + 1]
            rise90 = trough + int(np.flatnonzero(segment >= top)[0])
            below = np.flatnonzero(profile[trough:rise90 + 1] <= low)
            rise10 = trough + (int(below[-1]) + 1 if len(below) else 0)
            continuous = (bool(valid[rise10:rise90 + 1].all())
                          and climbs_steadily(profile, rise10, rise90, level_tol, stall))
        width = float(position[rise90] - position[rise10])
        if not clear:
            status = "plateaus unclear"
        elif seen < contrast:
            status = "below contrast limit"
        elif not continuous:
            status = "interrupted rise"
        elif not width_min <= width <= width_max:
            status = "width outside range"
        else:
            status = "usable"
        found.append({"transition": len(found) + 1, "from_mm": round(float(position[trough]), 3),
                      "to_mm": round(float(position[high]), 3), "rise_10_mm": round(float(position[rise10]), 3),
                      "rise_90_mm": round(float(position[rise90]), 3), "width_mm": round(width, 4),
                      "contrast": round(float(seen), 4), "plateaus": "clear" if clear else "unclear",
                      "status": status})
    usable = [item for item in found if item["status"] == "usable"]
    if not usable:
        raise Invalid("No local transition on this transect met the contrast and width limits.")
    if prefer == "narrow":
        adopted = min(usable, key=lambda item: (item["width_mm"], -item["contrast"]))
    else:
        adopted = max(usable, key=lambda item: (item["contrast"], -item["width_mm"]))
    adopted["status"] = "adopted"
    shown = image(frame)
    cv.line(shown, tuple(np.rint(knots[0]).astype(int)), tuple(np.rint(knots[1]).astype(int)), (.96, .52, .27), 2)

    def spot(at_mm):
        point = knots[0] + (at_mm / length) * (knots[1] - knots[0])
        return tuple(np.rint(point).astype(int))

    for item in found:
        tone = (.30, .72, .95) if item["status"] in ("usable", "adopted") else (.58, .58, .63)
        cv.circle(shown, spot(item["rise_90_mm"]), 4, tone, 1, cv.LINE_AA)
    cv.drawMarker(shown, spot(adopted["rise_10_mm"]), (.32, .88, .55), cv.MARKER_CROSS, 12, 2)
    cv.drawMarker(shown, spot(adopted["rise_90_mm"]), (.95, .38, .60), cv.MARKER_TILTED_CROSS, 15, 2)
    summary = {"edge_width_mm": adopted["width_mm"], "edge_origin_mm": adopted["rise_90_mm"],
               "edge_contrast": adopted["contrast"], "edge_preference": prefer,
               "transitions_found": len(found), "transitions_usable": len(usable),
               "light_side": "end" if sign > 0 else "start", "length_mm": round(length, 4),
               "usable_samples": int(valid.sum())}
    return summary, found, shown


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

"""Paired landmark adjustments and registered color difference on corrected geometry.

The four-corner registration (``photos.plane``) is never modified here.  A saved
landmark-adjust reading derives an optional, pair-local correction from matched
growth landmarks:

* a rigid fit (rotation and translation only, scale fixed at 1) that keeps the
  registered millimetre scale, and
* a similarity fit that adds one uniform scale factor.

Only the adopted correction is later applied, and only to readings that
explicitly reference the saved adjustment record.
"""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .model import number
from .photos import coordinates, image, region


# Audit samples are kept small enough for stored JSON; summary totals are exact.
MAX_PIXEL_ROWS = 9000


def lab(pixels):
    # Display-referred Lab, matching sectionlab.analysis.lab.
    return cv.cvtColor(np.asarray(pixels, np.float32).reshape(-1, 1, 3), cv.COLOR_RGB2LAB).reshape(-1, 3)


def rounded(values, digits=4):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def tint(shown, mask, color, amount=0.42):
    shown[mask] = shown[mask] * (1 - amount) + np.asarray(color) * amount


def _rotation_angle(matrix):
    return round(float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))), 4)


def _procrustes(reference_mm, comparison_mm, allow_scale):
    """Least-squares rigid / uniform-similarity fit, comparison -> reference."""
    ref_center = reference_mm.mean(axis=0)
    cmp_center = comparison_mm.mean(axis=0)
    ref_centered = reference_mm - ref_center
    cmp_centered = comparison_mm - cmp_center
    # Orthogonal Procrustes: R maximises trace(R C^T X), with det(R) = +1.
    u, _, vt = np.linalg.svd(cmp_centered.T @ ref_centered)
    correction = np.eye(2)
    correction[1, 1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    if allow_scale:
        scale = float(np.trace(rotation @ (cmp_centered.T @ ref_centered)) / np.sum(cmp_centered ** 2))
        if not math.isfinite(scale) or scale <= 0:
            raise Invalid("The matched landmarks do not support a stable scale fit.")
    else:
        scale = 1.0
    linear = scale * rotation
    translation = ref_center - linear @ cmp_center
    predicted = (linear @ comparison_mm.T).T + translation
    residuals_mm = np.linalg.norm(predicted - reference_mm, axis=1)
    return linear, rotation, scale, translation, predicted, residuals_mm


def _pixel_matrix(linear_mm, translation_mm, step):
    """Convert a millimetre-plane transform to the shared calibrated pixel grid."""
    spacing = np.diag(np.asarray(step, dtype=float))
    linear_px = np.linalg.inv(spacing) @ linear_mm @ spacing
    translation_px = np.linalg.inv(spacing) @ translation_mm
    matrix = np.eye(3)
    matrix[:2, :2] = linear_px
    matrix[:2, 2] = translation_px
    return matrix


def _candidate_record(name, step, rotation, scale, translation, predicted_mm,
                      residuals_mm, max_residual_mm, max_scale_pct):
    scale_change_pct = round(abs(scale - 1.0) * 100.0, 4)
    residuals_within = bool(float(residuals_mm.max()) <= max_residual_mm)
    scale_within = bool(scale_change_pct <= max_scale_pct + 1e-12)
    qualified = residuals_within and scale_within
    matrix = _pixel_matrix(scale * rotation, translation, step)
    predicted_px = (np.linalg.inv(np.diag(np.asarray(step, float))) @ predicted_mm.T).T
    return {
        "fit": name,
        "qualified": qualified,
        "residuals_within_limit": residuals_within,
        "scale_within_limit": scale_within,
        "scale": round(float(scale), 6),
        "scale_change_pct": scale_change_pct,
        "rotation_deg": _rotation_angle(rotation),
        "translation_mm": rounded(translation),
        "max_residual_mm": round(float(residuals_mm.max()), 4),
        "rms_residual_mm": round(float(np.sqrt(np.mean(residuals_mm ** 2))), 4),
        "matrix_px": np.round(matrix, 8).tolist(),
        "_predicted_px": predicted_px,
    }


def landmark_adjust(frame, other, marks, opts):
    max_residual_mm = number(opts.get("max_residual_mm", 1.0), "maximum remaining landmark deviation", 0.01, 25)
    max_scale_pct = number(opts.get("max_scale_pct", 2.0), "allowed uniform scale change", 0, 25)
    pixel_ref = coordinates(marks["reference"]["points"], frame.rgb.shape)
    pixel_cmp = coordinates(marks["comparison"]["points"], other.rgb.shape)
    if pixel_ref.shape != pixel_cmp.shape or len(pixel_ref) < 2:
        raise Invalid("Both landmark sets need at least two centres in corresponding order.")
    step = np.asarray(frame.step, dtype=float)
    reference_mm = pixel_ref * step
    comparison_mm = pixel_cmp * step

    rigid = _procrustes(reference_mm, comparison_mm, allow_scale=False)
    similarity = _procrustes(reference_mm, comparison_mm, allow_scale=True)
    candidates = []
    for name, result in (("rigid", rigid), ("similarity", similarity)):
        candidates.append(_candidate_record(
            name, frame.step, result[1], result[2], result[3], result[4], result[5],
            max_residual_mm, max_scale_pct,
        ))
    rigid_rec, similarity_rec = candidates

    if rigid_rec["qualified"]:
        adopted = "rigid"
        reason = ("Rigid rotation and translation keep the four-corner millimetre scale "
                  "(scale 1.000) while every landmark residual stays within the limit.")
    elif similarity_rec["qualified"]:
        adopted = "similarity"
        reason = ("The rigid fit leaves a landmark beyond the residual limit; the uniform-scale "
                  "fit stays within both the residual and scale limits and reduces node residuals.")
    else:
        adopted = None
        failed = []
        for record in candidates:
            problems = []
            if not record["residuals_within_limit"]:
                problems.append(f"max residual {record['max_residual_mm']} mm > {max_residual_mm:g} mm")
            if not record["scale_within_limit"]:
                problems.append(f"scale change {record['scale_change_pct']}% > {max_scale_pct:g}%")
            failed.append(record["fit"] + " (" + "; ".join(problems) + ")")
        reason = "No candidate is acceptable: " + " and ".join(failed) + "."

    rows = []
    for index in range(len(pixel_ref)):
        rows.append({
            "point": index + 1,
            "reference_mm": rounded(reference_mm[index]),
            "comparison_mm": rounded(comparison_mm[index]),
            "rigid_predicted_mm": rounded(rigid[4][index]),
            "rigid_residual_mm": round(float(rigid[5][index]), 4),
            "similarity_predicted_mm": rounded(similarity[4][index]),
            "similarity_residual_mm": round(float(similarity[5][index]), 4),
        })

    shown = image(frame)
    adopted_record = rigid_rec if adopted == "rigid" else similarity_rec if adopted == "similarity" else None
    for index, (ref_px, cmp_px) in enumerate(zip(pixel_ref, pixel_cmp), 1):
        start = tuple(np.rint(ref_px).astype(int))
        moved = tuple(np.rint(cmp_px).astype(int))
        cv.drawMarker(shown, start, (0.32, 0.89, 0.63), cv.MARKER_CROSS, 16, 2)
        cv.drawMarker(shown, moved, (0.98, 0.56, 0.31), cv.MARKER_TILTED_CROSS, 16, 2)
        cv.line(shown, start, moved, (0.76, 0.71, 0.34), 1)
        if adopted_record is not None:
            predicted = tuple(np.rint(adopted_record["_predicted_px"][index - 1]).astype(int))
            cv.circle(shown, predicted, 5, (0.35, 0.62, 0.98), 2, cv.LINE_AA)
        # Offset a small square from each reference node to carry its order number visually.
        cv.drawMarker(shown, (start[0] + 10, start[1] - 10), (0.95, 0.95, 0.95),
                      cv.MARKER_SQUARE, 7, 1, cv.LINE_AA)
    # Banner: green stripe marks an adopted correction, red marks that no fit qualified.
    banner_color = (0.25, 0.75, 0.45) if adopted is not None else (0.85, 0.3, 0.3)
    cv.rectangle(shown, (8, 8), (34, 34), banner_color, -1, cv.LINE_AA)
    cv.rectangle(shown, (8, 8), (34, 34), (0.95, 0.95, 0.95), 1, cv.LINE_AA)

    def public(record):
        return {key: value for key, value in record.items() if not key.startswith("_")}

    summary = {
        "max_residual_limit_mm": max_residual_mm,
        "max_scale_limit_pct": max_scale_pct,
        "adopted_fit": adopted if adopted is not None else "none",
        "adopted_reason": reason,
        "rigid_qualified": rigid_rec["qualified"],
        "rigid_rotation_deg": rigid_rec["rotation_deg"],
        "rigid_translation_mm": rigid_rec["translation_mm"],
        "rigid_max_residual_mm": rigid_rec["max_residual_mm"],
        "rigid_rms_residual_mm": rigid_rec["rms_residual_mm"],
        "similarity_qualified": similarity_rec["qualified"],
        "similarity_scale": similarity_rec["scale"],
        "similarity_scale_change_pct": similarity_rec["scale_change_pct"],
        "similarity_rotation_deg": similarity_rec["rotation_deg"],
        "similarity_translation_mm": similarity_rec["translation_mm"],
        "similarity_max_residual_mm": similarity_rec["max_residual_mm"],
        "similarity_rms_residual_mm": similarity_rec["rms_residual_mm"],
    }
    correction = {
        "thresholds": {"max_residual_mm": max_residual_mm, "max_scale_pct": max_scale_pct},
        "frame": {"width_px": int(frame.rgb.shape[1]), "height_px": int(frame.rgb.shape[0]),
                  "step_mm": rounded(frame.step, 6)},
        "landmark_count": int(len(pixel_ref)),
        "candidates": {"rigid": public(rigid_rec), "similarity": public(similarity_rec)},
        "adopted": adopted,
        "reason": reason,
    }
    if adopted is not None:
        correction["matrix_px"] = adopted_record["matrix_px"]
    return summary, rows, shown, correction


def load_correction(record, source_photo, other_photo, source_frame, other_frame):
    """Validate a saved landmark-adjust reading before it can drive another reading."""
    if record.get("mode") != "landmark-adjust":
        raise Invalid("Select a paired landmark adjustment reading.")
    saved_photos = record.get("photos", [])
    if [item.get("id") for item in saved_photos[:2]] != [source_photo["id"], other_photo["id"]]:
        raise Invalid("That adjustment was fitted to a different photograph pair or order.")
    correction = record.get("correction", {})
    adopted = correction.get("adopted")
    if not adopted:
        raise Invalid("That adjustment produced no qualified fit, so it cannot correct a reading.")
    frame_info = correction.get("frame", {})
    if frame_info.get("width_px") != source_frame.rgb.shape[1] or frame_info.get("height_px") != source_frame.rgb.shape[0]:
        raise Invalid("The calibrated frame changed after the adjustment was saved.")
    if other_frame.rgb.shape[:2] != source_frame.rgb.shape[:2]:
        raise Invalid("The two photographs need matching calibrated frame dimensions.")
    return correction


def _warp_comparison(other_frame, matrix_px, shape):
    height, width = shape[:2]
    affine = np.asarray(matrix_px, dtype=np.float64)[:2]
    warped_rgb = cv.warpAffine(other_frame.rgb, affine, (width, height), flags=cv.INTER_LINEAR,
                               borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_support = cv.warpAffine(other_frame.support.astype(np.uint8), affine, (width, height),
                                   flags=cv.INTER_NEAREST, borderMode=0).astype(bool)
    warped_useful = warped_support & (warped_rgb.max(axis=2) >= 0.018) & (warped_rgb.min(axis=2) < 0.993)
    return warped_rgb, warped_useful


def corrected_color_drift(frame, other, marks, opts):
    """Registered color difference with comparison pixels sampled through the correction."""
    correction = opts.get("_correction")
    if not correction or "matrix_px" not in correction:
        raise Invalid("A saved paired landmark adjustment is required for corrected color difference.")
    tolerance = number(opts.get("delta_e", 5), "color difference threshold", 0.5, 40)
    matrix_px = correction["matrix_px"]
    fit_name = correction.get("adopted")
    candidate = correction["candidates"][fit_name]
    warped_rgb, warped_useful = _warp_comparison(other, matrix_px, frame.rgb.shape)

    domain = region(frame, marks["region"])
    if not np.any(domain):
        raise Invalid("Mark a larger comparison window.")
    comparison_good = warped_useful
    paired = domain & frame.useful & comparison_good
    missing_cmp = domain & ~comparison_good
    missing_ref = domain & ~frame.useful
    missing_both = missing_cmp & missing_ref
    missing_cmp_only = missing_cmp & frame.useful
    missing_ref_only = missing_ref & comparison_good
    window_pixels = int(domain.sum())
    if paired.sum() == 0:
        raise Invalid("No position in this window has usable pixels in both photographs; nothing to compare.")

    reference_lab = lab(frame.rgb[paired])
    comparison_lab = lab(warped_rgb[paired])
    delta = np.linalg.norm(reference_lab - comparison_lab, axis=1)
    above = np.zeros(domain.shape, dtype=bool)
    above[paired] = delta > tolerance

    stride = max(1, int(math.ceil(math.sqrt(window_pixels / MAX_PIXEL_ROWS))))
    ys, xs = np.nonzero(domain)
    sampled = (ys % stride == 0) & (xs % stride == 0)
    delta_map = np.full(domain.shape, np.nan, dtype=np.float32)
    delta_map[paired] = delta
    rows = []
    for y, x in zip(ys[sampled], xs[sampled]):
        if paired[y, x]:
            status = "Paired"
            value = round(float(delta_map[y, x]), 4)
        elif missing_both[y, x]:
            status = "Missing in both"
            value = None
        elif missing_cmp_only[y, x]:
            status = "Missing in comparison"
            value = None
        else:
            status = "Missing in reference"
            value = None
        rows.append({
            "x_mm": round(float(x * frame.step[0]), 4),
            "y_mm": round(float(y * frame.step[1]), 4),
            "observation": status,
            "delta_e76": value,
        })

    shown = image(frame)
    tint(shown, paired & ~above, (0.23, 0.81, 0.67), 0.32)
    tint(shown, paired & above, (0.96, 0.51, 0.35), 0.45)
    tint(shown, missing_cmp_only, (0.86, 0.27, 0.62), 0.55)
    tint(shown, missing_ref_only, (0.45, 0.48, 0.72), 0.5)
    tint(shown, missing_both, (0.62, 0.2, 0.3), 0.6)
    contours, _ = cv.findContours(missing_cmp.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(shown, contours, -1, (1.0, 0.85, 0.35), 1, cv.LINE_AA)
    outline, _ = cv.findContours(domain.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(shown, outline, -1, (0.95, 0.95, 0.95), 1, cv.LINE_AA)
    # Corner badge records that the corrected geometry was used.
    cv.rectangle(shown, (8, 8), (34, 34), (0.25, 0.75, 0.45), -1, cv.LINE_AA)
    cv.rectangle(shown, (8, 8), (34, 34), (0.95, 0.95, 0.95), 1, cv.LINE_AA)

    paired_count = int(paired.sum())
    summary = {
        "median_delta_e76": round(float(np.median(delta)), 4),
        "p90_delta_e76": round(float(np.percentile(delta, 90)), 4),
        "delta_e_tolerance": tolerance,
        "above_tolerance_pct": round(100.0 * float(above.sum()) / paired_count, 3),
        "window_pixels": window_pixels,
        "paired_pixels": paired_count,
        "missing_evidence_pixels": int(missing_cmp.sum()),
        "missing_evidence_pct": round(100.0 * float(missing_cmp.sum()) / window_pixels, 3),
        "missing_in_reference_pixels": int(missing_ref_only.sum()),
        "missing_in_both_pixels": int(missing_both.sum()),
        "limited_evidence": paired_count < 32,
        "sample_stride_px": stride,
        "adjustment_fit": fit_name,
        "adjustment_scale": candidate["scale"],
        "adjustment_rotation_deg": candidate["rotation_deg"],
    }
    return summary, rows, shown

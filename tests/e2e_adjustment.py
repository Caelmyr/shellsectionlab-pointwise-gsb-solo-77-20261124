"""End-to-end checks for paired landmark adjustment and corrected color difference."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")
from sectionlab.__main__ import application  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("ok -", message)


def main():
    with tempfile.TemporaryDirectory() as data_dir:
        app = application(Path(data_dir))
        client = app.test_client()

        config = client.get("/section/config").get_json()
        check("landmark-adjust" in config["readings"], "landmark-adjust mode is advertised")
        check("corrected-color-drift" in config["readings"], "corrected-color-drift mode is advertised")

        demo = client.post("/section/demo").get_json()
        run_id = demo["id"]
        revision = demo["revision"]
        run = client.get(f"/section/runs/{run_id}").get_json()
        face = next(p for p in run["photos"] if p["role"] == "face")
        cross = next(p for p in run["photos"] if p["role"] == "cross")

        def register_marks(photo):
            return [m for m in run["marks"] if m["photo"] == photo["id"] and m["purpose"] == "register"]

        face_marks = register_marks(face)
        cross_marks = register_marks(cross)
        nodes_face = next(m for m in face_marks if m["title"] == "Four matched growth nodes")
        nodes_cross = next(m for m in cross_marks if m["title"] == "Four matched growth nodes")
        check(len(nodes_face["points"]) == 4 and len(nodes_cross["points"]) == 4,
              "demo carries four matched growth nodes on each photograph")

        # 1. Landmark adjustment with permissive limits: pure translation truth -> rigid adopted.
        payload = {
            "revision": revision, "mode": "landmark-adjust", "photo": face["id"], "other": cross["id"],
            "marks": {"reference": nodes_face["id"], "comparison": nodes_cross["id"]},
            "options": {"max_residual_mm": 1.0, "max_scale_pct": 2.0},
        }
        response = client.post(f"/section/runs/{run_id}/readings", json=payload)
        check(response.status_code == 201, f"adjustment reading saved ({response.status_code})")
        body = response.get_json()
        reading = body["reading"]
        run = body["run"]
        revision = run["revision"]
        summary = reading["summary"]
        check(summary["adopted_fit"] == "rigid", f"rigid fit adopted, got {summary['adopted_fit']}")
        check("millimetre scale" in summary["adopted_reason"], "adoption reason preserves mm scale")
        check(len(reading["rows"]) == 4, "per-landmark rows stored for all four nodes")
        for row in reading["rows"]:
            check(row["rigid_residual_mm"] <= summary["max_residual_limit_mm"],
                  f"node {row['point']} rigid residual within limit")
            check(row["similarity_residual_mm"] <= summary["max_residual_limit_mm"],
                  f"node {row['point']} similarity residual within limit")
        correction = reading["correction"]
        check(correction["adopted"] == "rigid" and len(correction["matrix_px"]) == 3,
              "adopted correction with 3x3 pixel matrix saved")
        check(abs(correction["candidates"]["rigid"]["scale"] - 1.0) < 1e-9,
              "rigid scale is exactly 1.000 (registered mm scale retained)")
        check(abs(correction["candidates"]["similarity"]["scale"] - 1.0) < 0.02,
              "similarity fit also stays near unit scale for translated demo")
        rigid = correction["candidates"]["rigid"]
        check(rigid["residuals_within_limit"] and rigid["scale_within_limit"],
              "rigid candidate flags both limits as satisfied")

        # Re-derive residuals straight from the saved matrix and saved landmark coordinates.
        matrix = np.asarray(correction["matrix_px"])
        frame = correction["frame"]
        px = np.array([[p[0] * (frame["width_px"] - 1), p[1] * (frame["height_px"] - 1)]
                       for p in nodes_cross["points"]])
        ref_px = np.array([[p[0] * (frame["width_px"] - 1), p[1] * (frame["height_px"] - 1)]
                           for p in nodes_face["points"]])
        predicted = (matrix[:2, :2] @ px.T).T + matrix[:2, 2]
        residual_px = np.linalg.norm(predicted - ref_px, axis=1)
        residual_mm = residual_px * frame["step_mm"][0]
        saved = np.array([row["rigid_residual_mm"] for row in reading["rows"]])
        check(np.allclose(residual_mm, saved, atol=5e-4),
              "saved per-point residuals reproduce from saved matrix + landmark readings")

        overlay = client.get(f"/section/runs/{run_id}/readings/{reading['id']}/overlay")
        check(overlay.status_code == 200 and overlay.data[:8] == b"\x89PNG\r\n\x1a\n",
              "adjustment overlay PNG is available")

        # 2. Perturbed marks beyond the tight limits -> no qualified result.
        perturbed_points = [
            [x + (0.012 if i % 2 else -0.012), y + 0.009]
            for i, (x, y) in enumerate(nodes_cross["points"])
        ]
        made = client.post(f"/section/runs/{run_id}/marks", json={
            "title": "Perturbed growth nodes", "purpose": "register", "photo": cross["id"],
            "points": perturbed_points, "revision": revision,
        })
        check(made.status_code == 201, "perturbed comparison landmark mark saved")
        revision = made.get_json()["revision"]
        perturbed_id = made.get_json()["marks"][-1]["id"]
        payload_fail = dict(payload, revision=revision,
                            marks={"reference": nodes_face["id"], "comparison": perturbed_id},
                            options={"max_residual_mm": 0.2, "max_scale_pct": 0.1})
        response = client.post(f"/section/runs/{run_id}/readings", json=payload_fail)
        check(response.status_code == 201, "unqualified adjustment still saves a record")
        fail_reading = response.get_json()["reading"]
        revision = response.get_json()["run"]["revision"]
        check(fail_reading["summary"]["adopted_fit"] == "none", "no fit adopted when limits are exceeded")
        check("matrix_px" not in fail_reading["correction"], "no correction matrix when nothing qualified")
        check("No candidate is acceptable" in fail_reading["summary"]["adopted_reason"],
              "failure reason names both candidates and violated limits")
        check(fail_reading["summary"]["rigid_max_residual_mm"] > 0.2,
              "saved rigid max residual documents why the limit failed")

        # 2b. Rotated + uniformly scaled comparison nodes: rigid fails, similarity passes.
        # Build the geometry in calibrated millimetres, then convert to normalized marks.
        sheet_w, sheet_h = run["width_mm"], run["height_mm"]
        angle = math.radians(4.0)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        scale = 1.018
        spread_mm = np.array([[60, 45], [230, 40], [245, 135], [70, 145]], dtype=float)
        center_mm = spread_mm.mean(axis=0)
        scaled_mm = (rot @ (spread_mm - center_mm).T).T / scale + center_mm
        spread_norm = spread_mm / [sheet_w, sheet_h]
        scaled_norm = scaled_mm / [sheet_w, sheet_h]
        made = client.post(f"/section/runs/{run_id}/marks", json={
            "title": "Spread reference nodes", "purpose": "register", "photo": face["id"],
            "points": spread_norm.tolist(), "revision": revision,
        })
        check(made.status_code == 201, "spread reference landmark mark saved")
        revision = made.get_json()["revision"]
        spread_id = made.get_json()["marks"][-1]["id"]
        made = client.post(f"/section/runs/{run_id}/marks", json={
            "title": "Scaled rotated growth nodes", "purpose": "register", "photo": cross["id"],
            "points": scaled_norm.tolist(), "revision": revision,
        })
        check(made.status_code == 201, "scaled/rotated comparison landmark mark saved")
        revision = made.get_json()["revision"]
        scaled_id = made.get_json()["marks"][-1]["id"]
        # Threshold sits between the rigid residual and the near-zero similarity residual.
        payload_scaled = dict(payload, revision=revision,
                              marks={"reference": spread_id, "comparison": scaled_id},
                              options={"max_residual_mm": 1.0, "max_scale_pct": 3.0})
        response = client.post(f"/section/runs/{run_id}/readings", json=payload_scaled)
        check(response.status_code == 201, "similarity-only adjustment record saved")
        scaled_reading = response.get_json()["reading"]
        revision = response.get_json()["run"]["revision"]
        ss = scaled_reading["summary"]
        check(ss["adopted_fit"] == "similarity",
              f"similarity adopted when rigid residual exceeds limit (got {ss['adopted_fit']})")
        check(not ss["rigid_qualified"] and ss["similarity_qualified"],
              "candidate qualification flags distinguish the two fits")
        check(abs(ss["similarity_scale"] - scale) < 0.002,
              f"fitted uniform scale recovers the constructed scale (got {ss['similarity_scale']})")
        check(ss["similarity_max_residual_mm"] < 0.05,
              "well-conditioned similarity fit drives node residuals near zero")
        check(ss["similarity_max_residual_mm"] < ss["rigid_max_residual_mm"],
              "similarity fit shrinks node residuals below the rigid fit")
        check("reduces node residuals" in ss["adopted_reason"],
              "adoption reason states similarity was chosen to shrink node residuals")
        check(scaled_reading["correction"]["matrix_px"] is not None,
              "similarity correction matrix stored for downstream use")

        # 3. Corrected color difference requires an adopted adjustment.
        inspect_mark = next(m for m in run["marks"]
                            if m["photo"] == face["id"] and m["title"] == "Discolored outer wall")
        drift_payload = {
            "revision": revision, "mode": "corrected-color-drift", "photo": face["id"], "other": cross["id"],
            "adjustment": reading["id"],
            "marks": {"region": inspect_mark["id"]}, "options": {"delta_e": 5},
        }
        response = client.post(f"/section/runs/{run_id}/readings", json=drift_payload)
        check(response.status_code == 201, f"corrected color difference saved ({response.status_code})")
        drift = response.get_json()["reading"]
        revision = response.get_json()["run"]["revision"]
        ds = drift["summary"]
        check(ds["window_pixels"] == ds["paired_pixels"] + ds["missing_evidence_pixels"]
              + ds["missing_in_reference_pixels"],
              "window pixels partition across paired and missing-evidence classes")
        check(ds["adjustment_fit"] == "rigid", "color difference records which fit was applied")
        check(len(drift["rows"]) > 0, "per-pixel audit rows present")
        check(len(drift["rows"]) <= 9000, "per-pixel audit stays within stored row cap")
        check({"x_mm", "y_mm", "observation", "delta_e76"} <= set(drift["rows"][0]),
              "audit rows carry coordinates, observation class and Lab distance")

        # Reopen saved reading and packet: everything needed to recheck is inside.
        reopened = client.get(f"/section/runs/{run_id}/readings/{drift['id']}").get_json()
        check(reopened["correction"]["adopted"] == "rigid"
              and reopened["summary"]["window_pixels"] == ds["window_pixels"],
              "saved corrected reading reopens with its correction record and exact summary values")
        check(reopened["marks"]["region"]["id"] == inspect_mark["id"],
              "saved reading keeps the exact mark references needed to recheck")
        packet = client.get(f"/section/runs/{run_id}/readings/{drift['id']}/packet")
        check(packet.status_code == 200 and packet.data[:2] == b"PK", "evidence packet downloads")

        # 4. The unqualified adjustment must be rejected as a correction source.
        bad_payload = dict(drift_payload, revision=revision, adjustment=fail_reading["id"])
        response = client.post(f"/section/runs/{run_id}/readings", json=bad_payload)
        check(response.status_code == 400, "corrected drift without qualified adjustment is rejected")

        # 5. Windows over the known invalid patches must flag missing evidence.
        # Cross photo carries a white patch; face photo carries a black patch.
        window_specs = [
            ("Window over comparison white patch",
             [[636 / 1039, 406 / 649], [672 / 1039, 406 / 649],
              [672 / 1039, 450 / 649], [636 / 1039, 450 / 649]],
             "comparison"),
            ("Window over reference black patch",
             [[724 / 1039, 488 / 649], [768 / 1039, 488 / 649],
              [768 / 1039, 517 / 649], [724 / 1039, 517 / 649]],
             "reference"),
        ]
        for title, points_outline, which_missing in window_specs:
            made = client.post(f"/section/runs/{run_id}/marks",
                               json={"title": title, "purpose": "inspect", "points": points_outline,
                                     "revision": revision, "photo": face["id"]})
            check(made.status_code == 201, f"window mark '{title}' saved")
            revision = made.get_json()["revision"]
            mark_id = made.get_json()["marks"][-1]["id"]
            response = client.post(
                f"/section/runs/{run_id}/readings",
                json=dict(drift_payload, revision=revision, marks={"region": mark_id}))
            check(response.status_code == 201, f"reading over '{title}' completes")
            miss_summary = response.get_json()["reading"]["summary"]
            revision = response.get_json()["run"]["revision"]
            if which_missing == "comparison":
                check(miss_summary["missing_evidence_pct"] > 0,
                      "missing comparison pixels explicitly counted as missing evidence")
            else:
                check(miss_summary["missing_in_reference_pixels"] > 0,
                      "missing reference pixels are reported separately")
            classes = {r["observation"] for r in response.get_json()["reading"]["rows"]}
            check(any(c != "Paired" for c in classes),
                  f"non-paired positions labelled in the audit table ({which_missing})")

        # 6. Original four-corner registration and plain color-drift keep working unchanged.
        plain = {
            "revision": revision, "mode": "color-drift", "photo": face["id"], "other": cross["id"],
            "marks": {"region": inspect_mark["id"]}, "options": {"delta_e": 5},
        }
        response = client.post(f"/section/runs/{run_id}/readings", json=plain)
        check(response.status_code == 201, "plain registered color difference still works")
        plain_summary = response.get_json()["reading"]["summary"]
        check("correction" not in response.get_json()["reading"],
              "plain color-drift stores no correction block")
        check(set(plain_summary) == {"median_delta_e76", "above_tolerance_pct", "paired_pixels"},
              "plain color-drift summary keeps its original shape")

        # 7. Wrong photo pair on an adjustment is rejected.
        run = client.get(f"/section/runs/{run_id}").get_json()
        revision = run["revision"]
        detail = next(p for p in run["photos"] if p["role"] == "detail")
        detail_region = next(m for m in run["marks"]
                             if m["photo"] == detail["id"] and m["purpose"] == "inspect")
        wrong = dict(drift_payload, revision=revision, photo=detail["id"],
                     marks={"region": detail_region["id"]})
        response = client.post(f"/section/runs/{run_id}/readings", json=wrong)
        check(response.status_code == 400, "adjustment from another photograph pair is rejected")

        # 8. Independent Procrustes check with a known rotation + scale + translation.
        from sectionlab.adjustment import _procrustes
        rng = np.random.default_rng(7)
        base = rng.normal(size=(6, 2)) * 5
        angle = math.radians(12)
        r_true = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        s_true = 1.03
        t_true = np.array([2.5, -1.7])
        moved = ((r_true @ base.T).T + t_true) / s_true  # comparison coords
        lin, rot, scale, translation, predicted, resid = _procrustes(base, moved, allow_scale=True)
        check(abs(scale - s_true) < 1e-9 and np.max(resid) < 1e-9,
              "similarity Procrustes recovers known rotation, scale and translation")
        lin0, rot0, scale0, _, _, resid0 = _procrustes(base, moved, allow_scale=False)
        check(abs(scale0 - 1.0) < 1e-12, "rigid Procrustes forces unit scale")
        check(np.mean(resid0 ** 2) > np.mean(resid ** 2),
              "rigid residuals exceed similarity residuals when true geometry includes scale")

        print("\nAll checks passed.")


if __name__ == "__main__":
    main()

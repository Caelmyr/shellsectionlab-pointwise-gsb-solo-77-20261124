"""End-to-end API checks for paired pore map and paired line intercept readings."""
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

from sectionlab import demo  # noqa: E402
from sectionlab.__main__ import application  # noqa: E402
from sectionlab.store import Library  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        app = application(root)
        client = app.test_client()

        # config advertises the new readings
        config = client.get("/section/config").get_json()
        assert "pore-pair-map" in config["readings"]
        assert "paired-pore-intercept" in config["readings"]
        assert config["readings"]["pore-pair-map"]["paired"] is True
        assert config["readings"]["paired-pore-intercept"]["paired"] is False
        print("config OK")

        library = app.extensions["section_library"]
        run = demo.create(library)
        rid = run["id"]
        photos = {p["role"]: p for p in run["photos"]}
        face, cross = photos["face"], photos["cross"]
        pore_marks = [m for m in run["marks"] if m["purpose"] == "pore"]
        pore_face = next(m for m in pore_marks if m["photo"] == face["id"])
        pore_cross = next(m for m in pore_marks if m["photo"] == cross["id"])
        stroke_face = next(m for m in run["marks"]
                           if m["photo"] == face["id"] and m["purpose"] == "stroke" and "pore" in m["title"].lower())

        # 1. paired pore map
        payload = {
            "revision": run["revision"], "mode": "pore-pair-map", "photo": face["id"], "other": cross["id"],
            "marks": {"region": pore_face["id"], "comparison": pore_cross["id"]},
            "options": {"luma": .31, "max_center_mm": 1.5, "min_overlap_mm2": .02},
        }
        resp = client.post(f"/section/runs/{rid}/readings", json=payload)
        assert resp.status_code == 201, resp.get_json()
        created = resp.get_json()
        run = created["run"]
        pair_reading = created["reading"]
        assert pair_reading["mode"] == "pore-pair-map"
        pairing = pair_reading["pairing"]
        assert pairing["reference_photo"] == face["id"] and pairing["comparison_photo"] == cross["id"]
        assert pairing["pairs"], "demo pores should pair across the two views"
        print("paired pore map OK:", pair_reading["summary"])

        # each pore used at most once on both sides
        ref_centers = [tuple(p["center_reference_mm"]) for p in pairing["pairs"]]
        cmp_centers = [tuple(p["center_comparison_mm"]) for p in pairing["pairs"]]
        assert len(set(ref_centers)) == len(ref_centers)
        assert len(set(cmp_centers)) == len(cmp_centers)

        # 2. paired line intercept referencing the saved pairing
        payload2 = {
            "revision": run["revision"], "mode": "paired-pore-intercept", "photo": face["id"],
            "marks": {"line": stroke_face["id"]},
            "options": {"luma": .31}, "pairing": pair_reading["id"],
        }
        resp2 = client.post(f"/section/runs/{rid}/readings", json=payload2)
        assert resp2.status_code == 201, resp2.get_json()
        intercept = resp2.get_json()["reading"]
        assert intercept["pairing_ref"] == pair_reading["id"]
        entries = {row.get("entry") for row in intercept["rows"] if "entry" in row}
        assert entries <= {"Adopted paired pore", "Single-photo dark spot", "Missing evidence"}
        print("paired intercept OK:", intercept["summary"])

        # 3. rejecting a non-pairing reading as the pairing reference
        run = client.get(f"/section/runs/{rid}").get_json()
        plain = client.post(f"/section/runs/{rid}/readings", json={
            "revision": run["revision"], "mode": "pore-map", "photo": face["id"],
            "marks": {"region": pore_face["id"]}, "options": {"luma": .31}})
        assert plain.status_code == 201, plain.get_json()
        plain_id = plain.get_json()["reading"]["id"]
        run = client.get(f"/section/runs/{rid}").get_json()
        bad = client.post(f"/section/runs/{rid}/readings", json={
            "revision": run["revision"], "mode": "paired-pore-intercept", "photo": face["id"],
            "marks": {"line": stroke_face["id"]}, "options": {"luma": .31}, "pairing": plain_id})
        assert bad.status_code == 400, bad.get_json()
        print("invalid pairing reference rejected OK")

        # 4. the ordinary readings remain unchanged in shape (regression)
        ordinary = client.get(f"/section/runs/{rid}/readings/{plain_id}").get_json()
        assert "pore_candidate_count" in ordinary["summary"]
        assert "pairing" not in ordinary

        # 5. evidence packet stays self-contained
        packet = client.get(f"/section/runs/{rid}/readings/{pair_reading['id']}/packet")
        assert packet.status_code == 200
        with zipfile.ZipFile(io.BytesIO(packet.data)) as archive:
            names = archive.namelist()
            assert "reading.json" in names and "measured-values.csv" in names and "proof-overlay.png" in names
            doc = json.loads(archive.read("reading.json"))
            assert doc["pairing"]["pairs"] == pairing["pairs"]
        print("evidence packet OK:", names)

        # 6. saved pairing survives reload from the immutable record
        reloaded = library.reading(library.run(rid), pair_reading["id"])
        assert reloaded["pairing"] == pairing
        print("reload OK")
    print("end-to-end checks passed")


if __name__ == "__main__":
    main()

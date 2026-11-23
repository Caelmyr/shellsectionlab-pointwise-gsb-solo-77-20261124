"""Routes for the shell section workbench and archived reading evidence."""
from io import BytesIO

from flask import Blueprint, Response, jsonify, render_template, request, send_file

from image_lab.validation import Conflict, Invalid

from . import adjustment, analysis, demo, evidence, photos
from .model import MARKS, READINGS, ROLES, choice, locate, mark_fields, obj, run_fields, text
from .store import Library, identity, stamp

MEASUREMENTS = {
    "section-tone": (analysis.section_tone, {"region": "inspect"}, {}),
    "section-contrast": (analysis.section_contrast, {"matrix": "matrix", "accent": "accent"}, {}),
    "inclusion-fraction": (analysis.inclusion_fraction, {"region": "decor", "matrix": "matrix", "accent": "accent"}, {}),
    "lamina-spacing": (analysis.lamina_spacing, {"region": "repeat"}, {}),
    "pore-map": (analysis.pore_map, {"region": "pore"}, {}),
    "pore-intercept": (analysis.pore_intercept, {"line": "stroke"}, {}),
    "growth_transition": (analysis.growth_transition, {"line": "stroke"}, {}),
    "landmark-offset": (analysis.landmark_offsets, {"reference": "register"}, {"comparison": "register"}),
    "landmark-adjust": (adjustment.landmark_adjust, {"reference": "register"}, {"comparison": "register"}),
    "polar-response": (analysis.polar_response, {"matte": "matte", "gloss": "gloss", "decor": "decor"}, {}),
    "paired-coverage": (analysis.paired_coverage, {"region": "inspect"}, {}),
    "surface-marks": (analysis.surface_marks, {"region": "inspect"}, {}),
    "color-drift": (analysis.color_drift, {"region": "inspect"}, {}),
    "corrected-color-drift": (adjustment.corrected_color_drift, {"region": "inspect"}, {}),
}
PAIRED = {"landmark-offset", "landmark-adjust", "polar-response", "paired-coverage",
          "color-drift", "corrected-color-drift"}
OPTIONS = {
    "inclusion-fraction": {"threshold": ["Inclusion likeness threshold", .5, .01]},
    "pore-map": {"luma": ["Dark pore threshold", .31, .01]},
    "pore-intercept": {"luma": ["Dark pore threshold", .31, .01]},
    "surface-marks": {"contrast": ["Spot contrast", .075, .005], "minimum_mm2": ["Minimum area (mm²)", .15, .01]},
    "color-drift": {"delta_e": ["Color difference threshold", 5, .1]},
    "corrected-color-drift": {"delta_e": ["Color difference threshold", 5, .1]},
    "landmark-adjust": {
        "max_residual_mm": ["Maximum remaining landmark deviation (mm)", 1, .05],
        "max_scale_pct": ["Allowed uniform scale change (%)", 2, .1],
    },
}


def input_json():
    return obj(request.get_json(silent=True))


def sample_frame(library, run, photo):
    return photos.plane({**photo, "pixels": photos.decode(library.blob(photo["blob"]).read_bytes())}, run)


def measure(library, run, data):
    mode = choice(data.get("mode"), MEASUREMENTS, "reading")
    work, source_roles, other_roles = MEASUREMENTS[mode]
    source = locate(run["photos"], data.get("photo"), "The source photograph")
    source_frame = sample_frame(library, run, source)
    options = obj(data.get("options", {}))
    marked = obj(data.get("marks", {}))
    selected = {}

    def read_marks(assigned, capture):
        for slot, purpose in assigned.items():
            mark = locate(run["marks"], marked.get(slot), MARKS[purpose])
            if mark["photo"] != capture["id"] or mark["purpose"] != purpose:
                raise Invalid(f"Select a {MARKS[purpose].lower()} mark from the specified photograph.")
            selected[slot] = mark

    read_marks(source_roles, source)
    used = [source]
    if mode in PAIRED:
        other = locate(run["photos"], data.get("other"), "The comparison photograph")
        if other["id"] == source["id"]:
            raise Invalid("Choose a different comparison photograph.")
        other_frame = sample_frame(library, run, other)
        applied_correction = None
        if mode == "corrected-color-drift":
            adjustment_id = data.get("adjustment")
            if not isinstance(adjustment_id, str):
                raise Invalid("Select the saved paired landmark adjustment to apply.")
            adjustment_row = next((row for row in run["readings"]
                                   if row["id"] == adjustment_id and row["mode"] == "landmark-adjust"), None)
            if adjustment_row is None:
                raise Invalid("Select a saved paired landmark adjustment from this run.")
            record = library.reading(run, adjustment_id)
            applied_correction = adjustment.load_correction(
                record, source, other, source_frame, other_frame)
        read_marks(other_roles, other)
        used.append(other)
        if applied_correction is not None:
            options["_correction"] = applied_correction
        result = work(source_frame, other_frame, selected, options)
    else:
        result = work(source_frame, selected, options)
    result = list(result)
    if len(result) == 4:
        summary, rows, pixels, correction = result
        options.pop("_correction", None)
    else:
        summary, rows, pixels = result
        correction = None
    if mode == "corrected-color-drift":
        # Persist the exact applied geometry and the reading it was adopted from,
        # so the narrow-band result can be reopened and rechecked independently.
        correction = {
            "adjustment_reading_id": adjustment_id,
            "adopted": applied_correction["adopted"],
            "reason": applied_correction["reason"],
            "thresholds": applied_correction["thresholds"],
            "frame": applied_correction["frame"],
            "candidates": applied_correction["candidates"],
            "matrix_px": applied_correction["matrix_px"],
        }
    saved_options = {key: value for key, value in options.items() if not key.startswith("_")}
    document = {
        "mode": mode, "title": READINGS[mode][0], "summary": summary, "rows": rows, "photos": used,
        "options": saved_options, "marks": selected, "sheet_mm": [run["width_mm"], run["height_mm"]],
        "source_revision": run["revision"], "overlay": library.keep(photos.png(pixels)),
    }
    if correction is not None:
        document["correction"] = correction
    return document


def install(app, root):
    library = Library(root)
    api = Blueprint("sectionlab", __name__, url_prefix="/section")

    @app.get("/")
    def section_desk():
        return render_template("section.html")

    @api.get("/health")
    def health():
        return jsonify(service="sectionlab", status="ready")

    @api.get("/config")
    def config():
        definitions = {}
        for mode, (title, help_text) in READINGS.items():
            _, source_roles, other_roles = MEASUREMENTS[mode]
            definitions[mode] = {"title": title, "help": help_text, "reference": source_roles, "comparison": other_roles,
                                 "paired": mode in PAIRED, "options": OPTIONS.get(mode, {})}
        return jsonify(readings=definitions, marks=MARKS, roles=ROLES)

    @api.get("/runs")
    def runs():
        return jsonify(library.list())

    @api.post("/runs")
    def new_run():
        return jsonify(library.new_run(run_fields(input_json()))), 201

    @api.post("/demo")
    def new_demo():
        return jsonify(demo.create(library)), 201

    @api.get("/runs/<key>")
    def run(key):
        return jsonify(library.run(key))

    @api.post("/runs/<key>/photos")
    def add_photo(key):
        incoming = request.files.get("file")
        if incoming is None:
            raise Invalid("Choose a photograph to upload.")
        data = incoming.stream.read(photos.MAX_PAYLOAD + 1)
        if not data or len(data) > photos.MAX_PAYLOAD:
            raise Invalid("Use a nonempty photograph of at most 20 MiB.")
        photos.decode(data)
        title = text(request.form.get("title"), "a photograph title")
        role = choice(request.form.get("role"), ROLES, "photograph role")
        try:
            revision = int(request.form.get("revision"))
        except (TypeError, ValueError) as error:
            raise Invalid("The run revision is required.") from error

        def append(run):
            if len(run["photos"]) >= 12:
                raise Invalid("This run already holds 12 sample photographs.")
            run["photos"].append({"id": identity(), "title": title, "name": title, "role": role,
                                  "filename": photos.basename(incoming.filename), "blob": library.keep(data),
                                  "corners": None, "synthetic": False, "uploaded": stamp()})

        return jsonify(library.change(key, revision, append)), 201

    @api.put("/runs/<key>/photos/<photo_key>/calibration")
    def calibrate(key, photo_key):
        data = input_json()
        corners = photos.corners(data.get("corners"))

        def update(run):
            locate(run["photos"], photo_key, "The photograph")["corners"] = corners

        return jsonify(library.change(key, data.get("revision"), update))

    @api.get("/runs/<key>/photos/<photo_key>/image")
    def image_preview(key, photo_key):
        run = library.run(key)
        photo = locate(run["photos"], photo_key, "The photograph")
        if request.args.get("view") == "source" or not photo["corners"]:
            pixels = photos.decode(library.blob(photo["blob"]).read_bytes())
        else:
            pixels = photos.image(sample_frame(library, run, photo))
        return Response(photos.png(pixels), mimetype="image/png")

    @api.get("/runs/<key>/photos/<photo_key>/source")
    def original(key, photo_key):
        run = library.run(key)
        photo = locate(run["photos"], photo_key, "The photograph")
        return send_file(library.blob(photo["blob"]), download_name=photo["filename"], as_attachment=True)

    @api.post("/runs/<key>/marks")
    def create_mark(key):
        data = input_json()
        mark = mark_fields(data)

        def change(run):
            photo = locate(run["photos"], data.get("photo"), "The photograph")
            if not photo["corners"]:
                raise Invalid("Register the sheet before marking its calibrated plane.")
            if len(run["marks"]) >= 100:
                raise Invalid("This run already holds 100 marks.")
            run["marks"].append({**mark, "id": identity(), "photo": photo["id"]})

        return jsonify(library.change(key, data.get("revision"), change)), 201

    @api.delete("/runs/<key>/marks/<mark_key>")
    def delete_mark(key, mark_key):
        data = input_json()

        def change(run):
            run["marks"].remove(locate(run["marks"], mark_key, "The mark"))

        return jsonify(library.change(key, data.get("revision"), change))

    @api.post("/runs/<key>/readings")
    def create_reading(key):
        data = input_json()
        run = library.run(key)
        if run["revision"] != data.get("revision"):
            raise Conflict("The run changed. Reload it before saving.")
        result = measure(library, run, data)
        updated, reading = library.save_reading(key, data["revision"], result)
        return jsonify(run=updated, reading=reading), 201

    @api.get("/runs/<key>/readings/<reading_id>")
    def reading_detail(key, reading_id):
        return jsonify(library.reading(library.run(key), reading_id))

    @api.get("/runs/<key>/readings/<reading_id>/overlay")
    def reading_overlay(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(library.blob(result["overlay"]), mimetype="image/png")

    @api.get("/runs/<key>/readings/<reading_id>/csv")
    def reading_csv(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(BytesIO(evidence.rows_csv(result)), download_name="measured-values.csv",
                         as_attachment=True, mimetype="text/csv")

    @api.get("/runs/<key>/readings/<reading_id>/packet")
    def reading_packet(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(BytesIO(evidence.packet(library, result)), download_name="sample-reading.zip",
                         as_attachment=True, mimetype="application/zip")

    app.register_blueprint(api)
    app.extensions["section_library"] = library

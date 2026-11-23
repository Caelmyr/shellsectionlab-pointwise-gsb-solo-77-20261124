"""Validation and document shapes for section runs, registered photographs and marks."""
import math
import re

from image_lab.validation import Invalid

ROLES = {
    "face": "Parallel-polarized light", "detail": "Macro detail", "cross": "Crossed polarization",
}
MARKS = {
    "matrix": "Shell matrix control", "accent": "Dark inclusion control", "repeat": "Growth laminae patch",
    "pore": "Pore search patch", "inspect": "Inspection patch",
    "matte": "Low-response witness", "gloss": "High-response witness", "decor": "Target inclusion patch",
    "register": "Matched section landmarks", "stroke": "Growth-boundary transect",
}
READINGS = {
    "section-tone": ("Section tone", "Read Lab tone and color variation in an observed patch."),
    "section-contrast": ("Matrix and inclusion contrast", "Compare the shell matrix and inclusion control colors."),
    "inclusion-fraction": ("Inclusion fraction", "Measure visible inclusion-like pixels along the matrix-to-inclusion color axis."),
    "inclusion-evidence": ("Evidenced inclusion fraction", "Confirm color-suspect inclusions against the polarization pair and bound the fraction."),
    "lamina-spacing": ("Lamina spacing", "Look for regularly spaced growth laminae in a marked patch."),
    "pore-map": ("Pore map", "Locate compact dark pore candidates and separate elongated fissures."),
    "pore-intercept": ("Line pore intercept", "Measure observed clear, dark and missing intervals on a calibrated line."),
    "growth_transition": ("Growth boundary transition", "Measure the 10–90% transition along a marked cross section."),
    "landmark-offset": ("Section landmark offsets", "Compare positions of the same landmarks between two images."),
    "polar-response": ("Polarization response", "Normalize the observed brightness change against low- and high-response witnesses."),
    "paired-coverage": ("Paired section visibility", "See usable and missing positions in two independently registered photographs."),
    "surface-marks": ("Local alteration map", "Find local mottling or highlights on one view."),
    "color-drift": ("Registered color difference", "Measure Lab differences only at positions visible in both photographs."),
}
POINT_MARKS = {"register", "stroke"}
IDENTITY = re.compile(r"^[0-9a-f]{24}$")


def identifier(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise Invalid("The item reference is invalid.")
    return value


def locate(items, value, label):
    wanted = identifier(value)
    for item in items:
        if item["id"] == wanted:
            return item
    raise Invalid(f"{label} is no longer available.")


def obj(value):
    if not isinstance(value, dict):
        raise Invalid("Expected a set of named fields.")
    return value


def text(value, label, limit=100, optional=False):
    if not isinstance(value, str) or len(value.strip()) > limit or (not optional and not value.strip()):
        raise Invalid(f"Enter {label} using at most {limit} characters.")
    return value.strip()


def number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise Invalid(f"Enter a valid {label}.")
    if not minimum <= value <= maximum:
        raise Invalid(f"{label.capitalize()} must be between {minimum:g} and {maximum:g}.")
    return value


def choice(value, values, label):
    if not isinstance(value, str) or value not in values:
        raise Invalid(f"Choose a valid {label}.")
    return value


def points(raw, minimum, maximum, label):
    if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
        raise Invalid(f"{label.capitalize()} needs between {minimum} and {maximum} ordered points.")
    result = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise Invalid(f"{label.capitalize()} needs coordinate pairs.")
        result.append([number(point[0], "position", 0, 1), number(point[1], "position", 0, 1)])
    return result


def run_fields(raw):
    data = obj(raw)
    return {
        "title": text(data.get("title"), "a run title"),
        "stock": text(data.get("stock", ""), "the stock description", 100, optional=True),
        "width_mm": number(data.get("width_mm"), "sample width (mm)", 20, 1500),
        "height_mm": number(data.get("height_mm"), "sample height (mm)", 20, 1500),
    }


def mark_fields(raw):
    data = obj(raw)
    purpose = choice(data.get("purpose"), MARKS, "mark purpose")
    if purpose == "stroke":
        minimum, maximum = 2, 2
    elif purpose == "register":
        minimum, maximum = 2, 16
    else:
        minimum, maximum = 3, 32
    return {
        "title": text(data.get("title"), "a mark title"),
        "purpose": purpose,
        "points": points(data.get("points"), minimum, maximum, MARKS[purpose]),
    }

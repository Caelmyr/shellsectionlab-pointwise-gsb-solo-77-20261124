"""Generate a synthetic shell with laminae, mineral inclusions and pores."""

import math

import cv2 as cv
import numpy as np

from .photos import png
from .store import identity, stamp

WIDTH, HEIGHT = 1040, 650
CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]
LANDMARKS = [(142, 210), (907, 193), (153, 516), (903, 527)]
GROWTH_NODES = [(318, 278), (386, 296), (463, 316), (543, 294)]
CONTROLS = [
    ("Light inner matrix", "matrix", (207, 331, 293, 379)),
    ("Dark inclusion", "accent", (373, 282, 402, 307)),
    ("Repeated shell laminae", "repeat", (487, 354, 640, 422)),
    ("Pores beside the growth band", "pore", (205, 395, 433, 530)),
    ("Discolored outer wall", "inspect", (579, 394, 772, 532)),
    ("Low-response outer material", "matte", (684, 302, 765, 345)),
    ("High-response inner material", "gloss", (472, 258, 535, 300)),
    ("Inclusion-bearing lamina", "decor", (305, 267, 575, 321)),
]


def normalized(points):
    return [[x / (WIDTH - 1), y / (HEIGHT - 1)] for x, y in points]


def outline(bounds):
    left, top, right, bottom = bounds
    return normalized([(left, top), (right, top), (right, bottom), (left, bottom)])


def exposed_section(role, shift, seed):
    rng = np.random.default_rng(seed)
    v, u = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    x, y = u - shift[0], v - shift[1]
    mouth = np.sqrt(((x - 531) / 451) ** 2 + ((y - 391) / 263) ** 2)
    inner_bow = 306 - ((x - 511) ** 2) * .00070
    material = (mouth < 1) & (y > inner_bow - 126)
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    pixels[:] = (52, 58, 66)
    pixels[material] = (198, 173, 141)
    distance = (y - inner_bow) + .10 * x
    ridges = np.sin(distance * (2 * math.pi / 10.0))
    fine = rng.normal(0, 2.0, (HEIGHT, WIDTH))
    ripples = (6 * ridges + 3 * np.sin(distance * (2 * math.pi / 3.8)) + fine)
    pixels[material] += ripples[material][:, None]
    inner = material & (y < 333) & (x > 110) & (x < 902)
    pixels[inner] += np.array([17, 12, 4])
    shell_outer = material & (mouth > .88) & (y > 343)
    pixels[shell_outer] -= np.array([29, 39, 36])
    for cx, cy, rx, ry in ((317, 278, 31, 19), (386, 296, 41, 30), (463, 316, 26, 16), (543, 294, 32, 14)):
        clast = ((x-cx)/rx)**2 + ((y-cy)/ry)**2 < 1
        pixels[clast] = np.array([98, 77, 82]) + fine[clast][:, None]
    weathered = np.clip((x - 809) / 37, 0, 1) * material * (y > 329) * (y < 473)
    pixels = pixels * (1 - weathered[:, :, None]) + np.array([104, 87, 78]) * weathered[:, :, None]
    if role == "cross":
        response = material * (4 + 31 * np.exp(-((x - 514) / 196) ** 2 - ((y - 282) / 98) ** 2))
        pixels += response[:, :, None] * np.array([1.10, 1.00, .78])
    elif role == "detail":
        pixels[material] += np.array((3, 1, -3))

    rgba = cv.cvtColor(np.clip(pixels, 0, 255).astype(np.uint8), cv.COLOR_RGB2RGBA)
    for x0, y0, radius in ((238, 413, 6), (288, 447, 8), (337, 431, 5), (344, 491, 9), (391, 462, 4)):
        cv.circle(rgba, (x0 + shift[0], y0 + shift[1]), radius + 1, (114, 81, 79, 255), -1, cv.LINE_AA)
        cv.circle(rgba, (x0 + shift[0], y0 + shift[1]), radius, (53, 38, 48, 255), -1, cv.LINE_AA)
    fissure = np.array([[248, 489], [263, 491], [276, 498], [304, 494]]) + shift
    cv.polylines(rgba, [fissure.astype(np.int32)], False, (58, 36, 49, 255), 3, cv.LINE_AA)
    for x0,y0,radius in ((612, 455, 8),(680, 423, 11),(749, 473, 7)):
        cv.circle(rgba, (x0 + shift[0], y0 + shift[1]), radius, (254, 215, 161, 255), -1, cv.LINE_AA)
    for idx,(x0,y0) in enumerate(LANDMARKS):
        cv.drawMarker(rgba, (x0 + shift[0] + idx % 2, y0 + shift[1]), (75, 43, 55, 255), cv.MARKER_CROSS, 13, 2)
    for idx,(x0,y0) in enumerate(GROWTH_NODES):
        node = (x0 + shift[0], y0 + shift[1])
        cv.drawMarker(rgba, node, (40, 33, 78, 255), cv.MARKER_DIAMOND, 15, 2)
        cv.putText(rgba, str(idx + 1), (node[0] + 8, node[1] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, .5, (40, 33, 78, 255), 1, cv.LINE_AA)
    if role == "face":
        rgba[488:517, 724:768] = 0
    elif role == "cross":
        rgba[412:445, 641:666] = (255, 255, 255, 255)
    cv.putText(rgba, "SYNTHETIC SHELL SECTION", (110, 615), cv.FONT_HERSHEY_SIMPLEX, .64, (243, 223, 207, 255), 2)
    return rgba


def create(library):
    photos, marks = [], []
    for title, role, shift, seed in (
        ("Parallel light / cut face", "face", (0,0), 128),
        ("Crossed light / same face", "cross", (3,-2), 129),
        ("Macro / outer wall", "detail", (-2,1), 130),
    ):
        photo = {"id": identity(), "title": title, "role": role, "filename": role + "-section.png",
                 "blob": library.keep(png(exposed_section(role, shift, seed))), "corners": CORNERS,
                 "synthetic": True, "uploaded": stamp()}
        photos.append(photo)
        items = [(title, purpose, outline(bounds)) for title, purpose, bounds in CONTROLS]
        items.extend([
            ("Across the outer growth boundary", "stroke", normalized([(784,399),(890,399)])),
            ("Through an observed pore", "stroke", normalized([(215,413),(265,413)])),
            ("Corresponding section landmarks", "register", normalized([
                (x + shift[0] + i % 2, y + shift[1]) for i,(x,y) in enumerate(LANDMARKS)
            ])),
            ("Four matched growth nodes", "register", normalized([
                (x + shift[0], y + shift[1]) for x, y in GROWTH_NODES
            ])),
        ])
        for title,purpose,points in items:
            marks.append({"id": identity(), "title": title, "purpose": purpose, "points": points, "photo": photo["id"]})
    return library.new_run({"title": "Laminated shell section", "stock": "Synthetic cut-face demonstration",
                            "width_mm": 312, "height_mm": 195}, photos, marks)

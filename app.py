import inspect
import os
import base64
from pathlib import Path

import gradio as gr
import joblib
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "moid_model.joblib"

BACKGROUND_PATH = BASE_DIR / "assets" / "background.png"
HEADER_PATH = BASE_DIR / "assets" / "header.png"

# NEW SECTION IMAGES
CHARACTERISTICS_PATH = BASE_DIR / "assets" / "asteroid-characteristics.png"
RESULT_PATH = BASE_DIR / "assets" / "prediction-result.png"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(path):

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    suffix = path.suffix.lower()

    if suffix == ".png":
        mime = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"

    return f"data:{mime};base64,{encoded}"


BACKGROUND_DATA = image_to_base64(BACKGROUND_PATH)
HEADER_DATA = image_to_base64(HEADER_PATH)

CHARACTERISTICS_DATA = image_to_base64(CHARACTERISTICS_PATH)
RESULT_DATA = image_to_base64(RESULT_PATH)


# ============================================================
# BACKGROUND
# ============================================================

BACKGROUND_HTML = f"""
<div id="background-layer">
    <img src="{BACKGROUND_DATA}" alt="">
</div>
"""


# ============================================================
# RANDOM VALUES
# ============================================================

def random_values():

    # --------------------------------------------------------
    # Plausible asteroid orbital values
    # --------------------------------------------------------
    #
    # H      : Absolute magnitude
    # i      : Inclination (degrees)
    # albedo : Surface reflectivity
    # e      : Orbital eccentricity
    # a      : Semi-major axis (AU)
    # om     : Longitude of ascending node (degrees)
    # w      : Argument of perihelion (degrees)
    # ma     : Mean anomaly (degrees)
    #
    # Values are randomly generated every time the button
    # is pressed.
    # --------------------------------------------------------

    H = round(np.random.uniform(14.0, 24.0), 2)

    i = round(np.random.uniform(0.0, 30.0), 2)

    albedo = round(np.random.uniform(0.03, 0.45), 3)

    e = round(np.random.uniform(0.02, 0.65), 3)

    a = round(np.random.uniform(0.85, 3.20), 3)

    om = round(np.random.uniform(0.0, 360.0), 2)

    w = round(np.random.uniform(0.0, 360.0), 2)

    ma = round(np.random.uniform(0.0, 360.0), 2)

    return (
        H,
        i,
        albedo,
        e,
        a,
        om,
        w,
        ma
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_moid(H, i, albedo, e, a, om, w, ma):

    try:

        values = np.array(
            [[
                float(H),
                float(albedo),
                float(e),
                float(a),
                float(i),
                float(om),
                float(w),
                float(ma),
            ]],
            dtype=float
        )

        prediction = float(model.predict(values)[0])

        prediction = max(0.0, prediction)

        # 1 AU = 149,597,870.7 km
        km = prediction * 149_597_870.7

        return (
            f"{prediction:.6f} AU",
            f"{km:,.2f} km"
        )

    except Exception as error:

        return (
            f"Error: {error}",
            "Please check your input values."
        )


# ============================================================
# CSS
# ============================================================

CSS = r"""

/* ============================================================
   FONT
   ============================================================ */

@import url("https://cdn.jsdelivr.net/npm/@fontsource-variable/geist-pixel@5.3.1/index.css");


/* ============================================================
   REMOVE SCROLLBARS
   ============================================================ */

* {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

*::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    background: transparent !important;
}


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100% !important;
    background: transparent !important;

    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

body {
    overflow-x: hidden !important;
}


/* ============================================================
   BACKGROUND
   ============================================================ */

#background-layer {
    position: fixed !important;

    top: 0 !important;
    left: 0 !important;

    width: 100% !important;
    height: 100vh !important;

    overflow: hidden !important;

    z-index: 0 !important;

    pointer-events: none !important;
}

#background-layer img {
    display: block !important;

    width: 100% !important;
    height: 100vh !important;

    max-width: none !important;

    object-fit: cover !important;
    object-position: center center !important;
}


/* ============================================================
   MAIN GRADIO CONTAINER
   ============================================================ */

.gradio-container {

    --pixel-glow:
        0 0 0 4px rgba(165, 249, 255, 0.95),
        0 0 0 8px rgba(165, 249, 255, 0.55),
        0 0 0 12px rgba(165, 249, 255, 0.25);

    --pixel-glow-hover:
        0 0 0 4px rgba(165, 249, 255, 1),
        0 0 0 8px rgba(165, 249, 255, 0.70),
        0 0 0 12px rgba(165, 249, 255, 0.40),
        0 0 0 16px rgba(165, 249, 255, 0.18);

    --pixel-glow-active:
        0 0 0 4px rgba(165, 249, 255, 0.95),
        0 0 0 8px rgba(165, 249, 255, 0.40);

    position: relative !important;

    z-index: 1 !important;

    max-width: 1150px !important;

    min-height: 100vh !important;

    margin: 0 auto !important;

    padding-left: 20px !important;
    padding-right: 20px !important;

    background: transparent !important;
    background-color: transparent !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    overflow-x: hidden !important;
}


/* ============================================================
   REMOVE LIGHT BLUE / GREY GRADIO WRAPPERS
   ============================================================ */

.gradio-container .column,
.gradio-container .form,
.gradio-container .styler,
.gradio-container .gap,
.gradio-container .stretch,
.gradio-container .row:not(.button-box):not(.result-box) {

    background: transparent !important;
    background-color: transparent !important;

    border: none !important;

    box-shadow: none !important;
}


/* ============================================================
   GENERIC BLOCKS
   ============================================================ */

.gradio-container .block {

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.header-box {

    width: 100% !important;

    display: flex !important;

    justify-content: center !important;
    align-items: center !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;

    padding: 25px 0 20px 0 !important;

    margin: 0 !important;
}

.header-box img {

    display: block !important;

    width: 100% !important;

    max-width: 1100px !important;

    height: auto !important;

    object-fit: contain !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;
}


/* ============================================================
   SECTION IMAGE
   ============================================================ */

.section-image {

    width: calc(100% - 28px) !important;

    margin: 16px 14px 26px 14px !important;

    padding: 0 !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;

    overflow: hidden !important;
}

.section-image img {

    display: block !important;

    width: 100% !important;

    height: auto !important;

    max-width: 1100px !important;

    margin: 0 auto !important;

    object-fit: contain !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;
}


/* ============================================================
   INPUT AREA
   ============================================================ */

.input-grid {

    width: 100% !important;

    display: grid !important;

    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;

    gap: 0 !important;

    margin: 0 !important;

    padding: 0 !important;

    overflow: visible !important;
}


/* ============================================================
   INPUT COLUMNS
   ============================================================ */

.input-column {

    min-width: 0 !important;

    width: 100% !important;

    overflow: visible !important;

    display: flex !important;

    flex-direction: column !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;
}


/* ============================================================
   INPUT CARDS
   ============================================================ */

.input-box {

    width: calc(100% - 28px) !important;

    min-width: 0 !important;

    box-sizing: border-box !important;

    background: #ffffff !important;

    border: 3px solid #000000 !important;

    border-radius: 0 !important;

    box-shadow: var(--pixel-glow) !important;

    padding: 22px !important;

    margin: 6px 14px 22px 14px !important;

    overflow: visible !important;
}


/* ============================================================
   VARIABLE NAME LABELS
   ============================================================ */

.gradio-container .input-box label,
.gradio-container .input-box label > span:not([class*="info"]),
.gradio-container .input-box span[data-testid="block-info"],
.gradio-container .input-box [data-testid="block-info"] {

    color: #000000 !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    font-size: 36px !important;

    font-weight: 700 !important;

    line-height: 1.2 !important;
}


/* ============================================================
   INPUT FIELD
   ============================================================ */

.input-box input {

    width: 100% !important;

    box-sizing: border-box !important;

    color: #000000 !important;

    background: #ffffff !important;

    border: 3px solid #000000 !important;

    border-radius: 0 !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    font-size: 22px !important;

    min-height: 52px !important;
}


/* ============================================================
   EXPLANATION TEXT
   ============================================================ */

.input-box small,
.input-box .info,
.input-box [class*="info"],
.input-box .description,
.input-box .wrap > small,
.gradio-container .input-box label .info,
.gradio-container .input-box label div[class*="info"],
.gradio-container .input-box label small {

    font-family:
        Arial,
        Helvetica,
        sans-serif !important;

    font-size: 14px !important;

    line-height: 1.45 !important;

    font-weight: 400 !important;

    color: #000000 !important;
}


/* ============================================================
   ONE BOX FOR BOTH BUTTONS
   ============================================================ */

.gradio-container .row\.button-box,
.button-box {

    width: auto !important;

    display: flex !important;

    flex-direction: row !important;

    justify-content: center !important;

    align-items: stretch !important;

    gap: 16px !important;

    box-sizing: border-box !important;

    background: #060b14 !important;
    background-color: #060b14 !important;

    border: 3px solid #000000 !important;

    border-radius: 0 !important;

    box-shadow: var(--pixel-glow) !important;

    padding: 18px !important;

    margin: 14px 14px 30px 14px !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

button.pixel-button {

    display: flex !important;

    visibility: visible !important;

    opacity: 1 !important;

    flex: 1 1 0 !important;

    min-height: 64px !important;

    align-items: center !important;

    justify-content: center !important;

    background: #ffffff !important;
    background-color: #ffffff !important;

    color: #000000 !important;

    border: 3px solid #000000 !important;

    border-radius: 0 !important;

    box-shadow: var(--pixel-glow) !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    font-size: 22px !important;

    font-weight: 700 !important;

    cursor: pointer !important;

    transition: none !important;
}


/* ============================================================
   BUTTON TEXT
   ============================================================ */

button.pixel-button span,
button.pixel-button div,
button.pixel-button span span {

    color: #000000 !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    font-size: 22px !important;

    font-weight: 700 !important;

    opacity: 1 !important;

    visibility: visible !important;
}


/* ============================================================
   HOVER
   ============================================================ */

button.pixel-button:hover {

    background: #ffffff !important;

    color: #000000 !important;

    box-shadow: var(--pixel-glow-hover) !important;

    transform: translateY(-2px) !important;
}


/* ============================================================
   CLICK
   ============================================================ */

button.pixel-button:active {

    transform: translateY(2px) !important;

    box-shadow: var(--pixel-glow-active) !important;
}


/* ============================================================
   PREDICTION RESULT BOX
   ============================================================ */

.gradio-container .row\.result-box,
.result-box {

    width: calc(100% - 28px) !important;

    box-sizing: border-box !important;

    display: flex !important;

    flex-direction: row !important;

    gap: 16px !important;

    background: #060b14 !important;

    background-color: #060b14 !important;

    border: 3px solid #000000 !important;

    border-radius: 0 !important;

    box-shadow: var(--pixel-glow) !important;

    padding: 22px !important;

    margin: 6px 14px 30px 14px !important;
}


/* ============================================================
   RESULT COLUMNS
   ============================================================ */

.result-box > div {

    min-width: 0 !important;

    flex: 1 1 0 !important;
}


/* ============================================================
   RESULT LABELS
   ============================================================ */

.result-box label,
.result-box label span,
.result-box [data-testid="block-info"] {

    color: #000000 !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    font-size: 24px !important;

    font-weight: 700 !important;
}


/* ============================================================
   RESULT VALUES
   ============================================================ */

.result-box input,
.result-box textarea {

    width: 100% !important;

    box-sizing: border-box !important;

    color: #000000 !important;

    background: #ffffff !important;

    border: 3px solid #000000 !important;

    border-radius: 0 !important;

    font-family:
        "Geist Pixel Variable",
        "Courier New",
        monospace !important;

    font-size: 24px !important;

    font-weight: 700 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .gradio-container {

        width: 94% !important;

        padding-left: 10px !important;
        padding-right: 10px !important;
    }


    .header-box {

        padding-top: 15px !important;
    }


    .header-box img {

        width: 100% !important;
    }


    /* Keep inputs in two columns */
    .input-grid {

        grid-template-columns:
            minmax(0, 1fr)
            minmax(0, 1fr) !important;

        gap: 0 !important;
    }


    .input-box {

        width: calc(100% - 12px) !important;

        margin-left: 6px !important;
        margin-right: 6px !important;

        padding: 12px !important;
    }


    .gradio-container .input-box label,
    .gradio-container .input-box label > span:not([class*="info"]),
    .gradio-container .input-box span[data-testid="block-info"],
    .gradio-container .input-box [data-testid="block-info"] {

        font-size: 24px !important;
    }


    .input-box input {

        font-size: 16px !important;

        min-height: 46px !important;
    }


    .input-box small,
    .input-box .info,
    .input-box [class*="info"],
    .input-box .description {

        font-size: 11px !important;
    }


    .button-box {

        flex-direction: column !important;

        gap: 12px !important;
    }


    button.pixel-button {

        width: 100% !important;

        flex: none !important;
    }


    .result-box {

        flex-direction: column !important;

        gap: 14px !important;
    }


    .result-box input,
    .result-box textarea {

        font-size: 18px !important;
    }


    .section-image {

        width: calc(100% - 12px) !important;

        margin-left: 6px !important;
        margin-right: 6px !important;
    }
}

"""


# ============================================================
# THEME
# ============================================================

THEME = gr.themes.Base(
    primary_hue="cyan",
    neutral_hue="slate"
)


# ============================================================
# GRADIO 5 vs GRADIO 6 COMPATIBILITY
# ============================================================

LAUNCH_ACCEPTS_STYLE = (
    "css" in inspect.signature(gr.Blocks.launch).parameters
)

if LAUNCH_ACCEPTS_STYLE:

    BLOCKS_STYLE_KWARGS = {}

    LAUNCH_STYLE_KWARGS = {
        "css": CSS,
        "theme": THEME
    }

else:

    BLOCKS_STYLE_KWARGS = {
        "css": CSS,
        "theme": THEME
    }

    LAUNCH_STYLE_KWARGS = {}


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="Asteroid MOID Prediction",
    **BLOCKS_STYLE_KWARGS
) as demo:


    # ========================================================
    # BACKGROUND
    # ========================================================

    gr.HTML(BACKGROUND_HTML)


    # ========================================================
    # HEADER IMAGE
    # ========================================================

    gr.HTML(
        f"""
        <div class="header-box">
            <img
                src="{HEADER_DATA}"
                alt="Asteroid MOID Prediction"
            >
        </div>
        """
    )


    # ========================================================
    # ASTEROID CHARACTERISTICS IMAGE
    # ========================================================

    gr.HTML(
        f"""
        <div class="section-image">
            <img
                src="{CHARACTERISTICS_DATA}"
                alt="Asteroid Characteristics"
            >
        </div>
        """
    )


    # ========================================================
    # INPUTS — 2 COLUMNS
    # ========================================================

    with gr.Row(elem_classes="input-grid"):

        # ----------------------------------------------------
        # LEFT COLUMN
        # ----------------------------------------------------

        with gr.Column(elem_classes="input-column"):

            H = gr.Number(
                label="H",
                value=18.5,
                info=(
                    "Absolute magnitude. A measure related "
                    "to the asteroid's intrinsic brightness and size."
                ),
                elem_classes="input-box"
            )


            i = gr.Number(
                label="i",
                value=8.0,
                info=(
                    "Inclination. The angle between the asteroid's "
                    "orbital plane and the reference plane, "
                    "measured in degrees."
                ),
                elem_classes="input-box"
            )


            e = gr.Number(
                label="e",
                value=0.25,
                info=(
                    "Orbital eccentricity. Describes how "
                    "elliptical the asteroid's orbit is."
                ),
                elem_classes="input-box"
            )


            a = gr.Number(
                label="a",
                value=1.40,
                info=(
                    "Semi-major axis of the asteroid's orbit, "
                    "measured in astronomical units (AU)."
                ),
                elem_classes="input-box"
            )


        # ----------------------------------------------------
        # RIGHT COLUMN
        # ----------------------------------------------------

        with gr.Column(elem_classes="input-column"):

            albedo = gr.Number(
                label="Albedo",
                value=0.15,
                info=(
                    "The fraction of sunlight reflected "
                    "by the asteroid's surface."
                ),
                elem_classes="input-box"
            )


            om = gr.Number(
                label="om",
                value=120.0,
                info=(
                    "Longitude of the ascending node, "
                    "measured in degrees."
                ),
                elem_classes="input-box"
            )


            w = gr.Number(
                label="w",
                value=75.0,
                info=(
                    "Argument of perihelion. Defines the "
                    "orientation of the orbit, measured in degrees."
                ),
                elem_classes="input-box"
            )


            ma = gr.Number(
                label="ma",
                value=180.0,
                info=(
                    "Mean anomaly. Represents the asteroid's "
                    "position along its orbit, measured in degrees."
                ),
                elem_classes="input-box"
            )


    # ========================================================
    # ONE BOX CONTAINING BOTH BUTTONS
    # ========================================================

    with gr.Row(elem_classes="button-box"):

        random_button = gr.Button(
            "RANDOM VALUES",
            elem_classes="pixel-button"
        )

        predict_button = gr.Button(
            "PREDICT MOID",
            elem_classes="pixel-button"
        )


    # ========================================================
    # PREDICTION RESULT IMAGE
    # ========================================================

    gr.HTML(
        f"""
        <div class="section-image">
            <img
                src="{RESULT_DATA}"
                alt="Prediction Result"
            >
        </div>
        """
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    with gr.Row(elem_classes="result-box"):

        predicted_au = gr.Textbox(
            label="Predicted MOID",
            value="—",
            interactive=False
        )

        predicted_km = gr.Textbox(
            label="Distance from Earth",
            value="—",
            interactive=False
        )


    # ========================================================
    # RANDOM BUTTON
    # ========================================================

    random_button.click(
        fn=random_values,

        inputs=[],

        outputs=[
            H,
            i,
            albedo,
            e,
            a,
            om,
            w,
            ma
        ]
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button.click(
        fn=predict_moid,

        inputs=[
            H,
            i,
            albedo,
            e,
            a,
            om,
            w,
            ma
        ],

        outputs=[
            predicted_au,
            predicted_km
        ]
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        **LAUNCH_STYLE_KWARGS
    )
import os
import random
import base64
from pathlib import Path

import joblib
import gradio as gr


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/moid_model.joblib"
BACKGROUND_PATH = Path("assets/background.png")

AU_TO_KM = 149_597_870.7


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found:\n"
        "model/moid_model.joblib\n\n"
        "Please run train.py first."
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD BACKGROUND IMAGE
# ============================================================

if not BACKGROUND_PATH.exists():
    raise FileNotFoundError(
        f"Background image not found:\n{BACKGROUND_PATH}\n\n"
        "Make sure background.png is inside the assets folder."
    )


# Detect image type
extension = BACKGROUND_PATH.suffix.lower()

if extension in [".jpg", ".jpeg"]:
    mime_type = "image/jpeg"
elif extension == ".webp":
    mime_type = "image/webp"
else:
    mime_type = "image/png"


# Convert image to Base64
with open(BACKGROUND_PATH, "rb") as f:
    image_base64 = base64.b64encode(
        f.read()
    ).decode("utf-8")


BACKGROUND_DATA = (
    f"data:{mime_type};base64,{image_base64}"
)


# ============================================================
# RANDOM VALUES
# ============================================================

def generate_random_values():

    H = round(
        random.uniform(12, 26),
        3
    )

    albedo = round(
        random.uniform(0.03, 0.40),
        4
    )

    e = round(
        random.uniform(0.01, 0.70),
        5
    )

    a = round(
        random.uniform(0.70, 4.50),
        5
    )

    i = round(
        random.uniform(0, 35),
        3
    )

    om = round(
        random.uniform(0, 360),
        3
    )

    w = round(
        random.uniform(0, 360),
        3
    )

    ma = round(
        random.uniform(0, 360),
        3
    )

    return (
        H,
        albedo,
        e,
        a,
        i,
        om,
        w,
        ma
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_moid(
    H,
    albedo,
    e,
    a,
    i,
    om,
    w,
    ma
):

    values = [
        H,
        albedo,
        e,
        a,
        i,
        om,
        w,
        ma
    ]

    # Check missing values
    if any(
        value is None
        for value in values
    ):
        return (
            "Please enter all 8 asteroid characteristics.",
            ""
        )

    # Convert to numbers
    try:

        values = [
            float(value)
            for value in values
        ]

    except (ValueError, TypeError):

        return (
            "Please enter valid numeric values.",
            ""
        )

    # Predict
    prediction = model.predict(
        [values]
    )[0]

    prediction = max(
        float(prediction),
        0.0
    )

    # AU -> km
    kilometers = (
        prediction *
        AU_TO_KM
    )

    return (
        f"{prediction:.8f} AU",
        f"{kilometers:,.2f} km"
    )


# ============================================================
# BACKGROUND HTML
# ============================================================

BACKGROUND_HTML = f"""
<div id="background-layer">

    <img
        src="{BACKGROUND_DATA}"
        alt=""
    >

</div>
"""


# ============================================================
# CSS
# ============================================================

CSS = """

/* ============================================================
   GLOBAL PAGE
   ============================================================ */

html,
body {

    margin: 0 !important;

    padding: 0 !important;

    min-height: 100% !important;

    background: transparent !important;

    font-family:
        'Oswald',
        sans-serif !important;
}


/* ============================================================
   FIXED BACKGROUND IMAGE
   ============================================================ */

#background-layer {

    position: fixed !important;

    top: 0 !important;

    left: 0 !important;

    width: 100vw !important;

    height: 100vh !important;

    margin: 0 !important;

    padding: 0 !important;

    overflow: hidden !important;

    z-index: 0 !important;

    pointer-events: none !important;
}


#background-layer img {

    display: block !important;

    width: 100vw !important;

    height: 100vh !important;

    object-fit: cover !important;

    object-position: center center !important;

    max-width: none !important;

}


/* ============================================================
   GRADIO CONTAINER
   ============================================================ */

.gradio-container {

    position: relative !important;

    z-index: 1 !important;

    max-width: 1150px !important;

    min-height: 100vh !important;

    margin: 0 auto !important;

    background: transparent !important;

    background-color: transparent !important;
}


/* ============================================================
   REMOVE DEFAULT GRADIO BACKGROUNDS
   ============================================================ */

.gradio-container .contain,
.gradio-container .form,
.gradio-container .block,
.gradio-container .panel,
.gradio-container .wrap {

    background-color: transparent !important;
}


/* ============================================================
   ALL TEXT
   ============================================================ */

.gradio-container,
.gradio-container *,
.gradio-container label,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container span {

    color: #000000 !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.hero {

    text-align: center !important;

    margin-top: 30px !important;

    margin-bottom: 25px !important;

    padding: 32px !important;

    background:
        rgba(255, 255, 255, 0.93) !important;

    border:
        2px solid #000000 !important;

    border-radius:
        8px !important;

    box-shadow:

        0 0 8px #a5f9ff,

        0 0 18px #a5f9ff !important;
}


/* ============================================================
   HEADER TITLE
   ============================================================ */

.hero-title {

    font-family:
        'Oswald',
        sans-serif !important;

    font-size:
        46px !important;

    font-weight:
        600 !important;

    line-height:
        1.1 !important;

    margin:
        0 0 12px 0 !important;

    color:
        #000000 !important;

    text-shadow:
        2px 2px 0 #a5f9ff !important;
}


/* ============================================================
   HEADER DESCRIPTION
   ============================================================ */

.hero-subtitle {

    font-family:
        'Oswald',
        sans-serif !important;

    font-size:
        18px !important;

    line-height:
        1.5 !important;

    max-width:
        800px !important;

    margin:
        0 auto !important;

    color:
        #000000 !important;
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {

    display: block !important;

    font-size:
        27px !important;

    font-weight:
        600 !important;

    color:
        #000000 !important;

    background:
        #ffffff !important;

    border:
        2px solid #000000 !important;

    border-radius:
        6px !important;

    padding:
        8px 15px !important;

    margin-top:
        25px !important;

    margin-bottom:
        15px !important;

    box-shadow:
        0 0 7px #a5f9ff !important;
}


/* ============================================================
   INPUT BOXES
   ============================================================ */

.input-box {

    background:
        #ffffff !important;

    border:
        2px solid #000000 !important;

    border-radius:
        6px !important;

    box-shadow:
        0 0 6px #a5f9ff !important;
}


/* ============================================================
   INPUT LABEL
   ============================================================ */

label {

    color:
        #000000 !important;

    font-weight:
        600 !important;
}


/* ============================================================
   INPUT FIELDS
   ============================================================ */

input,
textarea {

    background:
        #ffffff !important;

    color:
        #000000 !important;

    border-color:
        #000000 !important;

    font-family:
        'Oswald',
        sans-serif !important;
}


/* ============================================================
   INPUT DESCRIPTION
   ============================================================ */

small {

    color:
        #000000 !important;

    font-family:
        'Oswald',
        sans-serif !important;

    line-height:
        1.4 !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.pixel-button {

    background:
        #ffffff !important;

    color:
        #000000 !important;

    border:
        2px solid #000000 !important;

    border-radius:
        3px !important;

    font-family:
        'Oswald',
        sans-serif !important;

    font-weight:
        600 !important;

    letter-spacing:
        1px !important;

    box-shadow:

        3px 3px 0 #000000,

        0 0 8px #a5f9ff,

        0 0 16px #a5f9ff !important;

    transition:
        transform 0.12s ease,
        box-shadow 0.12s ease !important;
}


/* ============================================================
   BUTTON HOVER
   ============================================================ */

.pixel-button:hover {

    background:
        #ffffff !important;

    color:
        #000000 !important;

    transform:
        translate(-2px, -2px) !important;

    box-shadow:

        5px 5px 0 #000000,

        0 0 12px #a5f9ff,

        0 0 25px #a5f9ff,

        0 0 40px rgba(165, 249, 255, 0.8) !important;
}


/* ============================================================
   RANDOM BUTTON
   ============================================================ */

.random-button {

    margin-top:
        15px !important;

    margin-bottom:
        15px !important;

    min-height:
        48px !important;
}


/* ============================================================
   PREDICT BUTTON
   ============================================================ */

.predict-button {

    min-height:
        60px !important;

    font-size:
        22px !important;

    font-weight:
        600 !important;

    margin-top:
        10px !important;
}


/* ============================================================
   RESULT BOX
   ============================================================ */

.result-box {

    background:
        #ffffff !important;

    border:
        2px solid #000000 !important;

    border-radius:
        6px !important;

    box-shadow:

        0 0 8px #a5f9ff,

        0 0 16px rgba(165, 249, 255, 0.7) !important;
}


.result-box input {

    background:
        #ffffff !important;

    color:
        #000000 !important;

    font-size:
        20px !important;

    font-weight:
        600 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .hero {

        padding:
            25px 15px !important;
    }

    .hero-title {

        font-size:
            34px !important;
    }

    .hero-subtitle {

        font-size:
            16px !important;
    }

    .section-title {

        font-size:
            23px !important;
    }
}

"""


# ============================================================
# CREATE APP
# ============================================================

with gr.Blocks(
    title="Asteroid MOID Prediction"
) as demo:


    # ========================================================
    # BACKGROUND
    # ========================================================

    # This is deliberately placed BEFORE all other content.
    # It stays behind the entire application.

    gr.HTML(
        BACKGROUND_HTML
    )


    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-title">
                Asteroid MOID Prediction
            </div>

            <div class="hero-subtitle">
                Predict the Minimum Orbit Intersection Distance (MOID)
                using physical and orbital characteristics of an asteroid.
            </div>

        </div>
        """
    )


    # ========================================================
    # CHARACTERISTICS TITLE
    # ========================================================

    gr.Markdown(
        "## Asteroid Characteristics",
        elem_classes="section-title"
    )


    # ========================================================
    # INPUTS
    # ========================================================

    with gr.Row():


        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column():


            H = gr.Number(

                label="Absolute Magnitude (H)",

                info=(
                    "Measures the intrinsic brightness of the asteroid. "
                    "A lower H generally indicates a larger or more "
                    "reflective object."
                ),

                value=18.0,

                elem_classes="input-box"
            )


            albedo = gr.Number(

                label="Albedo",

                info=(
                    "The fraction of sunlight reflected by the asteroid's "
                    "surface. Higher albedo means more sunlight is reflected."
                ),

                value=0.15,

                elem_classes="input-box"
            )


            e = gr.Number(

                label="Eccentricity (e)",

                info=(
                    "Describes how elliptical the asteroid's orbit is. "
                    "A value close to 0 represents a nearly circular orbit."
                ),

                value=0.20,

                elem_classes="input-box"
            )


            a = gr.Number(

                label="Semi-major Axis (a)",

                info=(
                    "The semi-major axis of the asteroid's orbit, measured "
                    "in AU. It describes the characteristic size of the "
                    "orbital ellipse."
                ),

                value=1.50,

                elem_classes="input-box"
            )


        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column():


            i = gr.Number(

                label="Inclination (i)",

                info=(
                    "The angle between the asteroid's orbital plane and "
                    "the reference plane, measured in degrees."
                ),

                value=10.0,

                elem_classes="input-box"
            )


            om = gr.Number(

                label="Longitude of Ascending Node (Ω)",

                info=(
                    "Describes the orientation of the asteroid's orbital "
                    "plane relative to the reference direction, measured "
                    "in degrees."
                ),

                value=120.0,

                elem_classes="input-box"
            )


            w = gr.Number(

                label="Argument of Perihelion (ω)",

                info=(
                    "Describes the orientation of the asteroid's orbit "
                    "within its orbital plane, measured in degrees."
                ),

                value=180.0,

                elem_classes="input-box"
            )


            ma = gr.Number(

                label="Mean Anomaly (M)",

                info=(
                    "Describes the asteroid's position along its orbit "
                    "at a given time, measured in degrees."
                ),

                value=90.0,

                elem_classes="input-box"
            )


    # ========================================================
    # RANDOM VALUE BUTTON
    # ========================================================

    random_button = gr.Button(

        "GENERATE RANDOM VALUES",

        variant="secondary",

        elem_classes="pixel-button random-button"
    )


    random_button.click(

        fn=generate_random_values,

        inputs=[],

        outputs=[
            H,
            albedo,
            e,
            a,
            i,
            om,
            w,
            ma
        ]
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = gr.Button(

        "PREDICT MOID",

        variant="secondary",

        size="lg",

        elem_classes="pixel-button predict-button"
    )


    # ========================================================
    # RESULT TITLE
    # ========================================================

    gr.Markdown(
        "## Prediction Result",
        elem_classes="section-title"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    with gr.Row():

        output_au = gr.Textbox(

            label="Predicted MOID (AU)",

            interactive=False,

            elem_classes="result-box"
        )


        output_km = gr.Textbox(

            label="Predicted MOID (kilometers)",

            interactive=False,

            elem_classes="result-box"
        )


    # ========================================================
    # PREDICTION EVENT
    # ========================================================

    predict_button.click(

        fn=predict_moid,

        inputs=[
            H,
            albedo,
            e,
            a,
            i,
            om,
            w,
            ma
        ],

        outputs=[
            output_au,
            output_km
        ]
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    demo.launch(

        server_name="0.0.0.0",

        server_port=port,

        theme=gr.themes.Soft(),

        css=CSS
    )
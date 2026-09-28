from flask import Flask, render_template, request, redirect, session
import os
import tensorflow as tf

from fusion import multimodal_predict
from gradcam import generate_gradcam
from shap_explain import generate_shap
from explainability_engine import analyze_explanation_map
from agent import explain_gradcam, explain_shap, generate_clinical_interpretation

app = Flask(__name__)
app.secret_key = "smartcare_secret_key"

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

EXPLAIN_FILES = [
    "static/gradcam_xray.png",
    "static/gradcam_ct.png",
    "static/shap_xray.png",
    "static/shap_ct.png"
]

xray_model = tf.keras.models.load_model("xray_model_finetuned.h5", compile=False)
ct_model = tf.keras.models.load_model("ct_model_finetuned.h5", compile=False)


# =========================
# CLEAR SESSION
# =========================
@app.route("/clear")
def clear():

    session.clear()

    for f in EXPLAIN_FILES:
        if os.path.exists(f):
            os.remove(f)

    for f in ["uploads/xray.jpg", "uploads/ct.jpg"]:
        if os.path.exists(f):
            os.remove(f)

    return redirect("/predict")


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("home.html", active_page="home")


# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        xray = request.files.get("xray")
        ct = request.files.get("ct")

        if not xray or not ct:
            return render_template("predict.html",
                                   error="Upload both images",
                                   active_page="predict")

        xray_path = os.path.join(UPLOAD_FOLDER, "xray.jpg")
        ct_path = os.path.join(UPLOAD_FOLDER, "ct.jpg")

        xray.save(xray_path)
        ct.save(ct_path)

        xray_prob, ct_prob, fused_prob, confidence, conflict_status = multimodal_predict(
            xray_path, ct_path
        )

        diagnosis = "PNEUMONIA" if fused_prob >= 0.5 else "NORMAL"

        generate_gradcam(xray_model, xray_path, "Conv_1",
                         "static/gradcam_xray.png")

        generate_gradcam(ct_model, ct_path, "Conv_1",
                         "static/gradcam_ct.png")

        generate_shap(xray_path, "static/shap_xray.png")
        generate_shap(ct_path, "static/shap_ct.png")

        gradcam_stats = analyze_explanation_map("static/gradcam_xray.png")
        shap_stats = analyze_explanation_map("static/shap_xray.png")

        # Convert stats safely
        gradcam_stats = {k: float(v) for k, v in gradcam_stats.items()}
        shap_stats = {k: float(v) for k, v in shap_stats.items()}

        # Store SAFE Python types
        session["prediction"] = str(diagnosis)
        session["probability"] = float(round(float(fused_prob), 3))
        session["confidence"] = float(round(float(confidence), 3))

        session["clinical_report"] = str(generate_clinical_interpretation(
            diagnosis,
            float(round(float(fused_prob), 3)),
            float(round(float(confidence), 3)),
            conflict_status,
            gradcam_stats,
            shap_stats
        ))

        session["gradcam_xray_text"] = str(explain_gradcam("Chest X-ray", gradcam_stats))
        session["gradcam_ct_text"] = str(explain_gradcam("CT Scan", analyze_explanation_map("static/gradcam_ct.png")))

        session["shap_xray_text"] = str(explain_shap("Chest X-ray", shap_stats))
        session["shap_ct_text"] = str(explain_shap("CT Scan", analyze_explanation_map("static/shap_ct.png")))

        return redirect("/predict")

    return render_template(
        "predict.html",
        result=session.get("prediction"),
        probability=session.get("probability"),
        confidence=session.get("confidence"),
        clinical_report=session.get("clinical_report"),
        active_page="predict"
    )


# =========================
# GRADCAM
# =========================
@app.route("/gradcam")
def gradcam():

    explanations = {
        "xray": session.get("gradcam_xray_text"),
        "ct": session.get("gradcam_ct_text")
    }

    return render_template("gradcam.html",
                           explanations=explanations,
                           active_page="gradcam")


# =========================
# SHAP
# =========================
@app.route("/shap")
def shap():

    explanations = {
        "xray": session.get("shap_xray_text"),
        "ct": session.get("shap_ct_text")
    }

    return render_template("shap.html",
                           explanations=explanations,
                           active_page="shap")


if __name__ == "__main__":
    app.run(debug=True)
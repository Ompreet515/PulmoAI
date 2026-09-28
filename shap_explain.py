import shap
import tensorflow as tf
import numpy as np
import cv2
import os

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("xray_model_finetuned.h5")

# =========================
# SIMPLE LUNG MASK
# =========================
def get_lung_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]

    return mask


# =========================
# PREPROCESS
# =========================
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img


# =========================
# SHAP GENERATION 
# =========================
def generate_shap(img_path, output_path):

    # Use larger background set 
    background_dir = "data/xray/train/NORMAL"
    background = []

    for file in os.listdir(background_dir)[:30]:
        background.append(preprocess(os.path.join(background_dir, file)))

    background = np.array(background)

    test_img = preprocess(img_path)
    test_img_expanded = np.expand_dims(test_img, axis=0)

    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(test_img_expanded)[0]

    # Aggregate channels
    shap_map = np.mean(shap_values[0], axis=-1)

    # Smooth SHAP 
    shap_map = cv2.GaussianBlur(shap_map, (25, 25), 0)

    shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)

    orig = cv2.imread(img_path)

    shap_map = cv2.resize(shap_map, (orig.shape[1], orig.shape[0]))

    shap_map = np.uint8(255 * shap_map)
    shap_map = cv2.applyColorMap(shap_map, cv2.COLORMAP_JET)

    # Lung restriction
    lung_mask = get_lung_mask(orig)
    lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_GRAY2BGR)

    shap_map = cv2.bitwise_and(shap_map, lung_mask)

    overlay = cv2.addWeighted(orig, 0.65, shap_map, 0.35, 0)

    cv2.imwrite(output_path, overlay)

    print("Smoothed Lung-Restricted SHAP Saved")
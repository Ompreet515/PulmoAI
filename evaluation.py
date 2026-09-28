import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# =========================
# PATHS (UPDATE IF NEEDED)
# =========================
XRAY_MODEL_PATH = "xray_model_finetuned.h5"
CT_MODEL_PATH = "ct_model_finetuned.h5"

XRAY_VAL_DIR = "data/xray/val"
XRAY_TEST_DIR = "data/xray/test"

CT_VAL_DIR = "data/ct/val"
CT_TEST_DIR = "data/ct/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(rescale=1./255)

def load_data(directory):
    return datagen.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

# =========================
# LOAD MODELS
# =========================
xray_model = load_model(XRAY_MODEL_PATH)
ct_model = load_model(CT_MODEL_PATH)

# =========================
# LOAD DATA
# =========================
print("\nLoading Data...")

xray_val = load_data(XRAY_VAL_DIR)
xray_test = load_data(XRAY_TEST_DIR)

ct_val = load_data(CT_VAL_DIR)
ct_test = load_data(CT_TEST_DIR)

# =========================
# OPTIMAL THRESHOLD FUNCTION
# =========================
def get_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresh = 0.5
    best_acc = 0

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        acc = accuracy_score(y_true, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_thresh

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, val_data, test_data, name):
    print(f"\n==============================")
    print(f"Evaluating: {name}")
    print(f"==============================")

    # Validation predictions (for threshold tuning)
    val_prob = model.predict(val_data).ravel()
    y_val = val_data.classes

    best_thresh = get_best_threshold(y_val, val_prob)

    print(f"Optimal Threshold: {best_thresh:.4f}")

    # Test predictions
    test_prob = model.predict(test_data).ravel()
    y_test = test_data.classes
    y_pred = (test_prob > best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\nResults:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return y_test, y_pred, test_prob, best_thresh

# =========================
# RUN EVALUATION
# =========================

y_true_xray, y_pred_xray, y_prob_xray, thresh_x = evaluate_model(
    xray_model, xray_val, xray_test, "X-Ray Model"
)

y_true_ct, y_pred_ct, y_prob_ct, thresh_c = evaluate_model(
    ct_model, ct_val, ct_test, "CT Model"
)

# =========================
# FUSION (CONFIDENCE-AWARE)
# =========================
print("\n==============================")
print("Confidence-Aware Fusion")
print("==============================")

# Ensure same length
min_len = min(len(y_prob_xray), len(y_prob_ct))

y_prob_xray = y_prob_xray[:min_len]
y_prob_ct = y_prob_ct[:min_len]
y_true = y_true_xray[:min_len]

# Confidence scores (simple)
conf_xray = np.max([y_prob_xray, 1 - y_prob_xray], axis=0)
conf_ct = np.max([y_prob_ct, 1 - y_prob_ct], axis=0)

# Fusion
fusion_prob = (y_prob_xray * conf_xray + y_prob_ct * conf_ct) / (conf_xray + conf_ct + 1e-8)

fusion_pred = (fusion_prob > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_true, fusion_pred)
prec = precision_score(y_true, fusion_pred)
rec = recall_score(y_true, fusion_pred)
f1 = f1_score(y_true, fusion_pred)
auc = roc_auc_score(y_true, fusion_prob)

print(f"Fusion Accuracy  : {acc:.4f}")
print(f"Fusion Precision : {prec:.4f}")
print(f"Fusion Recall    : {rec:.4f}")
print(f"Fusion F1 Score  : {f1:.4f}")
print(f"Fusion ROC AUC   : {auc:.4f}")

# =========================
# SAVE FILES FOR FIGURES
# =========================
np.save("y_true_xray.npy", y_true_xray)
np.save("y_pred_xray.npy", y_pred_xray)
np.save("y_prob_xray.npy", y_prob_xray)

np.save("y_true_ct.npy", y_true_ct)
np.save("y_pred_ct.npy", y_pred_ct)
np.save("y_prob_ct.npy", y_prob_ct)

np.save("y_prob_fusion.npy", fusion_prob)

print("\nSaved all prediction files for figures.")
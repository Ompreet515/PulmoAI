import tensorflow as tf
import numpy as np
import cv2

xray_model = tf.keras.models.load_model("xray_model_finetuned.h5", compile=False)
ct_model = tf.keras.models.load_model("ct_model_finetuned.h5", compile=False)


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def multimodal_predict(xray_img, ct_img):

    px = xray_model.predict(preprocess(xray_img))[0][0]
    pc = ct_model.predict(preprocess(ct_img))[0][0]

    conf_xray = abs(px - 0.5) * 2
    conf_ct = abs(pc - 0.5) * 2

    conflict_status = "Agreement"

    if (px > 0.5 and pc < 0.5) or (px < 0.5 and pc > 0.5):
        conflict_status = "Disagreement between X-ray and CT"

    fused_prob = (px * conf_xray + pc * conf_ct) / (conf_xray + conf_ct + 1e-6)

    confidence = max(conf_xray, conf_ct)

    return px, pc, fused_prob, confidence, conflict_status
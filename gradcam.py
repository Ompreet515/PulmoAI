import tensorflow as tf
import numpy as np
import cv2

# =========================
# SIMPLE LUNG MASKING 
# =========================
def get_lung_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (lungs darker)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Clean mask
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]

    return mask


# =========================
# GRAD-CAM CORE
# =========================
def compute_gradcam(model, img_array, layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


# =========================
# SMOOTHED + LUNG-RESTRICTED GRAD-CAM
# =========================
def generate_gradcam(model, img_path, layer_name, output_path):

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Smoothed Grad-CAM (average multiple maps)
    heatmaps = []

    for i in range(5):
        noise = np.random.normal(0, 0.02, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1)

        heatmap = compute_gradcam(model, noisy_img, layer_name)
        heatmaps.append(heatmap)

    heatmap = np.mean(heatmaps, axis=0)

    # Resize to original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Lung restriction
    lung_mask = get_lung_mask(img)
    lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_GRAY2BGR)

    heatmap = cv2.bitwise_and(heatmap, lung_mask)

    overlay = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

    cv2.imwrite(output_path, overlay)

    print("Smoothed Lung-Restricted Grad-CAM Saved")
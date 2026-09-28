import cv2
import numpy as np

def analyze_explanation_map(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read explanation image: {image_path}")

    # Normalize channels
    red = img[:, :, 2] / 255.0
    blue = img[:, :, 0] / 255.0

    h, w = red.shape

    # --- INTENSITY FEATURES ---
    strong_red = np.mean(red > 0.6)
    strong_blue = np.mean(blue > 0.6)
    avg_red = np.mean(red)

    # --- SPATIAL FEATURES ---
    center = red[h//4:3*h//4, w//4:3*w//4]

    top = red[:h//4, :]
    bottom = red[3*h//4:, :]
    left = red[:, :w//4]
    right = red[:, 3*w//4:]

    center_focus = np.mean(center)
    periphery_focus = np.mean([np.mean(top), np.mean(bottom),
                               np.mean(left), np.mean(right)])

    return {
        "strong_red": float(strong_red),
        "strong_blue": float(strong_blue),
        "avg_red": float(avg_red),
        "center_focus": float(center_focus),
        "periphery_focus": float(periphery_focus)
    }

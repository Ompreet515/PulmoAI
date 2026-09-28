# PulmoAI
### Multimodal Pneumonia Detection Using Deep Learning and Explainable AI

---

# 1. Overview

PulmoAI is an AI-powered healthcare application developed for automated pneumonia detection using both chest X-ray and CT scan images. The framework employs a multimodal deep learning approach by utilizing separate MobileNetV2-based convolutional neural networks for each imaging modality and combining their predictions through a confidence-aware fusion mechanism. To improve transparency and trustworthiness, PulmoAI integrates Explainable AI techniques such as Grad-CAM and SHAP, enabling visual and feature-level interpretation of model decisions. An Agentic AI module further generates human-readable explanations for the prediction results. The complete system is deployed through a Flask-based web application for real-time image upload, prediction, and visualization.

---

# 2. Features

- Automated pneumonia detection using chest X-ray images.
- Automated pneumonia detection using CT scan images.
- MobileNetV2-based transfer learning architecture.
- Confidence-aware multimodal fusion for improved prediction reliability.
- Grad-CAM heatmap generation for visual interpretability.
- SHAP-based feature contribution analysis.
- Agentic AI-generated diagnostic explanations.
- Flask-based web interface for real-time prediction and visualization.
- User-friendly dashboard for image upload and result display.

---

# 3. Methodology / Workflow

```text
Input Images (Chest X-ray & CT Scan)
                ↓
      Image Preprocessing
                ↓
      MobileNetV2 Models
     (X-ray & CT Models)
                ↓
  Modality-Specific Classification
                ↓
    Confidence-Aware Fusion
                ↓
      Pneumonia Prediction
                ↓
       Grad-CAM & SHAP
                ↓
     Agentic AI Explanation
                ↓
      Flask Web Dashboard
```

### Workflow Description

1. Chest X-ray and CT scan images are uploaded through the web application.
2. Images are preprocessed using resizing and normalization techniques.
3. Separate MobileNetV2 models perform feature extraction and classification.
4. Predictions from both modalities are combined using confidence-aware fusion.
5. Grad-CAM and SHAP generate visual and feature-level explanations.
6. Agentic AI generates human-readable diagnostic interpretations.
7. Results are displayed through the Flask dashboard.

---

# 4. Performance

| Metric | Value |
|----------|----------|
| Accuracy | 92.8% |
| Precision | 92.4% |
| Recall | 96.4% |
| F1-Score | 94.4% |

The proposed PulmoAI framework demonstrates strong performance in pneumonia detection by effectively combining chest X-ray and CT scan information. The high recall value indicates improved sensitivity in identifying pneumonia-positive cases, which is particularly important in healthcare applications.

---

# 5. Project Structure

```text
PulmoAI/
│
├── app.py
├── agent.py
├── fusion.py
├── gradcam.py
├── shap_explain.py
├── explainability_engine.py
├── evaluation.py
├── train_xray.py
├── train_ct.py
│
├── models/
│   ├── xray_model_finetuned.h5
│   └── ct_model_finetuned.h5
│
├── templates/
│   ├── home.html
│   ├── predict.html
│   ├── gradcam.html
│   ├── shap.html
│   └── navbar.html
│
├── static/
│   └── style.css
│
├── uploads/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 6. Installation

### Clone the Repository

```bash
git clone https://github.com/USERNAME/PulmoAI.git
cd PulmoAI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 7. Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

Upload chest X-ray and CT scan images through the dashboard to generate predictions and explanations.

---

# 8. Dataset

The datasets used for training are not included in this repository due to size limitations.

### Chest X-ray Dataset

Chest X-Ray Pneumonia Dataset (Kaggle)

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### CT Scan Dataset

SARS-CoV-2 CT Scan Dataset (Kaggle)

https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset

After downloading, place the datasets inside the `data/` directory following the required folder structure.

---

# 9. Future Work

- Train the framework on larger and more diverse clinical datasets.
- Explore Vision Transformer-based multimodal architectures.
- Develop a cloud-based deployment for remote healthcare access.
- Extend the framework to support multiple lung disease classifications.
- Develop a mobile-friendly version for point-of-care diagnostics.
- Integrate real-time clinical reporting and decision support features.

---

# 10. Author

**Ompreet**  
B.Tech – Computer Science and Engineering (Artificial Intelligence & Machine Learning)  
SRM Institute of Science and Technology, Kattankulathur

---

## Project Description

PulmoAI is a multimodal deep learning framework designed for automated pneumonia detection using chest X-ray and CT scan images. The system utilizes separate MobileNetV2-based models for modality-specific classification and combines their predictions through a confidence-aware fusion mechanism to improve diagnostic reliability. To enhance transparency and trustworthiness, PulmoAI integrates Explainable AI techniques including Grad-CAM and SHAP, providing visual and feature-level interpretations of model predictions. An Agentic AI module further generates human-readable explanations to assist users in understanding diagnostic outcomes. The framework is deployed through a Flask-based web application that enables real-time image upload, prediction, and visualization. By combining multimodal learning, explainable AI, and intelligent deployment, PulmoAI offers an accurate, interpretable, and practical solution for AI-assisted pneumonia diagnosis.

## Author
Ompreet Choudhury
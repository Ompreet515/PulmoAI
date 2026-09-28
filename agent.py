import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"


def query_local_llm(prompt):

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]

    except Exception as e:
        return f"LLM Error: {str(e)}"


def explain_gradcam(modality, stats):

    prompt = f"""
You are an AI radiology expert.

Modality: {modality}
GradCAM Statistics: {stats}

Write EXACTLY TWO paragraphs:

Paragraph 1:
Medical explanation of activation patterns.

Paragraph 2:
Simple explanation understandable by a patient.

Do NOT use bullet points.
"""

    return query_local_llm(prompt)


def explain_shap(modality, stats):

    prompt = f"""
You are an AI radiology expert.

Modality: {modality}
SHAP Statistics: {stats}

Write two paragraphs with bold headings.

Medical Interpretation: Explain Grad-CAM and SHAP results from X-ray and CT images, highlighting clinically relevant regions linked to pneumonia.

Patient Explanation: Describe the same findings in simple, non-technical language for a patient for both Grad-CAM and SHAP images.

No bullet points, emojis, or formatting—plain text only.
"""

    return query_local_llm(prompt)


def generate_clinical_interpretation(prediction, probability, confidence,
                                     conflict_status,
                                     gradcam_stats,
                                     shap_stats):

    prompt = f"""
You are an AI clinical assistant.

Diagnosis: {prediction}
Probability: {probability}
Confidence: {confidence}

Write EXACTLY TWO paragraphs:

Paragraph 1:
Clinical medical interpretation.

Paragraph 2:
Simple explanation + suggested next steps.

Do NOT use bullet points.
"""

    return query_local_llm(prompt)
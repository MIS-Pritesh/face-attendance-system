import streamlit as st
import os
from deepface import DeepFace
from PIL import Image
import numpy as np

EMPLOYEE_FOLDER = "data/employee_images"

def get_employee_images():
    """Return list of full paths of images inside employee_images folder."""
    files = []
    for f in os.listdir(EMPLOYEE_FOLDER):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(EMPLOYEE_FOLDER, f))
    return files

def recognize_face(captured_img):
    """Compare captured image with all employee images."""
    employee_imgs = get_employee_images()
    if not employee_imgs:
        return False, "No employee images found!"

    # Convert to numpy
    captured_np = np.array(captured_img.convert("RGB"))

    # Extract embedding for captured image
    captured_embed = DeepFace.represent(
        img_path=captured_np,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    # Compare with stored employee embeddings
    for img_path in employee_imgs:
        emp_img = Image.open(img_path)
        emp_np = np.array(emp_img.convert("RGB"))

        emp_embed = DeepFace.represent(
            img_path=emp_np,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        distance = np.linalg.norm(np.array(captured_embed) - np.array(emp_embed))

        if distance < 10:  # threshold
            return True, f"Face Recognized! (Matched: {os.path.basename(img_path)})"

    return False, "Face NOT recognized."


# ---------------- STREAMLIT UI --------------------

st.title("📸 Live Face Recognition (Employee Verification)")

st.write("👉 Capture your face using the camera below")

# Live camera input
captured = st.camera_input("")

if captured is not None:
    # Read captured image
    image = Image.open(captured)

    st.info("🔍 Analyzing... Please wait")

    # Run recognition automatically
    status, message = recognize_face(image)

    if status:
        st.success(message)
    else:
        st.error(message)

import streamlit as st
import os
from PIL import Image
import numpy as np
from deepface import DeepFace

EMPLOYEE_FOLDER = "data/employee_images"

def get_employee_images():
    """Return list of full paths of images inside employee_images folder."""
    files = []
    for f in os.listdir(EMPLOYEE_FOLDER):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(EMPLOYEE_FOLDER, f))
    return files

def recognize_face(uploaded_img):
    """Compare uploaded image with all employee images."""
    employee_imgs = get_employee_images()

    if not employee_imgs:
        return False, "No employee images found!"

    try:
        # Convert uploaded image to numpy
        uploaded_np = np.array(uploaded_img.convert("RGB"))

        # Compute embedding of uploaded image
        uploaded_embed = DeepFace.represent(
            img_path=uploaded_np,
            model_name='Facenet',
            enforce_detection=False
        )[0]["embedding"]

        # Compare with each employee image
        for emp_path in employee_imgs:
            emp_img = Image.open(emp_path)
            emp_np = np.array(emp_img.convert("RGB"))

            emp_embed = DeepFace.represent(
                img_path=emp_np,
                model_name='Facenet',
                enforce_detection=False
            )[0]["embedding"]

            distance = np.linalg.norm(
                np.array(uploaded_embed) - np.array(emp_embed)
            )

            if distance < 10:   # threshold
                return True, f"Face Recognized (matched with {os.path.basename(emp_path)})"

        return False, "Face NOT recognized."

    except Exception as e:
        return False, f"Error processing image: {str(e)}"


# ------------------- STREAMLIT UI --------------------

st.title("🔍 Face Recognition - Employee Verification")

uploaded_file = st.file_uploader("Upload face image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Check Employee"):
        status, message = recognize_face(img)

        if status:
            st.success(message)
        else:
            st.error(message)

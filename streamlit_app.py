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

def recognize_face(img):
    """Compare input image with all employee images."""
    employee_imgs = get_employee_images()
    if not employee_imgs:
        return False, "No employee images found in data/employee_images."

    try:
        img_np = np.array(img.convert("RGB"))
        img_embed = DeepFace.represent(
            img_path=img_np,
            model_name='Facenet',
            enforce_detection=False
        )[0]["embedding"]

        for emp_path in employee_imgs:
            emp_img = Image.open(emp_path)
            emp_np = np.array(emp_img.convert("RGB"))

            emp_embed = DeepFace.represent(
                img_path=emp_np,
                model_name='Facenet',
                enforce_detection=False
            )[0]["embedding"]

            distance = np.linalg.norm(np.array(img_embed) - np.array(emp_embed))

            if distance < 10:  # Threshold
                return True, f"Face Recognized! (Matched with {os.path.basename(emp_path)})"

        return False, "Face NOT recognized."

    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"


# ---------------- STREAMLIT UI --------------------

st.title("🧑‍💼 Employee Face Recognition")

# OPTION SELECTOR
mode = st.radio(
    "Choose input method:",
    ("Upload Image", "Capture Live Image")
)

image = None

# 1️⃣ UPLOAD IMAGE OPTION
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload face image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded)

# 2️⃣ CAPTURE LIVE IMAGE OPTION
elif mode == "Capture Live Image":
    captured = st.camera_input("Capture your face")
    if captured is not None:
        image = Image.open(captured)

# When we have an image, process automatically
if image is not None:
    st.info("🔍 Processing image... please wait.")
    status, message = recognize_face(image)

    if status:
        st.success(message)
    else:
        st.error(message)

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image
import os
from datetime import datetime

EMPLOYEE_FILE = "data/employees.csv"
ATTENDANCE_FILE = "data/attendance.csv"

def load_employees():
    if not os.path.exists(EMPLOYEE_FILE):
        df = pd.DataFrame(columns=["id", "name", "path"])
        df.to_csv(EMPLOYEE_FILE, index=False)
    return pd.read_csv(EMPLOYEE_FILE)

def save_employee(df):
    df.to_csv(EMPLOYEE_FILE, index=False)

def load_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["id", "name", "date", "time", "status"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    return pd.read_csv(ATTENDANCE_FILE)

def save_attendance(df):
    df.to_csv(ATTENDANCE_FILE, index=False)

def register_employee(name, emp_id, img):
    df = load_employees()

    if emp_id in df["id"].astype(str).values:
        return False, f"Employee ID {emp_id} already registered."

    # Save image to folder
    save_path = f"data/employee_images/{emp_id}.png"
    img.save(save_path)

    # Append new employee to dataframe
    df.loc[len(df)] = [str(emp_id), name, save_path]
    save_employee(df)

    return True, f"Employee {name} with ID {emp_id} registered successfully."

def recognize_face(img):
    employees = load_employees()
    if employees.empty:
        return None, "No employees registered."

    uploaded_img = np.array(img.convert('RGB'))
    uploaded_embedding = DeepFace.represent(img_path=uploaded_img, model_name='Facenet', enforce_detection=False)[0]["embedding"]

    for _, row in employees.iterrows():
        emp_img_path = row["path"]
        if not os.path.exists(emp_img_path):
            continue
        emp_img = Image.open(emp_img_path)
        emp_img_np = np.array(emp_img.convert('RGB'))
        emp_embedding = DeepFace.represent(img_path=emp_img_np, model_name='Facenet', enforce_detection=False)[0]["embedding"]

        distance = np.linalg.norm(np.array(uploaded_embedding) - np.array(emp_embedding))
        if distance < 10:  # Threshold for recognition, adjust as needed
            return row["id"], f"Employee recognized: {row['name']}"

    return None, "Face not recognized."

def mark_attendance(emp_id, name):
    df = load_attendance()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    status = "Present"

    # Check if attendance already marked for today
    attendance_today = df[(df["id"] == emp_id) & (df["date"] == date_str)]
    if not attendance_today.empty:
        return False, "Attendance already marked for today."

    df.loc[len(df)] = [emp_id, name, date_str, time_str, status]
    save_attendance(df)
    return True, "Attendance marked successfully."

# Streamlit UI
st.title("Face Recognition Attendance System")

menu = ["Register Employee", "Recognize Employee"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register Employee":
    st.subheader("Register a New Employee")
    name = st.text_input("Employee Name")
    emp_id = st.text_input("Employee ID")
    img_file = st.file_uploader("Upload Employee Face Image", type=["png", "jpg", "jpeg"])

    if st.button("Register"):
        if not name or not emp_id or not img_file:
            st.error("Please fill all fields and upload an image.")
        else:
            img = Image.open(img_file)
            status, msg = register_employee(name, emp_id, img)
            if status:
                st.success(msg)
            else:
                st.error(msg)

elif choice == "Recognize Employee":
    st.subheader("Recognize Employee Face")
    img_file = st.file_uploader("Upload Face Image", type=["png", "jpg", "jpeg"])

    if img_file is not None:
        img = Image.open(img_file)
        emp_id, msg = recognize_face(img)
        if emp_id:
            st.success(msg)
            employees = load_employees()
            emp_name = employees.loc[employees["id"] == emp_id, "name"].values[0]

            mark_status, mark_msg = mark_attendance(emp_id, emp_name)
            if mark_status:
                st.success(mark_msg)
            else:
                st.warning(mark_msg)
        else:
            st.error(msg)

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from PIL import Image
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------

st.set_page_config(page_title="Face Attendance System", page_icon="📸", layout="wide")

# Create data folders
Path("data").mkdir(exist_ok=True)
Path("data/employee_images").mkdir(exist_ok=True)

EMPLOYEE_FILE = "data/employees.csv"
ATTENDANCE_FILE = "data/attendance.csv"

# Create CSV files if not exist
if not os.path.exists(EMPLOYEE_FILE):
    pd.DataFrame(columns=["id", "name", "image"]).to_csv(EMPLOYEE_FILE, index=False)

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["id", "name", "date", "time", "status"]).to_csv(ATTENDANCE_FILE, index=False)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

def load_employees():
    return pd.read_csv(EMPLOYEE_FILE)

def load_attendance():
    return pd.read_csv(ATTENDANCE_FILE)


# ---------------------------------------------------
# REGISTER EMPLOYEE
# ---------------------------------------------------

def register_employee(name, emp_id, image):
    save_path = f"data/employee_images/{emp_id}.jpg"
    image.save(save_path)

    df = load_employees()

    if emp_id in df["id"].values:
        return False, "Employee ID already exists!"

    df.loc[len(df)] = [emp_id, name, save_path]
    df.to_csv(EMPLOYEE_FILE, index=False)

    return True, "Employee registered successfully!"


# ---------------------------------------------------
# FACE RECOGNITION USING DEEPFACE
# ---------------------------------------------------

def recognize_face(uploaded_img):
    employees = load_employees()

    if employees.empty:
        return None, "No employees registered"

    uploaded_img.save("temp.jpg")

    for index, row in employees.iterrows():
        try:
            result = DeepFace.verify(
                img1_path="temp.jpg",
                img2_path=row["image"],
                model_name="Facenet",
                enforce_detection=False
            )

            if result["verified"]:
                return row["id"], f"Match found (distance: {result['distance']:.3f})"

        except:
            continue

    return None, "No match found"


# ---------------------------------------------------
# MARK ATTENDANCE
# ---------------------------------------------------

def mark_attendance(emp_id):
    employees = load_employees()
    attendance = load_attendance()

    emp = employees[employees["id"] == emp_id]
    today = datetime.now().strftime("%Y-%m-%d")

    if not emp.empty:
        name = emp.iloc[0]["name"]
    else:
        return False, "Employee ID not found!"

    # Check duplicate entry
    if not attendance[(attendance["id"] == emp_id) & (attendance["date"] == today)].empty:
        return False, "Attendance already marked today"

    new_record = {
        "id": emp_id,
        "name": name,
        "date": today,
        "time": datetime.now().strftime("%H:%M:%S"),
        "status": "Present"
    }

    attendance.loc[len(attendance)] = new_record
    attendance.to_csv(ATTENDANCE_FILE, index=False)

    return True, f"Attendance marked for {name}"


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.title("📸 Face Recognition Attendance System")
st.markdown("---")

menu = st.sidebar.radio("Navigation", ["Register Employee", "Mark Attendance", "View Records"])


# ---------------------------------------------------
# PAGE 1: REGISTER EMPLOYEE
# ---------------------------------------------------

if menu == "Register Employee":
    st.header("Register New Employee")

    name = st.text_input("Employee Name")
    emp_id = st.text_input("Employee ID")
    uploaded = st.file_uploader("Upload Employee Photo", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)

    if st.button("Register Employee"):
        if not uploaded or not name or not emp_id:
            st.warning("Please fill all fields")
        else:
            status, msg = register_employee(name, emp_id, img)
            if status:
                st.success(msg)
                st.balloons()
            else:
                st.error(msg)

    st.subheader("Registered Employees")
    st.dataframe(load_employees(), use_container_width=True)


# ---------------------------------------------------
# PAGE 2: MARK ATTENDANCE
# ---------------------------------------------------

elif menu == "Mark Attendance":
    st.header("Mark Attendance")

    uploaded = st.file_uploader("Upload image for recognition", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=300)

        if st.button("Recognize & Mark Attendance"):
            emp_id, msg = recognize_face(img)

            if emp_id:
                st.success(msg)
                ok, a_msg = mark_attendance(emp_id)
                if ok:
                    st.success(a_msg)
                    st.balloons()
                else:
                    st.warning(a_msg)
            else:
                st.error(msg)

    st.subheader("Today's Attendance")
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    st.dataframe(df[df["date"] == today], use_container_width=True)


# ---------------------------------------------------
# PAGE 3: VIEW RECORDS
# ---------------------------------------------------

elif menu == "View Records":
    st.header("Attendance History")
    df = load_attendance()

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")

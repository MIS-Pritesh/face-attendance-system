import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Attendance System", page_icon="📸", layout="wide")

st.title("📸 Face Attendance System")
st.info("⚠️ This is a simplified version. Face recognition will be added once deployment is stable.")

# Initialize session state
if 'employees' not in st.session_state:
    st.session_state.employees = []
if 'attendance' not in st.session_state:
    st.session_state.attendance = []

# Sidebar
page = st.sidebar.radio("Navigation", ["Register Employee", "Mark Attendance", "View Records"])

if page == "Register Employee":
    st.header("Register New Employee")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Employee Name")
        emp_id = st.text_input("Employee ID")
        
        uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            if st.button("Register Employee"):
                if name and emp_id:
                    st.session_state.employees.append({
                        'name': name,
                        'id': emp_id,
                        'registered': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success(f"✅ {name} registered successfully!")
                    st.balloons()
                else:
                    st.warning("Please fill all fields")
    
    with col2:
        st.subheader("Registered Employees")
        if st.session_state.employees:
            for emp in st.session_state.employees:
                st.write(f"**{emp['name']}** (ID: {emp['id']})")
        else:
            st.info("No employees yet")

elif page == "Mark Attendance":
    st.header("Mark Attendance")
    
    if st.session_state.employees:
        selected_emp = st.selectbox("Select Employee", 
                                    [f"{e['name']} ({e['id']})" for e in st.session_state.employees])
        
        if st.button("Mark Present"):
            emp_name = selected_emp.split(" (")[0]
            st.session_state.attendance.append({
                'name': emp_name,
                'date': datetime.now().strftime("%Y-%m-%d"),
                'time': datetime.now().strftime("%H:%M:%S"),
                'status': 'Present'
            })
            st.success(f"✅ Attendance marked for {emp_name}")
            st.balloons()
    else:
        st.warning("No employees registered yet")

else:
    st.header("Attendance Records")
    
    if st.session_state.attendance:
        df = pd.DataFrame(st.session_state.attendance)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")
    else:
        st.info("No attendance records yet")

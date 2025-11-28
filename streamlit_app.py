from deepface import DeepFace
import os
import cv2

EMPLOYEE_IMAGES_DIR = "data/employee_images"

def recognize_employee_from_images(uploaded_img):
    # Convert Streamlit uploaded file to numpy array for DeepFace
    img_array = np.array(uploaded_img) if not isinstance(uploaded_img, np.ndarray) else uploaded_img
    
    # Get embedding for uploaded image
    uploaded_embedding = DeepFace.represent(img_array, model_name="Facenet")[0]["embedding"]
    
    # Iterate over stored employee images
    for filename in os.listdir(EMPLOYEE_IMAGES_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            employee_img_path = os.path.join(EMPLOYEE_IMAGES_DIR, filename)
            employee_img = cv2.imread(employee_img_path)
            
            # Get embedding for stored employee image
            employee_embedding = DeepFace.represent(employee_img, model_name="Facenet")[0]["embedding"]
            
            # Calculate distance between embeddings (Euclidean or cosine)
            distance = np.linalg.norm(np.array(uploaded_embedding) - np.array(employee_embedding))
            
            # Threshold to determine match - you may tune this value
            if distance < 10:
                # Assuming filename format 'a1.png' -> id 'a1'
                emp_id = os.path.splitext(filename)[0]
                return emp_id, f"Employee recognized with ID: {emp_id}"
    
    return None, "Employee not recognized"

import streamlit as st
import cv2
import numpy as np
import os
import datetime
from openvino.runtime import Core
import pandas as pd

# Paths
MODEL_DET_XML = "models/face-detection-0200.xml"
MODEL_DET_BIN = "models/face-detection-0200.bin"
FACE_DB = "face_db"
SNAPSHOT_DB = "attendance_snapshots"
LOG_PATH = "attendance_log.csv"
ADMIN_PASSWORD = "admin123"

os.makedirs(FACE_DB, exist_ok=True)
os.makedirs(SNAPSHOT_DB, exist_ok=True)

# Load OpenVINO Model
ie = Core()
det_model = ie.read_model(model=MODEL_DET_XML, weights=MODEL_DET_BIN)
det_compiled = ie.compile_model(det_model, "CPU")

def detect_face(frame):
    resized = cv2.resize(frame, (256, 256))
    input_image = resized.transpose((2, 0, 1)).reshape(1, 3, 256, 256).astype(np.float32)
    results = det_compiled([input_image])[det_compiled.output(0)]
    height, width = frame.shape[:2]
    for det in results[0][0]:
        if det[2] > 0.5:
            xmin = int(det[3] * width)
            ymin = int(det[4] * height)
            xmax = int(det[5] * width)
            ymax = int(det[6] * height)
            return frame[ymin:ymax, xmin:xmax]
    return None

# Streamlit UI
st.set_page_config(page_title="1VISION Attendance System 🏫", layout="centered")
st.title("Smart Face Recognition Attendance System 📸")

menu = ["Student", "Admin"]
choice = st.sidebar.selectbox("Select Role", menu)

if choice == "Student":
    st.subheader("Student 🧑‍🎓 | Take attendance")
    img = st.camera_input("Take a pic of yourself 😛")

    if img is not None:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        face_crop = detect_face(frame)

        if face_crop is not None:
            matched = False
            for file in os.listdir(FACE_DB):
                ref_img = cv2.imread(os.path.join(FACE_DB, file))
                if ref_img is None:
                    continue
                try:
                    test_face = cv2.resize(face_crop, (100, 100))
                    ref_face = cv2.resize(ref_img, (100, 100))
                    diff = np.mean(cv2.absdiff(test_face, ref_face))
                    if diff < 30:
                        registered_name = os.path.splitext(file)[0]
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        img_path = f"{SNAPSHOT_DB}/{registered_name}_{timestamp}.jpg"
                        cv2.imwrite(img_path, face_crop)
                        with open(LOG_PATH, "a") as f:
                            f.write(f"{registered_name},{timestamp},{img_path}\n")
                        st.success(f"Welcome {registered_name}, attendance marked! 📝")
                        matched = True
                        break
                except:
                    continue
            if not matched:
                st.error("Unauthorised person, please try again. ❌")
        else:
            st.warning("No face detected. 😓")

elif choice == "Admin":
    st.subheader("Admin Login 🔐")
    admin_pw = st.text_input("Enter admin password", type="password")

    if admin_pw == ADMIN_PASSWORD:
        st.success("Logged in as Admin.")

        st.subheader("Register New Student Face ➕")
        new_name = st.text_input("Enter student name")
        reg_img = st.camera_input("Capture face for registration")

        if new_name and reg_img is not None:
            file_bytes = np.asarray(bytearray(reg_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            face_crop = detect_face(frame)

            if face_crop is not None:
                save_path = os.path.join(FACE_DB, f"{new_name}.jpg")
                cv2.imwrite(save_path, face_crop)
                st.success(f"Face registered for {new_name} ✅ ")
            else:
                st.warning("No face detected in the image. 😓")

        st.subheader("Attendance Logs 📋")
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH, header=None, names=["Name", "Timestamp", "Face Image"])
            st.dataframe(df)

            st.subheader("Face Snapshots (Last 24 hrs) 🖼️")
            now = datetime.datetime.now()
            updated_rows = []

            for _, row in df.iterrows():
                try:
                    ts = datetime.datetime.strptime(row["Timestamp"], "%Y-%m-%d_%H-%M-%S")
                    age = (now - ts).total_seconds()
                    if age <= 86400:
                        st.image(row["Face Image"], width=150, caption=f'{row["Name"]} @ {row["Timestamp"]}')
                        updated_rows.append(row)
                    else:
                        if os.path.exists(row["Face Image"]):
                            os.remove(row["Face Image"])
                except:
                    continue

            df_updated = pd.DataFrame(updated_rows)
            df_updated.to_csv(LOG_PATH, index=False, header=False)

            st.download_button(
                label="⬇️ Download Attendance CSV",
                data=df_updated.to_csv(index=False).encode("utf-8"),
                file_name="attendance_logs.csv",
                mime="text/csv"
            )
        else:
            st.info("No attendance logs found yet.")
    elif admin_pw != "":
        st.error("Incorrect password ❌")
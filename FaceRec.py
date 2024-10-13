import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import pickle

# Khởi tạo MTCNN và FaceNet
@st.cache_resource
def load_models():
    return MTCNN(), FaceNet()

detector, facenet = load_models()

# Load embeddings đã được tính toán trước
@st.cache_data
def load_embeddings():
    with open('id_card_embeddings.pkl', 'rb') as f:
        return pickle.load(f)

id_card_embeddings = load_embeddings()

def detect_and_align_face(image):
    faces = detector.detect_faces(image)
    if not faces:
        return None
    x, y, width, height = faces[0]['box']
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))
    return face

def extract_features(face):
    face = (face.astype('float32') - 127.5) / 128.0
    embedding = facenet.embeddings(np.array([face]))
    return embedding[0]

def compare_faces(embedding1, embedding2, threshold=0.7):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity > threshold, similarity

def verify_face(person_image):
    person_image = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)
    person_face = detect_and_align_face(person_image)
    if person_face is None:
        return None, 0, "Không thể phát hiện khuôn mặt trong ảnh"
    person_embedding = extract_features(person_face)
    
    best_match = None
    best_similarity = -1
    
    for filename, id_card_embedding in id_card_embeddings.items():
        similarity = cosine_similarity([person_embedding], [id_card_embedding])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = filename
    
    is_match = best_similarity > 0.7  # Có thể điều chỉnh ngưỡng này
    return is_match, best_similarity, best_match

st.title('Ứng dụng Xác thực Khuôn mặt Sinh viên')

uploaded_image = st.file_uploader("Tải lên ảnh chân dung của sinh viên", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    is_match, similarity, best_match = verify_face(image)
    
    if is_match is None:
        st.error("Không thể phát hiện khuôn mặt trong ảnh.")
    elif is_match:
        st.success(f"Xác thực thành công! Đây là sinh viên trong ảnh: {os.path.splitext(best_match)[0]}")
        st.write(f"Độ tương đồng: {similarity:.2f}")
    else:
        st.warning("Không tìm thấy sinh viên phù hợp trong cơ sở dữ liệu.")
        st.write(f"Độ tương đồng cao nhất: {similarity:.2f}")

    # Hiển thị thông tin chi tiết
    st.subheader("Thông tin chi tiết:")
    st.write(f"Tên file ảnh phù hợp nhất: {best_match}")
    st.write(f"Độ tương đồng: {similarity:.4f}")
    st.write(f"Ngưỡng xác thực: 0.7")

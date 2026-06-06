import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("emotion_ann.keras")

classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

st.title("Emotion Recognition using ANN")

uploaded_file = st.file_uploader(
    "Chọn ảnh khuôn mặt",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Ảnh đã chọn",
        width=300
    )

    # Chuyển grayscale
    image = image.convert("L")

    # Resize 48x48
    image = image.resize((48, 48))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = img_array.reshape(
        1,
        48,
        48,
        1
    )

    if st.button("Dự đoán cảm xúc"):

        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)

        emotion = classes[class_index]

        confidence = np.max(prediction) * 100

        st.success(
            f"Cảm xúc: {emotion}"
        )

        st.write(
            f"Độ tin cậy: {confidence:.2f}%"
        )
import os
import streamlit as st
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False
import numpy as np
from PIL import Image

MODEL_PATH = "saved_model/custom_model.h5"
CLASS_NAMES = ["nowildfire", "wildfire"]

@st.cache_resource
def load_model(path):
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB").resize(target_size)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image, model):
    tensor = preprocess_image(image)
    probs = model.predict(tensor)
    return probs[0]


def main():
    st.set_page_config(page_title="Wildfire Danger Forecast", page_icon="🔥")
    st.title("Forecasting Wildfire Danger with Deep Learning")
    st.write(
        "Upload an image to predict wildfire risk using the trained model. "
        "If the model file is not available, run `python main.py` first."
    )

    model = load_model(MODEL_PATH)
    if model is None:
        if not TF_AVAILABLE:
            st.warning(
                "TensorFlow is not available in this deployment environment, so predictions are disabled. "
                "To run predictions, run this app locally with TensorFlow installed or host the model and enable downloading."
            )
        else:
            st.error(
                f"Model file not found at `{MODEL_PATH}`. Run `python main.py` to train the model and save it."
            )
        return

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict wildfire risk"):
            with st.spinner("Predicting..."):
                probs = predict(image, model)
                no_fire, fire = float(probs[0]), float(probs[1])
                st.markdown(f"**No wildfire probability:** {no_fire * 100:.2f}%")
                st.markdown(f"**Wildfire probability:** {fire * 100:.2f}%")
                label = CLASS_NAMES[int(np.argmax(probs))]
                st.success(f"Predicted: {label}")
    else:
        st.info("Upload an image to classify wildfire risk.")

        if st.checkbox("Show sample test images"):
            test_dir = "Dataset/test"
            if os.path.exists(test_dir):
                for cls in ["nowildfire", "wildfire"]:
                    class_dir = os.path.join(test_dir, cls)
                    if os.path.isdir(class_dir):
                        sample_files = [
                            f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]
                        if sample_files:
                            st.write(f"Sample {cls} image")
                            st.image(os.path.join(class_dir, sample_files[0]), use_column_width=True)
            else:
                st.warning("No sample dataset images available in Dataset/test.")


if __name__ == "__main__":
    main()

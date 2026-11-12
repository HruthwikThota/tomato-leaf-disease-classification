
import streamlit as st
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

# Page configuration
st.set_page_config(
    page_title="Tomato Disease Classification",
    page_icon="🍅",
    layout="centered"
)

# Load trained model
MODEL_PATH = "tomato_svm_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()

scaler = bundle["scaler"]
model = bundle["model"]
le = bundle["label_encoder"]

# Disease information
disease_info = {
    "Tomato___Bacterial_spot":
        "Bacterial Spot: A bacterial disease causing dark, water-soaked spots on tomato leaves.",

    "Tomato___Early_blight":
        "Early Blight: A fungal disease producing brown spots with concentric rings on leaves.",

    "Tomato___Late_blight":
        "Late Blight: A serious disease causing dark, irregular lesions on leaves.",

    "Tomato___Leaf_Mold":
        "Leaf Mold: A fungal disease causing yellow patches and mold growth, often on the underside of leaves.",

    "Tomato___Septoria_leaf_spot":
        "Septoria Leaf Spot: A fungal disease producing small circular spots with dark borders.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Spider Mites: Tiny pests causing yellow speckles and damage to leaf tissue.",

    "Tomato___Target_Spot":
        "Target Spot: A fungal disease causing circular brown lesions with target-like patterns.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Yellow Leaf Curl Virus: A viral disease causing yellowing, curling and stunted leaf growth.",

    "Tomato___Tomato_mosaic_virus":
        "Tomato Mosaic Virus: A viral disease causing mosaic-like light and dark green patterns.",

    "Tomato___healthy":
        "Healthy: No disease class was identified in this image."
}

# Application interface
st.title("🍅 Tomato Leaf Disease Classification")

st.write(
    "Upload a tomato leaf image to predict its disease "
    "using a Traditional Machine Learning model."
)

st.info("Model: Linear Support Vector Machine (SVM)")

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Tomato Leaf",
        width=350
    )

    if st.button("Predict Disease", type="primary"):

        # Image preprocessing
        processed_image = image.convert("L")
        processed_image = processed_image.resize((64, 64))
        image_array = np.array(processed_image)

        # HOG feature extraction
        features = hog(
            image_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        # Feature scaling
        features_scaled = scaler.transform([features])

        # Prediction
        prediction = model.predict(features_scaled)
        predicted_class = le.inverse_transform(prediction)[0]

        # Display result
        disease_name = predicted_class.replace("Tomato___", "")
        disease_name = disease_name.replace("_", " ")

        st.success("Prediction Completed!")

        st.subheader("Predicted Disease")
        st.markdown(f"### {disease_name}")

        st.subheader("Disease Information")
        st.write(disease_info[predicted_class])

        st.metric(
            "Model Validation Accuracy",
            "56.84%"
        )

        st.warning(
            "This prediction is based on a Traditional ML model. "
            "It is not a confirmed agricultural diagnosis."
        )

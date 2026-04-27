import gdown
gdown.download(
    "https://drive.google.com/uc?id=1_87fGLSURkfO1Vy0tY8MzF6FokOMhE5l",
    "vgg19.h5",
    quiet=False
)
gdown.download(
    "https://drive.google.com/uc?id=1PpVEJbV3eCpUFVpiec_mFPd_kZ-2qkdM",
    "inception.h5",
    quiet=False
)
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.keras.backend.clear_session()

# =========================
# LOAD MODELS
# =========================
model_vgg = tf.keras.models.load_model("vgg19.h5",compile=False)
model_inc = tf.keras.models.load_model("inception.h5",compile=False)

# =========================
# GRAD-CAM FUNCTION
# =========================
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def apply_gradcam(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    return superimposed_img


# =========================
# UI
# =========================
st.title("🩺 Pneumonia Detection System")

model_choice = st.selectbox("Select Model", ["VGG19", "InceptionV3"])

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

# =========================
# PROCESS IMAGE
# =========================
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Choose model
    if model_choice == "VGG19":
        model = model_vgg
        size = (224, 224)
        last_layer = "block5_conv4"
    else:
        model = model_inc
        size = (299, 299)
        last_layer = "mixed10"

    # Preprocess
    img_resized = cv2.resize(img, size)
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"

    confidence = prediction if prediction > 0.5 else (1 - prediction)

    st.subheader(f"Result: {result}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # =========================
    # GRAD-CAM BUTTON
    # =========================
    if st.button("Show Grad-CAM"):
        heatmap = get_gradcam_heatmap(model, img_array, last_layer)
        cam_img = apply_gradcam(img_resized, heatmap)

        st.image(cam_img, caption="Grad-CAM Heatmap", use_column_width=True)

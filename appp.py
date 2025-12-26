# Run using: streamlit run app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- Page Config ----------------
st.set_page_config(page_title="Pixel Matrix", layout="wide")
st.title("🔍 Image Feature Matching (ORB)")

# ---------------- Sidebar Upload ----------------
st.sidebar.title("Upload Images")

img1_file = st.sidebar.file_uploader("First Image", type=["jpg", "png", "jpeg"])
img2_file = st.sidebar.file_uploader("Second Image", type=["jpg", "png", "jpeg"])

def load_image(file):
    return np.array(Image.open(file))

# ---------------- Sidebar Metrics ----------------
st.sidebar.header("Matching Metrics")

n_features = st.sidebar.slider("ORB Features", 100, 2000, 500)
blur_kernel = st.sidebar.slider("Gaussian Blur Kernel", 1, 31, 1, step=2)
max_distance = st.sidebar.slider("Max Match Distance", 10, 100, 40)

use_edges = st.sidebar.checkbox("Use Edge Detection")
if use_edges:
    t1 = st.sidebar.slider("Canny Threshold 1", 50, 300, 100)
    t2 = st.sidebar.slider("Canny Threshold 2", 50, 300, 200)

# ---------------- Matching Function ----------------
def match_images(img1, img2):
    orb = cv2.ORB_create(nfeatures=n_features)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)

    # Filter matches by distance
    good_matches = [m for m in matches if m.distance <= max_distance]

    match_percent = len(good_matches) / min(len(kp1), len(kp2)) * 100

    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched_img, match_percent

# ---------------- Main Logic ----------------
if img1_file and img2_file:
    img1 = load_image(img1_file)
    img2 = load_image(img2_file)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Optional preprocessing
    if blur_kernel > 1:
        gray1 = cv2.GaussianBlur(gray1, (blur_kernel, blur_kernel), 0)
        gray2 = cv2.GaussianBlur(gray2, (blur_kernel, blur_kernel), 0)

    if use_edges:
        gray1 = cv2.Canny(gray1, t1, t2)
        gray2 = cv2.Canny(gray2, t1, t2)

    col1, col2 = st.columns(2)
    col1.image(img1, caption="First Image", use_container_width=True)
    col2.image(img2, caption="Second Image", use_container_width=True)

    matched_img, match_percent = match_images(gray1, gray2)

    if matched_img is not None:
        st.image(
            matched_img,
            caption=f"Matched Features — {match_percent:.2f}%",
            use_container_width=True
        )
    else:
        st.warning("Not enough features detected.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Adjust ORB features, blur, edge detection, and match distance to see how match percentage changes."
)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import os
import requests
from io import BytesIO

# -------------------- CLIP MODEL SETUP --------------------
st.set_page_config(page_title="MotoMind AI", layout="centered")
st.title("MotoMind")
st.write("Find the best matching car by uploading an image or searching by name/company.")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Try loading CLIP safely
try:
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=MODEL_DIR)
except Exception as e:
    st.error("⚠️ Failed to load CLIP model — possibly due to download restrictions on Streamlit Cloud.")
    st.info("Please preload the model locally or include it in the repository under './models'.")
    st.stop()

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("CarsDataset.csv")

df = load_data()

cols_to_display = [c for c in df.columns.tolist() if c != 'Price']

# -------------------- BUILD / LOAD IMAGE EMBEDDINGS --------------------
@st.cache_data(show_spinner=False)
def build_or_load_embeddings(csv_length: int, expected_plus_one: bool = True) -> np.ndarray:
    path = "car_image_embeddings.npy"

    if os.path.exists(path):
        try:
            emb = np.load(path)
            if expected_plus_one and emb.shape[0] == csv_length + 1:
                return emb
            if not expected_plus_one and emb.shape[0] == csv_length:
                return emb
        except Exception:
            pass

    st.warning("⚠️ Building new image embeddings... This may take a while on first run.")
    image_urls = df['Car Image'].fillna('').astype(str).tolist()
    features: list[np.ndarray] = []

    for url in image_urls:
        try:
            if url.startswith("http://") or url.startswith("https://"):
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            elif os.path.exists(url):
                image = Image.open(url).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), color="white")

            image_tensor: torch.Tensor = preprocess(image)
            image_input = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                vec = model.encode_image(image_input).cpu().numpy()[0]
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        except Exception:
            vec = np.zeros((512,), dtype=np.float32)
        features.append(vec.astype(np.float32))

    emb = np.vstack(features)
    if expected_plus_one:
        pad = np.zeros((1, emb.shape[1]), dtype=np.float32)
        emb = np.vstack([emb, pad])
    np.save(path, emb)
    return emb


dataset_embeddings = build_or_load_embeddings(len(df), expected_plus_one=True)

# -------------------- CUSTOM STYLING --------------------
st.markdown("""
<style>
.custom-vertical-table {
    width: 100%;
    border-collapse: collapse;
}
.custom-vertical-table th, .custom-vertical-table td {
    border: 1px solid #444;
    padding: 10px;
    text-align: left;
}
.custom-vertical-table th {
    background-color: #111;
    color: white;
    width: 25%;
}
</style>
""", unsafe_allow_html=True)

# -------------------- IMAGE-BASED SEARCH --------------------
st.markdown("## 🔍 Image-Based Search")
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png", "webp"])
car_name_input = st.text_input("Optional: Enter Car Name")
car_company_input = st.text_input("Optional: Enter Car Company")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = preprocess(image)
    image_input = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        uploaded_features = model.encode_image(image_input)
        uploaded_features /= uploaded_features.norm(dim=-1, keepdim=True)

    filtered_df = df.copy()
    if car_name_input:
        filtered_df = filtered_df[filtered_df['Car Name'].str.lower() == car_name_input.lower()]
    if car_company_input:
        filtered_df = filtered_df[filtered_df['Car Company'].str.lower() == car_company_input.lower()]

    if filtered_df.empty:
        st.warning("No matching cars found based on filters.")
    else:
        filtered_df = filtered_df.reset_index(drop=False)
        valid_indices = filtered_df['index'].tolist()
        filtered_embeddings = dataset_embeddings[valid_indices]
        similarities = filtered_embeddings @ uploaded_features.cpu().numpy().T
        best_idx_local = int(np.argmax(similarities))
        best_car = df.iloc[valid_indices[best_idx_local]]

        st.success("✅ Best match found!")
        st.image(best_car['Car Image'], caption=best_car['Car Name'], use_container_width=True)

        specs_df = best_car[cols_to_display].to_frame(name="Value").reset_index()
        specs_df.columns = ["<strong>Specification</strong>", "<strong>Value</strong>"]
        st.markdown(specs_df.to_html(classes="custom-vertical-table", index=False, escape=False), unsafe_allow_html=True)

# -------------------- TEXT-BASED SEARCH --------------------
st.markdown("## 🔤 Text-Based Search")
name_input = st.text_input("Enter exact car name (for text search):")

if name_input:
    match = df[df['Car Name'].str.lower() == name_input.lower()]
    if not match.empty:
        car = match.iloc[0]
        st.image(car['Car Image'], caption=car['Car Name'], use_container_width=True)
        specs_df = car[cols_to_display].to_frame(name="Value").reset_index()
        specs_df.columns = ["<strong>Specification</strong>", "<strong>Value</strong>"]
        st.markdown(specs_df.to_html(classes="custom-vertical-table", index=False, escape=False), unsafe_allow_html=True)
    else:
        st.warning("No car found with that name.")

# -------------------- FEATURE-BASED RECOMMENDATION --------------------
st.markdown("## 🚗 Car Recommendations by Preferences")

def extract_mid_price(price_range: str | None) -> float | None:
    try:
        if not price_range:
            return None
        parts = price_range.replace(" ", "").split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return float(parts[0])
    except Exception:
        return None

def pick_price_source(row: pd.Series) -> str:
    price_value = row.get("Price", None)
    if not price_value and "Price in Lakhs" in row:
        price_value = row["Price in Lakhs"]
    return str(price_value) if price_value else ""

def safe_extract_mid_price(row: pd.Series) -> float:
    val = extract_mid_price(pick_price_source(row))
    return float(val) if val is not None else np.nan

df["Price_Mid"] = df.apply(safe_extract_mid_price, axis=1)

min_price = st.number_input("Enter Minimum Price (in Lakhs)", min_value=0.0, step=0.1)
max_price = st.number_input("Enter Maximum Price (in Lakhs)", min_value=0.0, step=0.1)
selected_company = st.selectbox("Select Car Company", ["Any"] + sorted(df["Car Company"].unique()))
selected_body_type = st.selectbox("Select Body Type", ["Any"] + sorted(df["Body Type"].unique()))

if st.button("Show Matching Cars"):
    if min_price >= max_price:
        st.error("Minimum price must be less than maximum price.")
    else:
        filtered_df = df[(df["Price_Mid"] >= min_price) & (df["Price_Mid"] <= max_price)]
        if selected_company != "Any":
            filtered_df = filtered_df[filtered_df["Car Company"] == selected_company]
        if selected_body_type != "Any":
            filtered_df = filtered_df[filtered_df["Body Type"] == selected_body_type]

        if filtered_df.empty:
            st.warning("No cars found matching your criteria.")
        else:
            top_3 = filtered_df.sample(n=min(3, len(filtered_df)))
            st.success(f"Found {len(top_3)} matching car(s):")
            for _, row in top_3.iterrows():
                st.image(row['Car Image'], caption=row['Car Name'], use_container_width=True)
                specs_df = row[cols_to_display].to_frame(name="Value").reset_index()
                specs_df.columns = ["<strong>Specification</strong>", "<strong>Value</strong>"]
                st.markdown(specs_df.to_html(classes="custom-vertical-table", index=False, escape=False), unsafe_allow_html=True)
                st.markdown("---")

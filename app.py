import streamlit as st
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import os
import requests
from io import BytesIO
import subprocess

# -------------------- CLIP MODEL SETUP --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Safe CLIP loading for Streamlit Cloud ---
MODEL_DIR = os.path.expanduser("~/.cache/clip")
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=MODEL_DIR)
except Exception as e:
    st.warning("⚠️ Retrying CLIP model load via Git clone (first-time setup)...")
    subprocess.run(["git", "clone", "https://github.com/openai/CLIP.git"])
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=MODEL_DIR)
# -----------------------------------------------------------

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("CarsDataset.csv")

df = load_data()

# Exclude raw 'Price' from display if 'Price in Lakhs' exists
cols_to_display = [c for c in df.columns.tolist() if c != 'Price']

# -------------------- BUILD / LOAD IMAGE EMBEDDINGS --------------------
@st.cache_data(show_spinner=False)
def build_or_load_embeddings(csv_length: int, expected_plus_one: bool = True) -> np.ndarray:
    path = "car_image_embeddings.npy"

    # Try loading saved embeddings
    if os.path.exists(path):
        try:
            emb = np.load(path)
            if expected_plus_one:
                if emb.shape[0] == csv_length + 1:
                    return emb
                if emb.shape[0] == csv_length:
                    pad = np.zeros((1, emb.shape[1]), dtype=emb.dtype)
                    emb = np.vstack([emb, pad])
                    np.save(path, emb)
                    return emb
            else:
                if emb.shape[0] == csv_length:
                    return emb
        except Exception:
            pass

    # Rebuild embeddings if missing or mismatched
    image_urls = df['Car Image'].fillna('').astype(str).tolist()
    features: list[np.ndarray] = []

    for url in image_urls:
        try:
            image = None
            if url and (url.startswith("http://") or url.startswith("https://")):
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            elif url and os.path.exists(url):
                image = Image.open(url).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), color="white")

            image_tensor: torch.Tensor = preprocess(image)  # type: ignore
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

if dataset_embeddings.shape[0] not in (len(df), len(df) + 1):
    st.warning(f"⚠ CSV has {len(df)} rows, embeddings have {dataset_embeddings.shape[0]} entries. Rebuild attempted but counts still differ.")

# -------------------- PAGE LAYOUT --------------------
st.set_page_config(page_title="MotoMind AI", layout="centered")
st.title("MotoMind")
st.write("Find the best matching car by uploading an image or searching by name/company.")

# --- Custom CSS ---
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
st.markdown("## Image-Based Search")
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png", "AVIF", "webp"])
car_name_input = st.text_input("Optional: Enter Car Name (for better matching)")
car_company_input = st.text_input("Optional: Enter Car Company (for better matching)")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor: torch.Tensor = preprocess(image)  # type: ignore
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
        filtered_indices = filtered_df['index'].tolist()

        valid_indices = [i for i in filtered_indices if i < len(dataset_embeddings)]

        if not valid_indices:
            st.error("No valid embeddings found for the selected filters.")
        else:
            filtered_embeddings = dataset_embeddings[valid_indices]
            similarities = filtered_embeddings @ uploaded_features.cpu().numpy().T
            best_idx_local = int(np.argmax(similarities))
            best_idx_global = valid_indices[best_idx_local]

            best_car = df.iloc[best_idx_global]
            st.success("Best match found!")
            st.image(best_car['Car Image'], caption=best_car['Car Name'], use_container_width=True)

            st.write("### Car Specifications")
            car_specs_df = best_car[cols_to_display].to_frame(name="Value")
            car_specs_df.reset_index(inplace=True)
            car_specs_df.columns = ["<strong>Specification</strong>", "<strong>Value</strong>"]
            st.markdown(car_specs_df.to_html(classes="custom-vertical-table", index=False, escape=False), unsafe_allow_html=True)

# -------------------- TEXT-BASED SEARCH --------------------
st.markdown("## Text-Based Search")
name_input = st.text_input("Enter exact car name (text search only):")

if name_input:
    match = df[df['Car Name'].str.lower() == name_input.lower()]
    if not match.empty:
        car = match.iloc[0]
        st.image(car['Car Image'], caption=car['Car Name'], use_container_width=True)
        st.write("### Car Specifications")
        car_specs_df = car[cols_to_display].to_frame(name="Value")
        car_specs_df.reset_index(inplace=True)
        car_specs_df.columns = ["<strong>Specification</strong>", "<strong>Value</strong>"]
        st.markdown(car_specs_df.to_html(classes="custom-vertical-table", index=False, escape=False), unsafe_allow_html=True)
    else:
        st.warning("No car found with that name.")

# -------------------- FEATURE-BASED RECOMMENDATION --------------------
st.markdown("## Car Based on Your Preferences")

def extract_mid_price(price_range: str | None) -> float | None:
    try:
        if not price_range:
            return None
        parts = price_range.replace(" ", "").split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return float(parts[0])
    except Exception:
        return None


def pick_price_source(row: pd.Series) -> str:
    price_value = row.get("Price", None)
    if isinstance(price_value, float) and np.isnan(price_value):
        price_value = None
    if not price_value and "Price in Lakhs" in row and isinstance(row["Price in Lakhs"], str) and row["Price in Lakhs"].strip():
        price_value = row["Price in Lakhs"]
    return price_value if isinstance(price_value, str) else (str(price_value) if price_value is not None else "")


def safe_extract_mid_price(row: pd.Series) -> float:
    val = extract_mid_price(pick_price_source(row))
    return float(val) if val is not None else np.nan

df["Price_Mid"] = df.apply(safe_extract_mid_price, axis=1)

# -------------------- SIDEBAR INPUTS --------------------
min_price = st.number_input("Enter Minimum Price (in Lakhs)", min_value=0.0, step=0.1)
max_price = st.number_input("Enter Maximum Price (in Lakhs)", min_value=0.0, step=0.1)
selected_company = st.selectbox("Select Car Company", ["Any"] + sorted(df["Car Company"].unique()))
selected_body_type = st.selectbox("Select Body Type", ["Any"] + sorted(df["Body Type"].unique()))

# -------------------- FILTERING LOGIC --------------------
if st.button("Show Matching Cars"):
    if min_price == 0.0 and max_price == 0.0:
        st.error("Please enter a valid Price Range.")
    elif min_price >= max_price:
        st.error("Minimum price must be less than Maximum price.")
    else:
        filtered_df = df[(df["Price_Mid"] >= min_price) & (df["Price_Mid"] <= max_price)]

        if selected_company != "Any":
            filtered_df = filtered_df[filtered_df["Car Company"] == selected_company]

        if selected_body_type != "Any":
            filtered_df = filtered_df[filtered_df["Body Type"] == selected_body_type]

        if filtered_df.empty:
            st.warning("No cars found matching the selected criteria.")
        else:
            top_3 = filtered_df.sample(n=min(3, len(filtered_df)))
            st.success(f"Found {len(top_3)} matching car(s):")

            base_cols = [
                "Car Company", "ARAI Mileage", "Fuel Type", "Engine Displacement",
                "Max Power", "Max Torque", "Seating Capacity", "Transmission Type",
                "Fuel Tank Capacity", "Body Type"
            ]
            if 'Price in Lakhs' in df.columns:
                base_cols.append('Price in Lakhs')
            elif 'Price' in df.columns:
                base_cols.append('Price')

            for _, row in top_3.iterrows():
                st.image(row['Car Image'], caption=row['Car Name'], use_container_width=True)
                specs_df = row[base_cols].to_frame(name="Value")
                specs_df.reset_index(inplace=True)
                specs_df.columns = ["<strong>Specification</strong>", "<strong>Value</strong>"]
                st.markdown(specs_df.to_html(classes="custom-vertical-table", index=False, escape=False), unsafe_allow_html=True)
                st.markdown("---")

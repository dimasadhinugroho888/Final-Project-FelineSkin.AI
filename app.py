import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import requests

# =========================
# 🔑 CONFIG
# =========================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

disease_map = {
    "Flea_Allergy": "Alergi kutu pada kucing",
    "Health": "Kucing sehat",
    "Ringworm": "Kurap pada kucing",
    "Scabies": "Kudis pada kucing"
}

# =========================
# 🤖 AI FUNCTION
# =========================
def get_ai_explanation(disease_name):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    if disease_name == "Kucing sehat":
        prompt = """
        Kucing dalam kondisi sehat.

        Berikan:
        - Tips perawatan harian
        - Cara menjaga kesehatan kulit
        - Pencegahan penyakit kulit
        - Kapan perlu ke dokter
        """
    else:
        prompt = f"""
        Jelaskan penyakit {disease_name} pada kucing dengan bahasa sederhana.

        Format:
        - Penjelasan
        - Penyebab
        - Gejala
        - Penanganan awal
        - Kapan ke dokter
        """

    models = [
        "tencent/hy3-preview:free",
        "mistralai/mistral-7b-instruct:free"
    ]

    for model in models:
        try:
            res = requests.post(url, headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }, timeout=10)

            result = res.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

        except:
            continue

    return "❌ AI gagal merespon."

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)

@st.cache_resource
def load_cat_detector():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    return model

@st.cache_data
def load_class_names():
    try:
        with open('class_names.txt') as f:
            return [x.strip() for x in f.readlines()]
    except:
        return ["Flea_Allergy", "Health", "Ringworm", "Scabies"]

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

# =========================
# DETEKSI KUCING
# =========================
def is_cat_image(img):
    model = load_cat_detector()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)

    # ImageNet indices for cat breeds
    cat_indices = [281, 282, 283, 284, 285]
    
    cat_prob = sum([probs[i].item() for i in cat_indices])
    
    top_prob, top_idx = torch.max(probs, 0)
    top_idx = top_idx.item()

    # Decision logic - Much stricter
    if cat_prob > 0.45: # High confidence
        return True, "Kucing terdeteksi"
    
    if top_idx in cat_indices and top_prob > 0.30: # Cat is the top prediction
        return True, "Kucing terdeteksi"
            
    return False, "Gambar tidak dikenali sebagai kucing. Harap gunakan foto kucing yang jelas."

# =========================
# CLOSE-UP DETECTION (🔥 BARU)
# =========================
def is_closeup_texture(img):
    img_np = np.array(img.resize((224,224)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Laplacian variance (sharpness/texture detail)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Canny for edge density (sensitive to fur/hair)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (224*224)

    # Organic textures like fur/skin usually have:
    # 1. Medium-High Laplacian variance (200 - 1500)
    # 2. Consistent edge density (0.08 - 0.25)
    # Graphics/Icons usually have very few edges or extreme contrast (> 2000 var)
    
    if 180 < lap_var < 2000 and 0.07 < edge_density < 0.3:
        return True
    return False

# =========================
# GRADCAM
# =========================
def gradcam(model, img_tensor, target):
    grads, acts = [], []

    def f_hook(m,i,o): acts.append(o)
    def b_hook(m,gi,go): grads.append(go[0])

    h1 = model.layer4.register_forward_hook(f_hook)
    h2 = model.layer4.register_backward_hook(b_hook)

    out = model(img_tensor)
    loss = out[0, target]

    model.zero_grad()
    loss.backward()

    h1.remove()
    h2.remove()

    pooled = torch.mean(grads[0], dim=[0,2,3])
    act = acts[0][0]

    for i in range(act.shape[0]):
        act[i] *= pooled[i]

    heat = torch.mean(act, dim=0).detach().numpy()
    heat = np.maximum(heat, 0)
    heat /= np.max(heat) if np.max(heat) else 1
    heat = cv2.resize(heat, (224,224))
    heat = np.uint8(255*heat)

    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)

# =========================
# MAIN
# =========================
def main():
    st.title("🐱 FelineSkin.AI")
    st.caption("Smart AI for Cat Skin Health 🐾")

    classes = load_class_names()
    model = load_model()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    try:
        model.load_state_dict(torch.load('cat_skin_disease_model.pth', map_location='cpu'))
    except:
        st.error("Model tidak ditemukan!")
        return

    model.eval()

    file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert('RGB')
        st.image(img)

        # ===== DETEKSI =====
        with st.spinner("Analisis gambar..."):
            is_cat, cat_info = is_cat_image(img)
            is_closeup = is_closeup_texture(img)

        if not is_cat:
            if is_closeup:
                st.warning("⚠️ Mode Close-up/Zoom terdeteksi, tetap diproses...")
            else:
                st.error(f"❌ {cat_info}")
                st.stop()

        # ===== PREDIKSI =====
        with st.spinner("Mendeteksi penyakit..."):
            tensor = preprocess(img)

            with torch.no_grad():
                out = model(tensor)
                probs = torch.nn.functional.softmax(out[0], dim=0)

            conf, idx = torch.max(probs, 0)
            label = classes[idx.item()]
            conf_pct = conf.item()*100

        indo_label = disease_map.get(label, label)

        st.success(f"🎯 {indo_label} ({conf_pct:.1f}%)")

        if conf_pct < 50:
            st.error("⚠️ Model sangat tidak yakin dengan hasil ini. Harap pastikan gambar adalah penyakit kulit kucing.")
            st.stop()
        elif conf_pct < 70:
            st.warning("⚠️ Keyakinan rendah, hasil mungkin tidak akurat. Harap pastikan pencahayaan cukup dan fokus pada area penyakit.")

        # ===== PROB =====
        st.write("## 📊 Probabilitas")
        for i, c in enumerate(classes):
            st.progress(probs[i].item(), text=f"{c}: {probs[i].item()*100:.1f}%")

        # ===== CAM =====
        st.write("## 🔍 Area Deteksi")
        heat = gradcam(model, tensor, idx.item())

        img_np = np.array(img.resize((224,224)))
        overlay = cv2.addWeighted(img_np, 0.6, heat, 0.4, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        st.image(overlay)
        st.caption("Merah = area paling berpengaruh")

        # ===== AI =====
        if label == "Health":
            st.write("## 🐾 Tips Perawatan Kucing Sehat")
        else:
            st.write("## 🧠 Analisis & Saran AI")

        with st.spinner("Mengambil penjelasan..."):
            ai = get_ai_explanation(indo_label)

        st.write(ai)
        st.warning("⚠️ Ini bukan diagnosis medis.")

        # ===== MAP =====
        st.write("## 🗺️ Cari Dokter")

        kategori = st.selectbox(
            "Pilih layanan:",
            ["dokter hewan", "klinik hewan", "puskeswan"]
        )

        lokasi = st.text_input("Masukkan lokasi")

        query = f"{kategori} terdekat di {lokasi}" if lokasi else f"{kategori} terdekat"
        map_url = f"https://www.google.com/maps?q={query.replace(' ', '+')}&output=embed"

        st.components.v1.iframe(map_url, height=500)

        st.link_button(
            "🔍 Buka di Google Maps",
            f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
        )

if __name__ == "__main__":
    main()

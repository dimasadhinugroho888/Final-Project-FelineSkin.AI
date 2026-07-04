import os
from urllib.parse import quote_plus

import cv2
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms


APP_NAME = "FelineSkin.AI"
MODEL_PATH = "cat_skin_disease_model.pth"
CLASS_NAMES_PATH = "class_names.txt"
LOGO_PATH = "logo.png"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_UPLOAD_SIZE_MB = 8

DISEASE_MAP = {
    "Flea_Allergy": "Alergi kutu pada kucing",
    "Health": "Kucing sehat",
    "Ringworm": "Kurap pada kucing",
    "Scabies": "Kudis pada kucing",
}

OPENROUTER_MODELS = [
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
]

CAT_IMAGENET_INDICES = {
    281,  # tabby
    282,  # tiger cat
    283,  # Persian cat
    284,  # Siamese cat
    285,  # Egyptian cat
}

CAT_TOPIC_KEYWORDS = {
    "kucing",
    "cat",
    "kitten",
    "anak kucing",
    "kulit",
    "bulu",
    "gatal",
    "garuk",
    "luka",
    "koreng",
    "jamur",
    "kurap",
    "ringworm",
    "scabies",
    "kudis",
    "kutu",
    "flea",
    "alergi",
    "dokter hewan",
    "veteriner",
    "vaksin",
    "makan",
    "pakan",
    "mandi",
    "obat",
    "salep",
    "pasir",
    "steril",
}


def get_openrouter_api_key():
    try:
        return st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    except Exception:
        return os.environ.get("OPENROUTER_API_KEY")


def call_openrouter(messages, temperature=0.35, max_tokens=700):
    api_key = get_openrouter_api_key()
    if not api_key:
        return (
            "API key OpenRouter belum dikonfigurasi. Tambahkan `OPENROUTER_API_KEY` "
            "di Streamlit Secrets atau environment variable."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://felineskin-ai.streamlit.app",
        "X-Title": APP_NAME,
    }

    last_error = None
    for model_name in OPENROUTER_MODELS:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=35,
            )
            try:
                result = response.json()
            except ValueError:
                result = {}

            if response.status_code >= 400:
                error_data = result.get("error", {}) if isinstance(result, dict) else {}
                error_message = error_data.get("message") or response.text[:300]
                last_error = f"HTTP {response.status_code} dari OpenRouter: {error_message}"
                continue

            choices = result.get("choices", [])

            if choices:
                content = choices[0].get("message", {}).get("content")
                if content:
                    return content.strip()

            last_error = "Respons OpenRouter tidak berisi jawaban."
        except requests.exceptions.Timeout:
            last_error = "Request ke OpenRouter timeout."
        except requests.exceptions.ConnectionError:
            last_error = "Tidak bisa terhubung ke OpenRouter."
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)

    return (
        "AI gagal merespon dari OpenRouter. "
        f"Detail terakhir: {last_error}. "
        "Pastikan Streamlit Secrets berisi OPENROUTER_API_KEY yang valid dan akun OpenRouter memiliki akses/kredit."
    )


def get_ai_explanation(disease_name, confidence):
    if disease_name == "Kucing sehat":
        user_prompt = """
Kucing terdeteksi sehat oleh model gambar.

Berikan dalam bahasa Indonesia:
- Ringkasan singkat
- Tips perawatan harian
- Cara menjaga kesehatan kulit dan bulu
- Pencegahan penyakit kulit
- Kapan tetap perlu ke dokter hewan
"""
    else:
        user_prompt = f"""
Model gambar memprediksi kondisi: {disease_name}
Confidence model: {confidence:.1f}%

Jelaskan dengan bahasa Indonesia yang sederhana.

Format:
- Penjelasan
- Penyebab umum
- Gejala yang perlu diamati
- Penanganan awal yang aman
- Tanda bahaya
- Kapan harus ke dokter hewan

Jangan menyatakan diagnosis pasti. Jangan memberikan dosis obat. Tekankan bahwa dokter hewan adalah rujukan utama.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah asisten edukasi kesehatan kucing. Jawab aman, ringkas, "
                "berbasis kehati-hatian, dan tidak menggantikan dokter hewan."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return call_openrouter(messages)


def is_cat_related_question(question):
    normalized = question.lower()
    return any(keyword in normalized for keyword in CAT_TOPIC_KEYWORDS)


def answer_cat_chat(question, diagnosis_context):
    if not is_cat_related_question(question):
        return (
            "Saya hanya bisa membantu pertanyaan seputar kucing, terutama kesehatan, "
            "kulit, bulu, perawatan, dan langkah aman setelah hasil analisis."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah chatbot FelineSkin.AI. Jawab hanya topik seputar kucing. "
                "Jika user bertanya di luar kucing, tolak dengan sopan dan arahkan kembali ke kucing. "
                "Jangan memberi diagnosis pasti, jangan memberi dosis obat, dan sarankan dokter hewan "
                "untuk kondisi berat, memburuk, luka terbuka, bernanah, menular, atau kucing terlihat lemas."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Konteks hasil analisis aplikasi:\n{diagnosis_context}\n\n"
                f"Pertanyaan user:\n{question}"
            ),
        },
    ]
    return call_openrouter(messages, temperature=0.25, max_tokens=550)


@st.cache_data
def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
            classes = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        classes = ["Flea_Allergy", "Health", "Ringworm", "Scabies"]

    return classes


@st.cache_resource
def load_skin_model(classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    except FileNotFoundError as exc:
        raise RuntimeError(f"File model `{MODEL_PATH}` tidak ditemukan.") from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "File model ditemukan, tetapi strukturnya tidak cocok dengan jumlah kelas aplikasi."
        ) from exc

    model.eval()
    return model


@st.cache_resource
def load_cat_detector():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


@st.cache_data
def get_preprocess_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def preprocess(img):
    return get_preprocess_transform()(img).unsqueeze(0)


def read_uploaded_image(uploaded_file):
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(f"Ukuran gambar terlalu besar ({size_mb:.1f} MB). Maksimal {MAX_UPLOAD_SIZE_MB} MB.")

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("File yang diupload bukan gambar valid.") from exc


def is_cat_image(img):
    model = load_cat_detector()
    tensor = preprocess(img)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)

    cat_prob = sum(probs[index].item() for index in CAT_IMAGENET_INDICES)
    top_prob, top_idx = torch.max(probs, 0)
    top_idx = top_idx.item()

    if cat_prob > 0.35:
        return True, f"Kucing terdeteksi ({cat_prob * 100:.1f}%)."

    if top_idx in CAT_IMAGENET_INDICES and top_prob.item() > 0.25:
        return True, f"Kucing terdeteksi ({top_prob.item() * 100:.1f}%)."

    return False, "Gambar tidak dikenali sebagai kucing. Gunakan foto kucing yang jelas atau close-up kulit/bulu."


def is_closeup_texture(img):
    img_np = np.array(img.resize((224, 224)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (224 * 224)

    return 150 < lap_var < 2200 and 0.05 < edge_density < 0.35


def gradcam(model, img_tensor, target):
    grads = []
    activations = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_inputs, grad_outputs):
        grads.append(grad_outputs[0])

    forward_handle = model.layer4.register_forward_hook(forward_hook)
    backward_handle = model.layer4.register_full_backward_hook(backward_hook)

    try:
        output = model(img_tensor)
        loss = output[0, target]
        model.zero_grad()
        loss.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if not grads or not activations:
        raise RuntimeError("Grad-CAM gagal membaca aktivasi model.")

    pooled_grads = torch.mean(grads[0], dim=[0, 2, 3])
    activation = activations[0][0]
    weighted_activation = activation * pooled_grads[:, None, None]

    heat = torch.mean(weighted_activation, dim=0).detach().cpu().numpy()
    heat = np.maximum(heat, 0)
    max_value = np.max(heat)
    if max_value > 0:
        heat /= max_value

    heat = cv2.resize(heat, (224, 224))
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def render_header():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🐱",
        layout="centered",
    )

    if os.path.exists(LOGO_PATH):
        col_logo, col_title = st.columns([1, 4], vertical_alignment="center")
        with col_logo:
            st.image(LOGO_PATH, width=96)
        with col_title:
            st.title(APP_NAME)
            st.caption("Smart AI for Cat Skin Health")
    else:
        st.title(APP_NAME)
        st.caption("Smart AI for Cat Skin Health")


def render_chatbot(diagnosis_context):
    st.write("## Chatbot Kucing")
    st.caption("Tanyakan hal seputar kucing, kulit, bulu, perawatan, atau hasil analisis ini.")

    if "cat_chat_messages" not in st.session_state:
        st.session_state.cat_chat_messages = []

    for message in st.session_state.cat_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Tanya seputar kucing...")
    if not question:
        return

    st.session_state.cat_chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Menjawab pertanyaan..."):
            answer = answer_cat_chat(question, diagnosis_context)
        st.write(answer)

    st.session_state.cat_chat_messages.append({"role": "assistant", "content": answer})


def main():
    render_header()

    classes = load_class_names()
    try:
        model = load_skin_model(tuple(classes))
    except RuntimeError as exc:
        st.error(str(exc))
        return

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=140)
        st.header("Panduan Foto")
        st.write("- Gunakan foto yang terang dan tidak blur.")
        st.write("- Fokuskan pada area kulit atau bulu yang bermasalah.")
        st.write("- Hindari gambar hasil edit/filter.")
        st.write("- Maksimal ukuran file 8 MB.")
        st.warning("Aplikasi ini bukan pengganti diagnosis dokter hewan.")

    uploaded_file = st.file_uploader("Upload gambar kucing", type=["jpg", "png", "jpeg"])
    if not uploaded_file:
        return

    try:
        img = read_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.image(img, caption="Gambar yang dianalisis", use_container_width=True)

    with st.spinner("Menganalisis kualitas dan objek gambar..."):
        try:
            is_cat, cat_info = is_cat_image(img)
        except Exception as exc:
            is_cat = False
            cat_info = f"Deteksi kucing gagal dijalankan: {exc}"
        is_closeup = is_closeup_texture(img)

    if not is_cat:
        if is_closeup:
            st.warning("Mode close-up kulit/bulu terdeteksi. Hasil tetap diproses, tetapi pastikan gambar berasal dari kucing.")
        else:
            st.error(cat_info)
            st.stop()
    else:
        st.info(cat_info)

    with st.spinner("Mendeteksi kondisi kulit..."):
        tensor = preprocess(img)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out[0], dim=0)

        conf, idx = torch.max(probs, 0)
        label = classes[idx.item()]
        conf_pct = conf.item() * 100

    indo_label = DISEASE_MAP.get(label, label)
    st.success(f"Hasil: {indo_label} ({conf_pct:.1f}%)")

    if conf_pct < 50:
        st.error("Model sangat tidak yakin. Upload ulang foto yang lebih jelas sebelum memakai hasil ini.")
        st.stop()
    if conf_pct < 70:
        st.warning("Keyakinan model rendah. Gunakan hasil sebagai indikasi awal, bukan keputusan medis.")

    st.write("## Probabilitas")
    for i, class_name in enumerate(classes):
        display_name = DISEASE_MAP.get(class_name, class_name)
        probability = probs[i].item()
        st.progress(probability, text=f"{display_name}: {probability * 100:.1f}%")

    st.write("## Area Deteksi")
    try:
        heat = gradcam(model, tensor, idx.item())
        img_np = np.array(img.resize((224, 224)))
        overlay = cv2.addWeighted(img_np, 0.6, heat, 0.4, 0)
        st.image(overlay, caption="Merah = area paling berpengaruh", use_container_width=False)
    except RuntimeError as exc:
        st.warning(f"Grad-CAM tidak dapat ditampilkan: {exc}")

    st.write("## Analisis dan Saran AI" if label != "Health" else "## Tips Perawatan Kucing Sehat")
    with st.spinner("Mengambil penjelasan AI..."):
        ai_explanation = get_ai_explanation(indo_label, conf_pct)
    st.write(ai_explanation)
    st.warning("Ini bukan diagnosis medis. Konsultasikan ke dokter hewan untuk pemeriksaan pasti.")

    diagnosis_context = (
        f"Hasil model: {indo_label}\n"
        f"Confidence: {conf_pct:.1f}%\n"
        f"Penjelasan AI sebelumnya:\n{ai_explanation}"
    )
    render_chatbot(diagnosis_context)

    st.write("## Cari Dokter")
    kategori = st.selectbox("Pilih layanan:", ["dokter hewan", "klinik hewan", "puskeswan"])
    lokasi = st.text_input("Masukkan lokasi")
    query = f"{kategori} terdekat di {lokasi}" if lokasi else f"{kategori} terdekat"
    encoded_query = quote_plus(query)

    st.components.v1.iframe(f"https://www.google.com/maps?q={encoded_query}&output=embed", height=500)
    st.link_button("Buka di Google Maps", f"https://www.google.com/maps/search/{encoded_query}")


if __name__ == "__main__":
    main()

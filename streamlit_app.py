import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

translations = {
    "person": "شخص", "chair": "كرسي", "cup": "كوباية",
    "cell phone": "موبايل", "book": "كتاب", "bottle": "إزازة",
    "laptop": "لابتوب", "car": "عربية", "dog": "كلب", "cat": "قطة"
}

st.set_page_config(page_title="بصيرة", page_icon="👁️")
st.title("👁️ بصيرة — المساعد البصري الذكي")
st.write("صور أي حاجة بكاميرا تليفونك، ارفع الصورة هنا، وهقولك هي إيه")

uploaded_file = st.file_uploader("ارفع صورة من معرض الصور", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

    with st.spinner("بصيرة بيشوف الصورة..."):
        results = model(image, verbose=False)[0]
        names = [model.names[int(c)] for c in results.boxes.cls]
        unique = list(dict.fromkeys(names))

        if not unique:
            st.warning("مش شايف حاجة واضحة في الصورة دي. جرب صورة تانية.")
        else:
            arabic = [translations.get(n, n) for n in unique]
            description = "أنا شايف: " + "، ".join(arabic)

            st.success(description)

            tts = gTTS(text=description, lang='ar', slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            st.audio(audio_bytes, format='audio/mp3')

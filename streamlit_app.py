import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io
import os
import time
from google import genai
from google.genai import types

# --- إعداد Gemini (مفتاحين احتياطيين) ---
GEMINI_KEYS = [
    st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY")),
    st.secrets.get("GEMINI_API_KEY_2", None),
]

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

translations = {
    "person": "شخص", "chair": "كرسي", "cup": "كوباية",
    "cell phone": "موبايل", "book": "كتاب", "bottle": "إزازة",
    "laptop": "لابتوب", "car": "عربية", "dog": "كلب", "cat": "قطة",
    "remote": "ريموت", "keyboard": "كيبورد", "mouse": "ماوس",
    "scissors": "مقص", "clock": "ساعة", "vase": "فازة",
    "banana": "موزة", "apple": "تفاحة", "orange": "برتقالة",
    "spoon": "معلقة", "fork": "شوكة", "knife": "سكينة",
    "bed": "سرير", "toilet": "حمام", "tv": "تليفزيون",
    "microwave": "ميكروويف", "oven": "فرن", "sink": "حوض",
}

def text_to_speech(text):
    """صوت باللهجة المصرية (tld=com.eg)."""
    try:
        tts = gTTS(text=text, lang='ar', tld='com.eg', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception:
        return None

def get_position(box, img_width, img_height):
    """تحديد اتجاه الشيء (يمين/شمال/فوق/تحت/قصاد)."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    nx = cx / img_width
    ny = cy / img_height

    # الاتجاه الأفقي
    if nx < 0.33:
        h = "على شمالك"
    elif nx > 0.67:
        h = "على يمينك"
    else:
        h = "قصادك"

    # الاتجاه الرأسي (بس لو مش في النص)
    if ny < 0.30:
        return "فوق " + h
    elif ny > 0.70:
        return "تحت " + h
    return h

def describe_with_gemini(image, yolo_description):
    """تحسين الوصف بـ Gemini مع تبديل المفاتيح والنماذج."""
    prompt = f"""أنت مساعد بصري ذكي لمستخدم كفيف مصري.
الوصف المبدئي من كشف الأشياء: "{yolo_description}"
حلل الصورة واديني وصف نهائي بالعربي المصري، دقيق ومفيد.
لو فيه أخطاء في الوصف المبدئي صححها.
خلي الكلام طبيعي زي ما بتتكلم مع صاحبك، ومختصر."""

    for api_key in GEMINI_KEYS:
        if not api_key:
            continue
        try:
            client = genai.Client(api_key=api_key)
            for model_name in GEMINI_MODELS:
                for attempt in range(2):
                    try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[prompt, image],
                                config=types.GenerateContentConfig(
                                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                                )
                            )
                        except Exception:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[prompt, image]
                            )
                        return response.text
                    except Exception as e:
                        error_str = str(e)
                        if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                            time.sleep(1)
                            continue
                        else:
                            break
        except Exception:
            continue
    return None


# ═══════════ واجهة Streamlit ═══════════
st.set_page_config(page_title="بصيرة", page_icon="👁️", layout="centered")

# CSS عشان الواجهة تبقى أكبر وأوضح
st.markdown("""
<style>
    .big-title { font-size: 2.5rem; font-weight: bold; text-align: center; margin: 1rem 0; }
    .result-box {
        font-size: 1.6rem;
        padding: 1.5rem;
        background-color: #d4edda;
        border-radius: 15px;
        border: 3px solid #28a745;
        text-align: center;
        margin: 1.5rem 0;
        line-height: 1.6;
    }
    .welcome-box {
        font-size: 1.2rem;
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    div[data-testid="stCameraInput"] button {
        font-size: 1.3rem !important;
        padding: 1rem 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">👁️ بصيرة</div>', unsafe_allow_html=True)

# ترحيب مرة واحدة
if "welcomed" not in st.session_state:
    welcome_text = "أهلاً بيك في بصيرة. اضغط على زرار الكاميرا، وصور أي حاجة قدامك، وأنا هقولك هي إيه وفين."
    welcome_audio = text_to_speech(welcome_text)
    if welcome_audio:
        st.audio(welcome_audio, format='audio/mp3', autoplay=True)
    st.markdown(f'<div class="welcome-box">{welcome_text}</div>', unsafe_allow_html=True)
    st.session_state.welcomed = True

# الكاميرا بس
st.markdown("### 📷 صور اللي قدامك")
camera_image = st.camera_input("صور", label_visibility="collapsed")

if camera_image is not None:
    image = Image.open(camera_image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    with st.spinner("بصيرة بيشوف الصورة..."):
        # YOLO
        results = model(image, verbose=False, conf=0.15)[0]
        img_w, img_h = image.size

        # جمع الأشياء مع الاتجاه (بدون تكرار)
        seen = {}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            name = model.names[cls_id]
            if name in seen:
                continue
            position = get_position(box, img_w, img_h)
            arabic_name = translations.get(name, name)
            seen[name] = f"{arabic_name} {position}"

        if not seen:
            yolo_description = "مش شايف حاجة واضحة في الصورة. جرب تصور تاني وإضاءة أحسن."
        else:
            yolo_description = "أنا شايف: " + "، و".join(seen.values())

        # Gemini (لو فشل، نستخدم YOLO)
        final_description = describe_with_gemini(image, yolo_description)
        if final_description is None:
            final_description = yolo_description

        # عرض النتيجة بشكل كبير
        st.markdown(f'<div class="result-box">{final_description}</div>', unsafe_allow_html=True)

        # الصوت
        result_audio = text_to_speech(final_description)
        if result_audio:
            st.audio(result_audio, format='audio/mp3', autoplay=True)

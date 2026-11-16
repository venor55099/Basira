import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io
import os
import time
from google import genai
from google.genai import types

# --- إعداد Gemini ---
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
    """صوت باللهجة المصرية."""
    try:
        tts = gTTS(text=text, lang='ar', tld='com.eg', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception:
        return None

def get_position(box, img_width, img_height):
    """تحديد اتجاه الشيء."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    nx = cx / img_width
    ny = cy / img_height

    if nx < 0.33:
        h = "على شمالك"
    elif nx > 0.67:
        h = "على يمينك"
    else:
        h = "قصادك"

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
لو فيه أخطاء صححها. خلي الكلام طبيعي ومختصر."""

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
st.set_page_config(page_title="بصيرة", page_icon="👁️", layout="wide")

# CSS: الكاميرا تاخد الشاشة كلها
st.markdown("""
<style>
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        margin: 0 !important;
    }
    div[data-testid="stAppViewContainer"] > .main {
        padding: 0 !important;
    }

    /* الكاميرا تاخد الشاشة كلها */
    div[data-testid="stCameraInput"] {
        width: 100vw !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        position: fixed !important;
        top: 0; left: 0;
        z-index: 1;
    }
    div[data-testid="stCameraInput"] > div {
        width: 100vw !important;
        height: 100vh !important;
    }
    div[data-testid="stCameraInput"] video {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: cover !important;
    }

    /* زرار الكاميرا يبقى كبير وواضح */
    div[data-testid="stCameraInput"] button {
        font-size: 1.4rem !important;
        padding: 1rem 2rem !important;
        z-index: 10 !important;
        position: relative;
    }

    /* مربع النتيجة يظهر فوق الكاميرا */
    .result-box {
        position: fixed !important;
        bottom: 20px; left: 20px; right: 20px;
        z-index: 9999 !important;
        font-size: 1.6rem;
        padding: 1.2rem;
        background-color: rgba(212, 237, 218, 0.97);
        border-radius: 15px;
        border: 3px solid #28a745;
        text-align: center;
        line-height: 1.6;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* مربع الترحيب */
    .welcome-box {
        position: fixed !important;
        top: 20px; left: 20px; right: 20px;
        z-index: 9999 !important;
        font-size: 1.3rem;
        padding: 1rem;
        background-color: rgba(227, 242, 253, 0.97);
        border-radius: 10px;
        text-align: center;
        border: 2px solid #2196f3;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ترحيب مرة واحدة
if "welcomed" not in st.session_state:
    welcome_text = "أهلاً بيك في بصيرة. اضغط على زرار الكاميرا، وصور أي حاجة قدامك، وأنا هقولك هي إيه وفين."
    welcome_audio = text_to_speech(welcome_text)
    if welcome_audio:
        st.audio(welcome_audio, format='audio/mp3', autoplay=True)
    st.markdown(f'<div class="welcome-box">{welcome_text}</div>', unsafe_allow_html=True)
    st.session_state.welcomed = True

# الكاميرا (ملء الشاشة بـ CSS)
camera_image = st.camera_input("صور", label_visibility="collapsed")

if camera_image is not None:
    image = Image.open(camera_image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    with st.spinner("بصيرة بيشوف الصورة..."):
        results = model(image, verbose=False, conf=0.15)[0]
        img_w, img_h = image.size

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
            yolo_description = "مش شايف حاجة واضحة. جرب تصور تاني وإضاءة أحسن."
        else:
            yolo_description = "أنا شايف: " + "، و".join(seen.values())

        final_description = describe_with_gemini(image, yolo_description)
        if final_description is None:
            final_description = yolo_description

        st.markdown(f'<div class="result-box">{final_description}</div>', unsafe_allow_html=True)

        result_audio = text_to_speech(final_description)
        if result_audio:
            st.audio(result_audio, format='audio/mp3', autoplay=True)

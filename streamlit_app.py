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
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash"]

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
}

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='ar', tld='com.eg', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception:
        return None

def get_position(box, img_width, img_height):
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


# ═══════════ CSS الصحيح — الكاميرا ملء الشاشة ═══════════
st.set_page_config(page_title="بصيرة", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    /* ===== 1. إخفاء عناصر Streamlit ===== */
    #MainMenu, header, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ===== 2. تصفير كل الحشو والهوامش ===== */
    html, body, .stApp {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] > .main > div,
    .block-container,
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
        gap: 0 !important;
    }

    /* ===== 3. الكاميرا تاخد الشاشة كلها ===== */
    [data-testid="stCameraInput"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        max-width: 100vw !important;
        max-height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 10 !important;
        background: #000 !important;
        border: none !important;
        border-radius: 0 !important;
    }

    /* كل عناصر الكاميرا الداخلية */
    [data-testid="stCameraInput"] > div,
    [data-testid="stCameraInput"] > div > div,
    [data-testid="stCameraInput"] [data-testid="stVerticalBlock"] {
        width: 100% !important;
        height: 100% !important;
        max-width: 100% !important;
        max-height: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        background: transparent !important;
    }

    /* الفيديو والـcanvas يغطوا الشاشة */
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] canvas,
    [data-testid="stCameraInput"] img {
        width: 100vw !important;
        height: 100vh !important;
        max-width: 100vw !important;
        max-height: 100vh !important;
        object-fit: cover !important;
        display: block !important;
        border: none !important;
        border-radius: 0 !important;
    }

    /* ===== 4. زرار التصوير — كبير وواضح في النص تحت ===== */
    [data-testid="stCameraInput"] button {
        position: fixed !important;
        bottom: 140px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 9999 !important;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        padding: 1rem 2.5rem !important;
        border-radius: 50px !important;
        background: #28a745 !important;
        color: #fff !important;
        border: 3px solid #fff !important;
        box-shadow: 0 4px 25px rgba(0,0,0,0.7) !important;
        cursor: pointer !important;
    }

    /* ===== 5. صندوق النتيجة فوق الكاميرا ===== */
    .result-box {
        position: fixed !important;
        bottom: 20px !important;
        left: 15px !important;
        right: 15px !important;
        z-index: 9998 !important;
        font-size: 1.5rem !important;
        padding: 1rem !important;
        background: rgba(212, 237, 218, 0.97) !important;
        border: 3px solid #28a745 !important;
        border-radius: 15px !important;
        text-align: center !important;
        line-height: 1.5 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    }

    /* ===== 6. صندوق الترحيب فوق ===== */
    .welcome-box {
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        right: 15px !important;
        z-index: 9998 !important;
        font-size: 1.2rem !important;
        padding: 0.8rem 1rem !important;
        background: rgba(227, 242, 253, 0.97) !important;
        border: 2px solid #2196f3 !important;
        border-radius: 10px !important;
        text-align: center !important;
    }

    /* ===== 7. إخفاء مشغل الصوت بصريًا مع الحفاظ على عمله ===== */
    [data-testid="stAudio"] {
        position: fixed !important;
        bottom: -200px !important;
        left: -200px !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        opacity: 0.01 !important;
    }

    /* ===== 8. السبينر مخفي خلف الكاميرا ===== */
    [data-testid="stSpinner"] {
        z-index: 5 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════ الترحيب (مرة واحدة) ═══════════
if "welcomed" not in st.session_state:
    welcome_text = "أهلاً بيك في بصيرة. اضغط على الزرار الأخضر في النص، وصور أي حاجة قدامك، وأنا هقولك هي إيه وفين."
    welcome_audio = text_to_speech(welcome_text)
    if welcome_audio:
        st.audio(welcome_audio, format='audio/mp3', autoplay=True)
    st.markdown(f'<div class="welcome-box">{welcome_text}</div>', unsafe_allow_html=True)
    st.session_state.welcomed = True

# ═══════════ الكاميرا (ملء الشاشة) ═══════════
camera_image = st.camera_input("صور", label_visibility="collapsed")

if camera_image is not None:
    image = Image.open(camera_image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

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

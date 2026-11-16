import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import io
import os
import time
from google import genai
from google.genai import types

# --- إعداد Gemini (نفس الكود القديم) ---
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


# ═══════════ CSS (نفس الكود اللي ضبطناه) ═══════════
st.set_page_config(page_title="بصيرة", page_icon="👁️", layout="wide")
st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # نفس الـCSS بالكامل

# ═══════════ الترحيب (نفس الكود) ═══════════
if "welcomed" not in st.session_state:
    welcome_text = "أهلاً بيك في بصيرة. اضغط على الزرار الأخضر في النص، وصور أي حاجة قدامك، وأنا هقولك هي إيه وفين."
    welcome_audio = text_to_speech(welcome_text)
    if welcome_audio:
        st.audio(welcome_audio, format='audio/mp3', autoplay=True)
    st.markdown(f'<div class="welcome-box">{welcome_text}</div>', unsafe_allow_html=True)
    st.session_state.welcomed = True

# ═══════════ زر تبديل الكاميرا ═══════════
# دالة JavaScript للتبديل بين الكاميرا الأمامية والخلفية
switch_camera_js = """
() => {
    // البحث عن عنصر الفيديو في الصفحة
    const video = document.querySelector('video');
    if (!video || !video.srcObject) {
        alert('الكاميرا مش شغالة، جرب تصور الأول.');
        return;
    }
    const tracks = video.srcObject.getVideoTracks();
    const currentFacing = tracks[0]?.getSettings()?.facingMode || 'user';
    const newFacing = currentFacing === 'user' ? 'environment' : 'user';
    // إيقاف الكاميرا الحالية
    tracks.forEach(track => track.stop());
    // طلب الكاميرا الجديدة
    navigator.mediaDevices.getUserMedia({ video: { facingMode: newFacing } })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        })
        .catch(err => {
            alert('مش قادر أبدل الكاميرا: ' + err.message);
        });
}
"""

# زرار التبديل (بيظهر فوق الكاميرا)
if st.button("🔄 تبديل الكاميرا (أمامية / خلفية)"):
    streamlit_js_eval(js_expressions=switch_camera_js, key="switch_cam")

# ═══════════ الكاميرا ═══════════
camera_image = st.camera_input("صور", label_visibility="collapsed")

if camera_image is not None:
    # ... (باقي معالجة الصورة زي ما هي بالظبط)

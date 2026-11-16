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
}

def text_to_speech(text):
    """تحويل النص لصوت باللهجة المصرية."""
    try:
        tts = gTTS(text=text, lang='ar', tld='com.eg', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception:
        return None

def describe_with_gemini(image, yolo_description):
    """تحسين وصف YOLO باستخدام Gemini مع تبديل النماذج والمفاتيح."""
    prompt = f"""أنت مساعد بصري ذكي لمستخدم كفيف.
وصف مبدئي من نظام كشف الأشياء: "{yolo_description}"
حلل الصورة دي واديني وصف نهائي بالعربي، دقيق ومفيد.
لو الوصف المبدئي فيه أخطاء، صححها. وصف المشهد ككل."""

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


# --- واجهة Streamlit ---
st.set_page_config(page_title="بصيرة", page_icon="👁️", layout="centered")
st.title("👁️ بصيرة — المساعد البصري الذكي")

# ═══════════════════════════════════════════════════════════
# الترحيب: صوت واحد بس، مرة واحدة فقط، أول ما التطبيق يفتح
# ═══════════════════════════════════════════════════════════
if "welcomed" not in st.session_state:
    welcome_text = "أهلاً بيك في بصيرة. اضغط على زرار الكاميرا، وصور أي حاجة قدامك، وأنا هقولك هي إيه."
    welcome_audio = text_to_speech(welcome_text)
    if welcome_audio:
        st.audio(welcome_audio, format='audio/mp3', autoplay=True)
    st.session_state.welcomed = True
    st.info(welcome_text)

# ═══════════════════════════════════════════════════════════
# اختيار طريقة التصوير (نص بس — مفيش صوت تلقائي هنا)
# ═══════════════════════════════════════════════════════════
st.subheader("📸 اختار طريقة التصوير")
input_method = st.radio(
    "اختار طريقة إدخال الصورة:",
    ["📷 الكاميرا (صور دلوقتي)", "🖼️ رفع صورة من المعرض"],
    help="لو بتستخدم قارئ شاشة، اختار الخيار الأول وصور مباشرة"
)

image = None

if input_method == "📷 الكاميرا (صور دلوقتي)":
    st.write("اضغط على زرار تصوير الصورة اللي تحت، وصور اللي قدامك.")
    # زرار اسمع التعليمات (اختياري — المستخدم يضغط لو حب)
    if st.button("🔊 اسمع التعليمات"):
        instr_audio = text_to_speech("اضغط على زرار تصوير الصورة اللي تحت، وصور اللي قدامك.")
        if instr_audio:
            st.audio(instr_audio, format='audio/mp3', autoplay=True)
    
    camera_image = st.camera_input("صور اللي قدامك")
    if camera_image is not None:
        image = Image.open(camera_image)
else:
    uploaded_file = st.file_uploader("ارفع صورة من معرض الصور", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# ═══════════════════════════════════════════════════════════
# معالجة الصورة — الصوت الوحيد اللي يشتغل هنا هو صوت النتيجة
# ═══════════════════════════════════════════════════════════
if image is not None:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    st.image(image, caption="الصورة اللي رفعتها", use_column_width=True)

    with st.spinner("بصيرة بيشوف الصورة..."):
        # YOLO
        results = model(image, verbose=False, conf=0.15)[0]
        names = [model.names[int(c)] for c in results.boxes.cls]
        unique = list(dict.fromkeys(names))

        if not unique:
            yolo_description = "مش شايف حاجة واضحة في الصورة دي. جرب تصور تاني وإضاءة أحسن."
        else:
            arabic = [translations.get(n, n) for n in unique]
            yolo_description = "أنا شايف: " + "، ".join(arabic)

        # Gemini
        final_description = describe_with_gemini(image, yolo_description)
        if final_description is None:
            final_description = yolo_description

        # عرض النتيجة
        st.success(final_description)

        # صوت النتيجة — الصوت الوحيد اللي يشتغل هنا
        result_audio = text_to_speech(final_description)
        if result_audio:
            st.audio(result_audio, format='audio/mp3', autoplay=True)

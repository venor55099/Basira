import streamlit as st
from streamlit_camera_input_live import camera_input_live
from ultralytics import YOLO
from PIL import Image
import io
import os
import time
from cairo_tts import CairoTTS
from gemini_flux import GeminiFlux
import soundfile as sf

# --- 1. إعداد Gemini (مع مفاتيح متعددة) ---
# يتم تحميل المفاتيح من إعدادات Secrets في Streamlit Cloud
GEMINI_KEYS = []
i = 1
while f"GEMINI_KEY_{i}" in st.secrets:
    GEMINI_KEYS.append(st.secrets[f"GEMINI_KEY_{i}"])
    i += 1

# تهيئة مدير المفاتيح (يتم مرة واحدة فقط)
@st.cache_resource
def load_gemini_flux():
    if not GEMINI_KEYS:
        return None
    # "both" تعني أنه سيوزع الطلبات على كل المفاتيح والنماذج المتاحة
    return GeminiFlux(keys=GEMINI_KEYS, mode="both")

flux = load_gemini_flux()

# --- 2. تحميل نموذج YOLO (مرة واحدة) ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

# --- 3. تحميل محرك الصوت المصري (مرة واحدة) ---
@st.cache_resource
def load_tts_model():
    # التحميل على المعالج (CPU) لأنه أسرع وأكثر استقرارًا على السيرفرات المجانية
    return CairoTTS.from_pretrained("mohammedaly22/Cairo-TTS", device="cpu")

tts = load_tts_model()

# --- 4. قاموس الترجمة العربية ---
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

# --- 5. دالة تحديد الاتجاهات ---
def get_position(box, img_width, img_height):
    """تحديد اتجاه الشيء (يمين/شمال/فوق/تحت/قصاد)."""
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

# --- 6. دالة تحسين الوصف بـ Gemini ---
def enhance_description_with_gemini(yolo_description):
    """تحسين الوصف باستخدام مدير المفاتيح المتعدد."""
    if flux is None:
        return None # العودة لوصف YOLO في حال عدم وجود مفاتيح

    prompt = f"""أنت مساعد بصري ذكي لمستخدم كفيف مصري.
الوصف المبدئي من كشف الأشياء: "{yolo_description}"
حلل الوصف واديني جملة نهائية بالعربي المصري، دقيقة ومفيدة.
لو فيه أخطاء في الوصف المبدئي صححها. خلي الكلام طبيعي ومختصر."""

    try:
        response = flux.generate(prompt)
        return response["response"]
    except Exception:
        return None # العودة لوصف YOLO في حال فشل كل المحاولات

# --- 7. دالة تحويل النص إلى صوت مصري ---
def text_to_speech(text):
    """توليد صوت باللهجة المصرية باستخدام Cairo-TTS."""
    try:
        # توليد مصفوفة الصوت
        audio_array = tts.generate(text)
        # تحويلها إلى صيغة WAV قابلة للتشغيل في المتصفح
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio_array, 24000, format='WAV')
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"حدث خطأ في توليد الصوت: {e}")
        return None

# --- 8. واجهة Streamlit والتنسيق ---
st.set_page_config(page_title="بصيرة", page_icon="👁️", layout="wide")

# CSS لجعل الصفحة كلها كاميرا وإخفاء العناصر غير الضرورية
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding: 0 !important; max-width: 100% !important;}
    div[data-testid="stCameraInputLive"] video {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: cover !important;
    }
    .result-box {
        position: fixed;
        bottom: 20px; left: 20px; right: 20px;
        z-index: 9999;
        font-size: 1.5rem; padding: 1rem;
        background-color: rgba(212, 237, 218, 0.95);
        border-radius: 15px; border: 3px solid #28a745;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 9. الترحيب الصوتي (مرة واحدة فقط) ---
if "welcomed" not in st.session_state:
    welcome_text = "أهلاً بيك في بصيرة. الكاميرا شغالة، صور أي حاجة قدامك وأنا هقولك هي إيه وفين."
    welcome_audio = text_to_speech(welcome_text)
    if welcome_audio:
        st.audio(welcome_audio, format='audio/wav', autoplay=True)
    st.session_state.welcomed = True

# --- 10. الكاميرا المفتوحة تلقائيًا ---
# ملاحظة: specific facing_mode="environment" قد لا يكون مدعومًا في هذه المكتبة.
# الكود التالي سيحاول فتح الكاميرا الافتراضية (غالبًا الخلفية على الموبايل).
image = camera_input_live(key="camera_live", show_controls=False)

# --- 11. المعالجة عند توفر صورة ---
if image is not None:
    pil_image = Image.open(image)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    with st.spinner("بصيرة بيشوف الصورة..."):
        # كشف الأشياء
        results = model(pil_image, verbose=False, conf=0.15)[0]
        img_w, img_h = pil_image.size

        # جمع الأشياء مع تحديد الاتجاه (بدون تكرار)
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
            yolo_description = "مش شايف حاجة واضحة. جرب تقرب شوية."
        else:
            yolo_description = "أنا شايف: " + "، و".join(seen.values())

        # تحسين الوصف بـ Gemini
        final_description = enhance_description_with_gemini(yolo_description)
        if final_description is None:
            final_description = yolo_description # استخدام وصف YOLO كخطة بديلة

        # عرض النتيجة في صندوق فوق الكاميرا
        st.markdown(f'<div class="result-box">{final_description}</div>', unsafe_allow_html=True)

        # تشغيل الصوت المصري
        result_audio = text_to_speech(final_description)
        if result_audio:
            st.audio(result_audio, format='audio/wav', autoplay=True)

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import google.generativeai as genai

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지", layout="wide")
st.title("YOLO 객체 탐지기")

# --- 1. 유틸리티 함수: 이미지 최적화 (속도 향상) ---
def resize_image(image):
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

# --- 2. 모델 로드 ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

MODEL_PATH = "model/best.pt"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다: {e}")
    st.stop()

# --- 3. 세션 상태 및 콜백 함수 ---
if "detect_clicked" not in st.session_state:
    st.session_state.detect_clicked = False
if "interpret_clicked" not in st.session_state:
    st.session_state.interpret_clicked = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = ""
if "yolo_results" not in st.session_state:
    st.session_state.yolo_results = {}
if "gemini_result" not in st.session_state:
    st.session_state.gemini_result = None

def on_detect_btn():
    st.session_state.detect_clicked = True
    st.session_state.interpret_clicked = False
    st.session_state.gemini_result = None
    st.session_state.yolo_results = {}

def on_interpret_btn():
    st.session_state.interpret_clicked = True

# --- 4. UI 구성 ---
st.sidebar.header("🤖 Gemini API 설정")
gemini_api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
st.sidebar.button("탐지 실행", on_click=on_detect_btn)

uploaded_file = st.file_uploader("탐지할 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.detect_clicked = False
        st.session_state.yolo_results = {}
        st.session_state.gemini_result = None
        
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    # 탐지 실행
    if st.session_state.detect_clicked:
        if not st.session_state.yolo_results:
            with st.spinner("이미지 분석 중..."):
                results = model.predict(source=image, conf=conf_threshold)
                r = results[0]
                im_rgb = Image.fromarray(r.plot()[..., ::-1])
                
                det_data = [{"ID": i, "클래스명": model.names[int(b.cls.item())], "Conf": f"{b.conf.item():.4f}"} for i, b in enumerate(r.boxes)]
                st.session_state.yolo_results = {"image": im_rgb, "data": det_data, "count": len(r.boxes)}

        if st.session_state.yolo_results:
            with col2:
                st.subheader("탐지 결과")
                st.image(st.session_state.yolo_results["image"], use_container_width=True)
            
            st.subheader("객체 정보")
            st.table(st.session_state.yolo_results["data"])
            
            # AI 해석
            st.subheader("💡 Gemini AI 해석")
            if not gemini_api_key:
                st.warning("API Key가 필요합니다.")
            else:
                st.button("결과 해석 요청하기", on_click=on_interpret_btn)
                if st.session_state.interpret_clicked:
                    if not st.session_state.gemini_result:
                        with st.spinner("분석 중..."):
                            try:
                                genai.configure(api_key=gemini_api_key)
                                # 코드 내 모델 호출 부분 수정
                                # 현재 2.0 모델이 가장 안정적입니다.
                                 model_gemini = genai.GenerativeModel('gemini-2.0-flash')
                                
                                # 속도 최적화: 이미지 리사이징 후 전달
                                optimized_img = resize_image(image)
                                prompt = f"다음은 객체 탐지 결과입니다: {st.session_state.yolo_results['data']}. 상황을 설명해주세요."
                                
                                response = model_gemini.generate_content(
                                    [prompt, optimized_img],
                                    generation_config=genai.types.GenerationConfig(max_output_tokens=300)
                                )
                                st.session_state.gemini_result = response.text
                            except Exception as e:
                                st.error(f"오류 발생: {e}")
                    
                    if st.session_state.gemini_result:
                        st.info(st.session_state.gemini_result)

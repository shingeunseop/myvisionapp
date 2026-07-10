import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import google.generativeai as genai

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지", layout="wide")
st.title("YOLO 객체 탐지기")

# --- Gemini API 설정 (화면 로드 후 직접 입력) ---
st.sidebar.header("🤖 Gemini API 설정")
gemini_api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
st.sidebar.markdown("*(API Key가 있어야 결과 해석 기능이 동작합니다.)*")
st.sidebar.divider()

# 1. 모델 로드 (캐싱하여 재로딩 방지)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 모델 경로 (깃헙 리포지토리 기준 상대 경로로 설정)
MODEL_PATH = "model/best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다: {e}")
    st.stop()

# 2. 이미지 업로드 UI
uploaded_file = st.file_uploader("탐지할 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # PIL을 사용하여 이미지 읽기
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    # 파라미터 조정 UI
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

    if st.sidebar.button("탐지 실행"):
        with st.spinner("이미지를 분석하고 있습니다..."):
            # 3. 모델 예측
            results = model.predict(source=image, conf=conf_threshold)
            r = results[0]
            
            # 4. 바운딩 박스 그려진 이미지 추출 (BGR -> RGB 변환)
            im_array = r.plot(line_width=2) 
            im_rgb = im_array[..., ::-1] 
            res_image = Image.fromarray(im_rgb)

            with col2:
                st.subheader("탐지 결과 이미지")
                st.image(res_image, use_container_width=True)

            # 5. 결과 정보 출력 (데이터프레임 형태)
            st.write("---")
            st.subheader("탐지된 객체 정보")
            st.write(f"**총 탐지된 객체 수:** {len(r.boxes)}")

            if len(r.boxes) > 0:
                det_data = []
                for i, box in enumerate(r.boxes):
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    conf = float(box.conf.item())
                    xyxy = box.xyxy.cpu().numpy().ravel()
                    
                    det_data.append({
                        "ID": i,
                        "클래스명": class_name,
                        "정확도(Conf)": f"{conf:.4f}",
                        "BBox (x1, y1, x2, y2)": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                    })
                st.table(det_data)

                # --- 6. Gemini 결과 해석 기능 ---
                st.write("---")
                st.subheader("💡 Gemini AI 결과 해석")
                
                # API 키가 입력되지 않았을 경우 경고 문구 출력
                if not gemini_api_key:
                    st.warning("결과를 해석하려면 왼쪽 사이드바에 Gemini API Key를 입력해주세요.")
                else:
                    if st.button("결과 해석 요청하기"):
                        with st.spinner("Gemini가 탐지 결과를 바탕으로 상황을 분석 중입니다..."):
                            try:
                                # 화면에서 입력받은 API 키로 설정
                                genai.configure(api_key=gemini_api_key)
                                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                                
                                detected_summary = ", ".join([f"{item['클래스명']}({item['정확도(Conf)']})" for item in det_data])
                                
                                prompt = f"""
                                당신은 전문적인 이미지 분석가입니다. YOLO 객체 탐지 모델이 이 이미지에서 다음과 같은 객체들을 발견했습니다:
                                - 탐지된 객체 목록: {detected_summary}
                                
                                첨부된 원본 이미지와 위 탐지 결과 데이터를 종합하여, 이 사진이 어떤 상황인지, 
                                주요 객체들이 어떤 상호작용을 하고 있는지 한글로 자연스럽고 친절하게 설명해주세요.
                                """
                                
                                response = gemini_model.generate_content([prompt, image])
                                st.info(response.text)
                                
                            except Exception as e:
                                st.error(f"Gemini API 호출 중 오류가 발생했습니다. API Key가 올바른지 확인해주세요.\n\n에러 내용: {e}")

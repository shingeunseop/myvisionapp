import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import google.generativeai as genai

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지", layout="wide")
st.title("YOLO 객체 탐지기")

# --- 1. 모델 로드 ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

MODEL_PATH = "model/best.pt"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다: {e}")
    st.stop()

# --- 2. 강력한 Session State (상태 유지) 및 콜백 함수 ---
# 화면이 백 번 새로고침되어도 데이터가 날아가지 않도록 설계된 공간입니다.
if "detect_clicked" not in st.session_state:
    st.session_state.detect_clicked = False
if "interpret_clicked" not in st.session_state:
    st.session_state.interpret_clicked = False
if "current_file_bytes" not in st.session_state:
    st.session_state.current_file_bytes = b""
if "yolo_results" not in st.session_state:
    st.session_state.yolo_results = {}
if "gemini_result" not in st.session_state:
    st.session_state.gemini_result = None

# 콜백 1: '탐지 실행' 버튼을 눌렀을 때 작동
def on_detect_btn():
    st.session_state.detect_clicked = True
    st.session_state.interpret_clicked = False
    st.session_state.gemini_result = None  # 새로 탐지하면 기존 AI 해석은 지움
    st.session_state.yolo_results = {}     # 재탐지를 위해 기존 데이터 비움

# 콜백 2: '결과 해석 요청하기' 버튼을 눌렀을 때 작동
def on_interpret_btn():
    st.session_state.interpret_clicked = True

# --- 3. 사이드바 설정 ---
st.sidebar.header("🤖 Gemini API 설정")
gemini_api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
st.sidebar.markdown("*(API Key가 있어야 결과 해석 기능이 동작합니다.)*")
st.sidebar.divider()

conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

# 💡 핵심: on_click을 사용해 버튼 클릭 사실을 시스템에 강제 각인시킵니다.
st.sidebar.button("탐지 실행", on_click=on_detect_btn)

# --- 4. 메인 화면 로직 ---
uploaded_file = st.file_uploader("탐지할 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 파일의 고유 데이터를 읽어와서, '새로운 사진'이 올라왔는지 완벽하게 감지합니다.
    file_bytes = uploaded_file.getvalue()
    
    if st.session_state.current_file_bytes != file_bytes:
        # 다른 사진이 올라오면 모든 과거 기록을 싹 초기화합니다.
        st.session_state.current_file_bytes = file_bytes
        st.session_state.detect_clicked = False
        st.session_state.interpret_clicked = False
        st.session_state.yolo_results = {}
        st.session_state.gemini_result = None
        
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    # --- 5. 탐지 실행 및 결과 화면 ---
    if st.session_state.detect_clicked:
        
        # 아직 탐지 결과가 메모리에 없다면 YOLO를 돌려서 저장합니다.
        if not st.session_state.yolo_results:
            with st.spinner("이미지를 분석하고 있습니다..."):
                results = model.predict(source=image, conf=conf_threshold)
                r = results[0]
                
                im_array = r.plot(line_width=2) 
                im_rgb = im_array[..., ::-1] 
                res_image = Image.fromarray(im_rgb)
                
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
                        "BBox": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                    })
                    
                # 결과를 딕셔너리 형태로 단단히 묶어 저장
                st.session_state.yolo_results = {
                    "image": res_image,
                    "data": det_data,
                    "count": len(r.boxes)
                }

        # 메모리에 저장된 YOLO 결과가 있다면 언제든 화면에 뿌려줍니다. (화면이 깜빡여도 유지됨)
        if st.session_state.yolo_results:
            with col2:
                st.subheader("탐지 결과 이미지")
                st.image(st.session_state.yolo_results["image"], use_container_width=True)
                
            st.write("---")
            st.subheader("탐지된 객체 정보")
            st.write(f"**총 탐지된 객체 수:** {st.session_state.yolo_results['count']}")
            
            if st.session_state.yolo_results["count"] > 0:
                st.table(st.session_state.yolo_results["data"])
                
                # --- 6. AI 결과 해석 ---
                st.write("---")
                st.subheader("💡 Gemini AI 결과 해석")
                
                if not gemini_api_key:
                    st.warning("결과를 해석하려면 왼쪽 사이드바에 Gemini API Key를 입력해주세요.")
                else:
                    # 💡 핵심: 해석 버튼 역시 on_click 콜백으로 처리합니다.
                    st.button("결과 해석 요청하기", on_click=on_interpret_btn)
                    
                    if st.session_state.interpret_clicked:
                        # 해석 결과가 메모리에 없다면 Gemini API 호출
                        if not st.session_state.gemini_result:
                            with st.spinner("Gemini가 탐지 결과를 바탕으로 상황을 분석 중입니다..."):
                                try:
                                    genai.configure(api_key=gemini_api_key)
                                    gemini_model = genai.GenerativeModel('gemini-3.5-flash')
                                    
                                    detected_summary = ", ".join([f"{item['클래스명']}({item['정확도(Conf)']})" for item in st.session_state.yolo_results["data"]])
                                    
                                    prompt = f"""
                                    당신은 전문적인 이미지 분석가입니다. YOLO 객체 탐지 모델이 이 이미지에서 다음과 같은 객체들을 발견했습니다:
                                    - 탐지된 객체 목록: {detected_summary}
                                    
                                    첨부된 원본 이미지와 위 탐지 결과 데이터를 종합하여, 이 사진이 어떤 상황인지, 
                                    주요 객체들이 어떤 상호작용을 하고 있는지 한글로 자연스럽고 친절하게 설명해주세요.
                                    """
                                    
                                    response = gemini_model.generate_content([prompt, image])
                                    
                                    # 얻어낸 결과를 메모리에 영구 저장!
                                    st.session_state.gemini_result = response.text
                                    
                                except Exception as e:
                                    st.error(f"Gemini API 호출 중 오류가 발생했습니다.\n\n에러 내용: {e}")
                                    st.session_state.interpret_clicked = False  # 실패 시 다시 누를 수 있도록 초기화
                                    
                        # 메모리에 저장된 해석 결과가 있다면 화면에 출력
                        if st.session_state.gemini_result:
                            st.success("✅ AI 해석 완료")
                            st.info(st.session_state.gemini_result)

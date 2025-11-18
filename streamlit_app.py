# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1YRU7LMB9nbcSeAPwjYrtzHNwErovcMdD")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    
     labels[0]: {
      "texts": ["고양이상", "아이브", "레전드"],
     "images": ["https://cdn.mhnse.com/news/photo/202509/463155_586604_229.jpg", "https://thumb.pann.com/tc_480/http://fimg5.pann.com/new/download.jsp?FileID=67817415"],
      "videos": ["https://www.youtube.com/watch?v=0pqsNFFHyhc"]
     },
labels[1]: {
      "texts": ["겨울", "에스파", "부산 여자"],
     "images": ["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzK0-RA_POTJWRUWR7Je7fQUrem7K7zRaw1A&s", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfKzuJ8BksDWMQmpPARG94rBNjRdkRAUpWWQ&s"],
      "videos": ["https://www.youtube.com/watch?v=g15OJDuGDCw&list=RDg15OJDuGDCw&start_radio=1"]
     },
    labels[2]: {
      "texts": ["아이돌 3대 미녀", "엔믹스", "자연 미인"],
     "images": ["data:image/jpeg;base64,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", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQbmpwpqvXbFR4ihRA7e-cIZiDtxQWf_1MbxA&s"],
      "videos": ["https://www.youtube.com/watch?v=6loho6S--Ag&list=RD6loho6S--Ag&start_radio=1"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")

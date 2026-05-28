import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tumor FD Analyzer", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] { background-color: #f8f8f8 !important; }

div.stButton > button {
    background-color: #1a1a2e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: none !important;
    box-shadow: none !important;
}
div.stButton > button:hover,
div.stButton > button:active,
div.stButton > button:focus {
    background-color: #1a1a2e !important;
    color: #ffffff !important;
    outline: none !important;
    box-shadow: none !important;
}

button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #555555 !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a1a2e !important;
    border-bottom: 2px solid #1a1a2e !important;
    font-weight: 700 !important;
}
button[data-baseweb="tab"]:hover {
    background-color: transparent !important;
    color: #1a1a2e !important;
}

.rec-card {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 30px 36px;
    border-left: 6px solid #2a52be;
    margin-top: 20px;
}
.rec-title { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 8px; }
.rec-sub   { font-size: 1.05rem; color: #333333; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Session state 초기화 ───────────────────────────────────────────────────────
if "fd_df" not in st.session_state:
    st.session_state.fd_df = None
if "weights" not in st.session_state:
    st.session_state.weights = {"CA9": 1.0, "ABCB1": 1.0, "MKI67": 1.0}
if "thresholds" not in st.session_state:
    st.session_state.thresholds = {"high": 1.0, "low": -1.0}

# ── Helper functions ───────────────────────────────────────────────────────────
def compute_fd(df, weights):
    genes = ["CA9", "ABCB1", "MKI67"]
    z_cols = []
    for g in genes:
        z_col = f"{g}_z"
        if z_col in df.columns:
            z_cols.append((z_col, weights.get(g, 1.0)))
        elif g in df.columns:
            vals = df[g].astype(float)
            std = vals.std()
            df[z_col] = (vals - vals.mean()) / std if std > 0 else 0.0
            z_cols.append((z_col, weights.get(g, 1.0)))
    if not z_cols:
        return df
    fd = sum(df[col] * w for col, w in z_cols) / len(z_cols)
    df["FD_score"] = fd
    return df

def classify(score, high_thr, low_thr):
    if score >= high_thr:
        return "Core"
    elif score <= low_thr:
        return "Edge"
    else:
        return "Hybrid"

def get_recommendation(df, high_thr, low_thr):
    counts   = df["Region"].value_counts()
    total    = len(df)
    dominant = counts.idxmax() if len(counts) else "Hybrid"

    if dominant == "Core":
        route  = "정맥 주사 (IV Injection)"
        reason = ("종양 코어 영역이 우세합니다. 저산소 환경으로 인해 국소 주사의 침투가 "
                  "제한되므로 전신 혈류를 통한 정맥 투여가 권장됩니다.")
    elif dominant == "Edge":
        route  = "국소 주사 (Local Injection)"
        reason = ("종양 경계 영역이 우세합니다. 혈관 접근성이 양호하여 "
                  "국소 직접 주사로 높은 약물 농도를 확보할 수 있습니다.")
    else:
        route  = "복합 투여 (Combined Approach)"
        reason = ("코어·경계 혼합 영역이 우세합니다. "
                  "전신 투여와 국소 투여를 병행하는 복합 전략이 권장됩니다.")
    return route, reason

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 데이터 입력 & FD 계산",
    "💊 약물 투여 경로 추천",
    "🔬 민감도 분석",
    "🗺️ H&E 공간 지도"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📂 데이터 입력 및 FD 점수 계산")
    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### 파일 업로드")
        fd_file = st.file_uploader("fd_input.csv", type=["csv"], key="fd_csv")

        st.markdown("### 가중치 설정")
        w_ca9   = st.slider("CA9 가중치",   0.0, 2.0,
                            float(st.session_state.weights["CA9"]),   0.1)
        w_abcb1 = st.slider("ABCB1 가중치", 0.0, 2.0,
                            float(st.session_state.weights["ABCB1"]), 0.1)
        w_mki67 = st.slider("MKI67 가중치", 0.0, 2.0,
                            float(st.session_state.weights["MKI67"]), 0.1)

        st.markdown("### 임계값 설정")
        thr_high = st.slider("High FD (Core)",  -3.0, 3.0,
                             float(st.session_state.thresholds["high"]), 0.1)
        thr_low  = st.slider("Low FD (Edge)",   -3.0, 3.0,
                             float(st.session_state.thresholds["low"]),  0.1)

        if st.button("▶ FD 점수 계산"):
            if fd_file is None:
                st.error("fd_input.csv 파일을 먼저 업로드하세요.")
            else:
                df = pd.read_csv(fd_file)
                weights = {"CA9": w_ca9, "ABCB1": w_abcb1, "MKI67": w_mki67}

                if "FD_score" not in df.columns:
                    df = compute_fd(df, weights)
                else:
                    st.info("CSV에 FD_score 컬럼이 존재합니다. 기존 값을 사용합니다.")

                df["Region"] = df["FD_score"].apply(
                    lambda s: classify(s, thr_high, thr_low))

                st.session_state.fd_df      = df
                st.session_state.weights    = weights
                st.session_state.thresholds = {"high": thr_high, "low": thr_low}
                st.success("✅ FD 점수 계산 완료!")

    with col_r:
        if st.session_state.fd_df is not None:
            df = st.session_state.fd_df
            st.markdown("### FD 점수 분포")

            fig, ax = plt.subplots(figsize=(5, 3.5))
            region_colors = {"Core": "#e74c3c", "Hybrid": "#f39c12", "Edge": "#2ecc71"}
            for region, grp in df.groupby("Region"):
                ax.hist(grp["FD_score"], bins=20, alpha=0.7,
                        color=region_colors.get(region, "gray"), label=region)
            ax.axvline(st.session_state.thresholds["high"],
                       color="#e74c3c", linestyle="--", linewidth=1.2, label="High thr")
            ax.axvline(st.session_state.thresholds["low"],
                       color="#2ecc71", linestyle="--", linewidth=1.2, label="Low thr")
            ax.set_xlabel("FD Score", fontsize=10)
            ax.set_ylabel("Count",    fontsize=10)
            ax.set_title("FD Score Distribution", fontsize=11)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### 데이터 미리보기")
            preview_cols = [c for c in ["barcode", "FD_score", "Region"] if c in df.columns]
            st.dataframe(df[preview_cols].head(20), use_container_width=True)
        else:
            st.info("파일을 업로드하고 FD 점수 계산 버튼을 누르세요.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 💊 약물 투여 경로 추천")
    st.markdown("---")

    if st.session_state.fd_df is None:
        st.warning("Tab 1에서 먼저 FD 점수를 계산하세요.")
    else:
        df       = st.session_state.fd_df
        thr_high = st.session_state.thresholds["high"]
        thr_low  = st.session_state.thresholds["low"]
        route, reason = get_recommendation(df, thr_high, thr_low)

        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">🩺 권장 투여 경로: {route}</div>
            <div class="rec-sub">{reason}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔬 민감도 분석")
    st.markdown("---")

    if st.session_state.fd_df is None:
        st.warning("Tab 1에서 먼저 FD 점수를 계산하세요.")
    else:
        df_base  = st.session_state.fd_df.copy()
        thr_high = st.session_state.thresholds["high"]
        thr_low  = st.session_state.thresholds["low"]

        st.markdown("임계값 범위를 변화시켰을 때 Core / Hybrid / Edge 비율 변화를 확인합니다.")
        sens_range = st.slider("분석 임계값 범위 (±)", 0.1, 2.0, 1.0, 0.1)

        if st.button("민감도 분석 실행"):
            steps   = np.arange(-sens_range, sens_range + 0.05, 0.1)
            core_r, hybrid_r, edge_r = [], [], []
            for delta in steps:
                tmp = df_base.copy()
                tmp["Region"] = tmp["FD_score"].apply(
                    lambda s: classify(s, thr_high + delta, thr_low + delta))
                cnt = tmp["Region"].value_counts()
                tot = len(tmp)
                core_r.append(cnt.get("Core",   0) / tot * 100)
                hybrid_r.append(cnt.get("Hybrid", 0) / tot * 100)
                edge_r.append(cnt.get("Edge",   0) / tot * 100)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(steps, core_r,   color="#e74c3c", lw=2, label="Core %")
            ax2.plot(steps, hybrid_r, color="#f39c12", lw=2, label="Hybrid %")
            ax2.plot(steps, edge_r,   color="#2ecc71", lw=2, label="Edge %")
            ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax2.set_xlabel("Threshold Delta", fontsize=10)
            ax2.set_ylabel("Region %",        fontsize=10)
            ax2.set_title("Sensitivity Analysis", fontsize=11)
            ax2.legend(fontsize=9)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧬 H&E 기반 공간 유전자 발현 맵")

    img_file = st.file_uploader("tissue_hires_image.png 업로드", type=["png", "jpg"], key="img")
    pos_file = st.file_uploader("tissue_positions.csv 업로드", type=["csv"], key="pos")
    sf_file  = st.file_uploader("scalefactors_json.json 업로드", type=["json"], key="sf")

    if img_file and pos_file and sf_file:
        import json
        from PIL import Image
        import matplotlib.pyplot as plt

        # ── ① scale factor ──────────────────────────────────────────
        sf = json.load(sf_file)
        scale = sf.get("tissue_hires_scalef", 1.0)

        # ── ② df에 픽셀 좌표가 이미 있는지 먼저 확인 ─────────────────
        has_fullres = ("pxl_row_in_fullres" in df.columns and
                       "pxl_col_in_fullres" in df.columns)
        has_pxl     = ("pxl_row" in df.columns and
                       "pxl_col" in df.columns)

        if has_fullres:
            # ✅ Case A: df에 좌표 있음 → merge 불필요!
            merged = df.copy()
            row_col, col_col = "pxl_row_in_fullres", "pxl_col_in_fullres"
            st.info("✅ df에서 직접 좌표 사용 (pxl_row_in_fullres)")

        elif has_pxl:
            merged = df.copy()
            row_col, col_col = "pxl_row", "pxl_col"
            st.info("✅ df에서 직접 좌표 사용 (pxl_row)")

        else:
            # ✅ Case B: pos 파일에서 merge 필요
            pos_file.seek(0)
            first_line = pos_file.readline().decode("utf-8").strip()
            pos_file.seek(0)

            if "barcode" in first_line or "pxl_row" in first_line:
                pos = pd.read_csv(pos_file)
            else:
                pos = pd.read_csv(
                    pos_file, header=None,
                    names=["barcode", "in_tissue", "array_row",
                           "array_col", "pxl_row", "pxl_col"]
                )

            pos.columns = pos.columns.str.strip()
            if "barcode" not in pos.columns:
                pos = pos.rename(columns={pos.columns[0]: "barcode"})
            if "in_tissue" in pos.columns:
                pos = pos[pos["in_tissue"] == 1].copy()

            # barcode 정규화 (-1 suffix 제거 후 비교)
            df["barcode_key"]  = df["barcode"].astype(str).str.strip().str.replace(r"-\d+$", "", regex=True)
            pos["barcode_key"] = pos["barcode"].astype(str).str.strip().str.replace(r"-\d+$", "", regex=True)

            if "pxl_row_in_fullres" in pos.columns:
                row_col, col_col = "pxl_row_in_fullres", "pxl_col_in_fullres"
            else:
                row_col, col_col = "pxl_row", "pxl_col"

            merged = df.merge(
                pos[["barcode_key", row_col, col_col]],
                on="barcode_key", how="inner"
            )

            if merged.empty:
                # 디버깅: barcode 샘플 출력
                st.error("❌ barcode 매칭 실패")
                st.write("df barcode 예시:", df["barcode"].head(3).tolist())
                st.write("pos barcode 예시:", pos["barcode"].head(3).tolist())
                st.stop()

            row_col_use, col_col_use = row_col, col_col

        # ── ③ 픽셀 좌표 스케일 적용 ─────────────────────────────────
        if has_fullres or has_pxl:
            row_col_use = row_col
            col_col_use = col_col

        merged["px_row"] = merged[row_col_use] * scale
        merged["px_col"] = merged[col_col_use] * scale

        # ── ④ FD_score 컬럼 확인 ────────────────────────────────────
        if "FD_score" not in merged.columns:
            st.error("❌ FD_score 컬럼이 없습니다. Tab 1에서 먼저 계산해주세요.")
            st.stop()

        # ── ⑤ 시각화 ────────────────────────────────────────────────
        img = Image.open(img_file)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        sc = ax.scatter(
            merged["px_col"],   # x축 = 열(col)
            merged["px_row"],   # y축 = 행(row)
            c=merged["FD_score"],
            cmap="RdYlGn_r",
            s=20,
            alpha=0.8,
            linewidths=0
        )
        plt.colorbar(sc, ax=ax, label="FD Score")
        ax.set_title("Spatial FD Score Map", fontsize=14)
        ax.axis("off")
        st.pyplot(fig)
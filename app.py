import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from ir_simulator import generate_ir_spectrum, optimize_mixture_ratios
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="Polymer IR Simulator Pro", layout="wide")

st.title("🧪 고분자 IR 스펙트럼 시뮬레이터 (ML + Quantum)")
st.markdown("""
이 시스템은 **GNN(Graph Neural Network)**과 **양자화학(xTB)**을 결합하여 고분자 및 혼합물의 IR 스펙트럼을 예측합니다.
중합도($n$)에 따른 사슬 효과 및 수소 결합 효과가 물리적으로 모델링됩니다.
""")

# 세션 상태 초기화
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False
if "components" not in st.session_state:
    st.session_state.components = []

# 사이드바 설정
st.sidebar.header("📋 성분 구성 및 제어")

# 1. 성분 입력 테이블
default_data = {
    "SMILES": ["CCCCCCCCCCCP(=O)(O)OCC(C)OC(=O)C(=C)C", ""], # 10-MDP 예시
    "배합비(Ratio)": [1.0, 0.0],
    "중합도(n)": [1, 1]
}
df = pd.DataFrame(default_data)
edited_df = st.sidebar.data_editor(df, num_rows="dynamic", use_container_width=True)

# 2. 시뮬레이션 옵션
st.sidebar.header("⚙️ 시뮬레이션 옵션")
show_labels = st.sidebar.toggle("작용기 라벨 표시", value=True)
use_qc = st.sidebar.checkbox("양자화학(xTB) 정밀 보정", value=False, 
                            help="신규 분자는 계산에 5~10초가 소요되지만 물리적 정확도가 비약적으로 향상됩니다.")

if use_qc:
    st.sidebar.warning("⚠️ QC 계산 중에는 앱이 잠시 멈출 수 있습니다. (캐싱 지원)")

generate_btn = st.sidebar.button("IR 스펙트럼 생성 및 분석", use_container_width=True)

# 3. 혼합물 배합비 최적화 (실험 데이터 기반 역예측)
st.sidebar.divider()
st.sidebar.header("🔍 배합비 최적화 (역예측)")
uploaded_file = st.sidebar.file_uploader("실험 IR 데이터 업로드 (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        exp_data = pd.read_csv(uploaded_file)
        if "wavenumber" in exp_data.columns and "transmittance" in exp_data.columns:
            st.sidebar.success("실험 데이터 로드 완료")
            if st.sidebar.button("최적 배합비 계산 시작"):
                with st.spinner("최적 배합비 탐색 중..."):
                    components = []
                    for _, row in edited_df.iterrows():
                        if row["SMILES"]:
                            components.append({"smiles": row["SMILES"], "ratio": 1.0, "n": row["중합도(n)"]})
                    
                    optimized_ratios = optimize_mixture_ratios(
                        components, 
                        exp_data["wavenumber"].values, 
                        exp_data["transmittance"].values
                    )
                    
                    if optimized_ratios is not None:
                        for i, ratio in enumerate(optimized_ratios):
                            edited_df.iloc[i, 1] = round(ratio, 3)
                        st.sidebar.success("최적화 완료! 테이블에 반영되었습니다.")
                        st.rerun()
    except Exception as e:
        st.sidebar.error(f"오류: {e}")

# 메인 로직 처리
if generate_btn:
    st.session_state.generate_clicked = True
    components = []
    for _, row in edited_df.iterrows():
        s = str(row.get("SMILES", "")).strip()
        r = float(row.get("배합비(Ratio)", 0))
        n = int(row.get("중합도(n)", 1))
        if s and r > 0:
            components.append({"smiles": s, "ratio": r, "n": n})
    st.session_state.components = components

if st.session_state.generate_clicked:
    components = st.session_state.components
    if not components:
        st.warning("유효한 성분 정보를 입력해주세요.")
    else:
        try:
            with st.spinner("시뮬레이션 엔진 가동 중 (ML + Quantum)..."):
                wavenumbers, transmittance, all_identified_groups, mols = generate_ir_spectrum(components, use_qc=use_qc)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("🖼️ 분자 구조(2D)")
                    for i, (mol, comp) in enumerate(zip(mols, components)):
                        if mol is not None:
                            mol_img = Draw.MolToImage(mol, size=(300, 300))
                            st.image(mol_img, caption=f"Component {i+1}: {comp['smiles']}", use_container_width=True)
                            st.divider()
                
                with col2:
                    st.subheader("📈 시뮬레이션 결과")
                    
                    # 차트 테마 설정
                    chart_theme = st.radio("차트 테마", ["Dark", "Light"], horizontal=True, index=0)
                    theme_template = "plotly_dark" if chart_theme == "Dark" else "plotly_white"
                    line_color = "white" if chart_theme == "Dark" else "black"
                    label_color = "#00BFFF" if chart_theme == "Dark" else "blue"
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wavenumbers, y=transmittance,
                        mode='lines',
                        line=dict(color=line_color, width=1.5),
                        name='Simulated IR'
                    ))
                    
                    if show_labels:
                        for fg in all_identified_groups:
                            actual_y = np.interp(fg["wavenumber"], wavenumbers[::-1], transmittance[::-1])
                            fig.add_annotation(
                                x=fg["wavenumber"], y=actual_y,
                                text=fg["name"],
                                showarrow=True, arrowhead=1,
                                ax=0, ay=-40 if actual_y > 30 else 40,
                                font=dict(color=label_color, size=10),
                                bgcolor="rgba(0,0,0,0.5)" if chart_theme == "Dark" else "rgba(255,255,255,0.7)",
                                bordercolor=label_color, borderwidth=1
                            )
                    
                    fig.update_layout(
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Transmittance (%)",
                        xaxis=dict(range=[4000, 400], gridcolor='rgba(128,128,128,0.2)'),
                        yaxis=dict(range=[0, 105], gridcolor='rgba(128,128,128,0.2)'),
                        template=theme_template,
                        hovermode="x unified",
                        height=600,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 데이터 다운로드 버튼
                    csv_data = pd.DataFrame({"wavenumber": wavenumbers, "transmittance": transmittance}).to_csv(index=False)
                    st.download_button("결과 데이터(CSV) 다운로드", csv_data, "simulated_ir.csv", "text/csv")

        except Exception as e:
            st.error(f"시뮬레이션 중 오류 발생: {e}")
            import traceback
            st.code(traceback.format_exc())

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from ir_simulator import generate_ir_spectrum, optimize_mixture_ratios
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="고분자/믹스처 IR 스펙트럼 시뮬레이터", layout="wide")

st.title("고분자 및 혼합물 가상 IR 시뮬레이터 (ML 기반)")
st.markdown("SMILES 문자열과 배합 비율을 입력하여 혼합물(고분자/믹스처)의 가상 IR 스펙트럼을 예측합니다.")

# 입력 섹션 (데이터 에디터)
st.subheader("혼합물 구성 입력 (SMILES 및 배합비)")
st.info("표 아래의 빈 줄을 클릭하여 새로운 성분을 추가하거나, 기존 데이터를 수정할 수 있습니다.")

with st.expander("💡 중합도(n) 및 배합비 설정 가이드", expanded=False):
    st.markdown("""
    - **배합비(Ratio)**: 혼합물 내 해당 성분의 상대적 양입니다. (자동 정규화됨)
    - **중합도(n)**: 분자의 사슬 길이를 결정하는 지수입니다.
        - **n = 1**: **단량체(Monomer)**. 이중결합 등 모든 반응성 작용기가 보존됩니다.
        - **n > 1**: **고분자(Polymer)**. 값이 커질수록 주쇄(Backbone) 비율이 높아지며, 중합 가능한 결합이 포화(Saturation)됩니다.
        - **효과**: $n$이 증가할수록 사슬 경직도에 의한 **피크 시프트(Shift)**와 비정질 특성에 의한 **피크 브로드닝(Broadening)**이 물리적으로 반영됩니다.
    """)

if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame([{"SMILES": "CC(=C)C(=O)OCCCCCCCCCCOP(=O)(O)O", "배합비(Ratio)": 1.0, "중합도(n)": 1}])

edited_df = st.data_editor(st.session_state.input_df, num_rows="dynamic", use_container_width=True, key="main_editor")

col_btn, col_toggle = st.columns([1, 3])
with col_btn:
    generate_btn = st.button("IR 스펙트럼 생성")
with col_toggle:
    show_labels = st.toggle("작용기 라벨 표시 (파란 글씨)", value=True)

# --- 사이드바: 실험 데이터 기반 배합비 최적화 ---
st.sidebar.header("실험 데이터 기반 최적화")
st.sidebar.markdown("실측된 IR 데이터를 업로드하면 AI가 최적의 배합비를 계산합니다.")

uploaded_file = st.sidebar.file_uploader("실험 IR 데이터 업로드 (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        exp_data = pd.read_csv(uploaded_file)
        if len(exp_data.columns) < 2:
            st.sidebar.error("CSV 파일은 최소 2개의 열(Wavenumber, Transmittance)을 포함해야 합니다.")
        else:
            w_col = exp_data.columns[0]
            t_col = exp_data.columns[1]
            st.sidebar.success(f"데이터 로드 완료: {len(exp_data)} 포인트")
            
            if st.sidebar.button("최적 배합비 추정 시작"):
                # 현재 입력된 SMILES 목록 가져오기
                current_components = []
                for _, row in edited_df.iterrows():
                    s = str(row.get("SMILES", "")).strip()
                    if s:
                        current_components.append({"smiles": s, "ratio": 1.0})
                
                if not current_components:
                    st.sidebar.warning("먼저 구성 성분(SMILES)을 테이블에 입력해주세요.")
                else:
                    with st.spinner("AI 최적화 엔진 가동 중..."):
                        optimized_ratios = optimize_mixture_ratios(
                            current_components, 
                            exp_data[w_col].values, 
                            exp_data[t_col].values
                        )
                        
                        if optimized_ratios:
                            # 세션 상태 업데이트하여 테이블에 반영
                            new_data = []
                            for i, row in edited_df.iterrows():
                                s = str(row.get("SMILES", "")).strip()
                                if s:
                                    new_data.append({"SMILES": s, "배합비(Ratio)": optimized_ratios[len(new_data)]})
                            
                            st.session_state.input_df = pd.DataFrame(new_data)
                            st.sidebar.success("최적화 완료! 테이블에 반영되었습니다.")
                            st.rerun()
                        else:
                            st.sidebar.error("최적화 실패. 데이터를 확인해주세요.")
    except Exception as e:
        st.sidebar.error(f"파일 처리 오류: {e}")


if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False

if generate_btn:
    st.session_state.generate_clicked = True
    st.session_state.components = []
    for _, row in edited_df.iterrows():
        s = str(row.get("SMILES", "")).strip()
        r = float(row.get("배합비(Ratio)", 0))
        n = int(row.get("중합도(n)", 1))
        if s and r > 0:
            st.session_state.components.append({"smiles": s, "ratio": r, "n": n})

if st.session_state.generate_clicked:
    components = st.session_state.get("components", [])
            
    if not components:
        st.warning("최소 1개 이상의 유효한 SMILES 문자열과 양수(>0) 배합비를 입력해주세요.")
    else:
        try:
            with st.spinner("스펙트럼 생성 중..."):
                # ir_simulator의 멀티 모달 함수 사용
                wavenumbers, transmittance, all_identified_groups, mols = generate_ir_spectrum(components)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("분자 구조(2D)")
                    for mol, comp in zip(mols, components):
                        if mol is not None:
                            mol_img = Draw.MolToImage(mol, size=(300, 300))
                            st.image(mol_img, caption=f"{comp['smiles']} (Ratio: {comp['ratio']})", use_container_width=True)
                            st.divider()
                    
                with col2:
                    st.subheader("혼합물 예측 IR 스펙트럼")
                    # 결과 시각화
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(wavenumbers, transmittance, color='black', linewidth=1.5)
                    
                    # X축 역전
                    ax.set_xlim(4000, 400)
                    ax.set_ylim(0, 105)
                    
                    title_str = " + ".join([c['smiles'] for c in components])
                    ax.set_title(f"Simulated Mixture IR Spectrum: {title_str}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
                    ax.set_ylabel("Transmittance (%)", fontsize=12)
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    
                    # 식별된 작용기 텍스트 라벨 추가
                    if show_labels:
                        for fg in all_identified_groups:
                            ax.text(fg["wavenumber"], 105 - (fg["intensity"] * 100) - 5, fg["name"], 
                                     horizontalalignment='center', verticalalignment='bottom', 
                                     fontsize=10, color='blue', rotation=90)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                
                # 피크 정보 표로 출력
                st.subheader("주요 작용기 흡수 피크 정보 (배합비 반영)")
                st.dataframe([
                    {"SMILES": fg["smiles"], "작용기 구분": fg["name"], "진동 주파수 (cm⁻¹)": fg["wavenumber"], "합산 기여도(강도)": round(fg["intensity"], 4)}
                    for fg in all_identified_groups
                ], use_container_width=True)
                
        except ValueError as e:
            st.error(f"오류: {e}")
        except Exception as e:
            st.error(f"예기치 않은 오류가 발생했습니다: {e}")

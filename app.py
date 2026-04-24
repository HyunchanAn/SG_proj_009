import streamlit as st
import matplotlib.pyplot as plt
from ir_simulator import generate_ir_spectrum
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="가상 IR 스펙트럼 시뮬레이터", layout="wide")

st.title("가상 IR(Infrared Spectroscopy) 시뮬레이터")
st.markdown("SMILES 문자열을 입력하여 분자의 가상 IR 스펙트럼을 시뮬레이션하고 시각화합니다.")

# 입력 섹션
smiles_input = st.text_input("SMILES 문자열 입력", value="CC(=O)Oc1ccccc1C(=O)O", help="예: 아스피린은 CC(=O)Oc1ccccc1C(=O)O")

if st.button("IR 스펙트럼 생성"):
    if not smiles_input.strip():
        st.warning("SMILES 문자열을 입력해주세요.")
    else:
        try:
            with st.spinner("스펙트럼 생성 중..."):
                mol = Chem.MolFromSmiles(smiles_input)
                if mol is None:
                    raise ValueError("유효하지 않은 SMILES 문자열입니다.")
                mol_img = Draw.MolToImage(mol, size=(300, 300))

                # ir_simulator의 함수 재사용
                wavenumbers, transmittance, identified_groups = generate_ir_spectrum(smiles_input)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("분자 구조(2D)")
                    st.image(mol_img, use_container_width=True)
                    
                with col2:
                    st.subheader("IR 스펙트럼")
                    # 결과 시각화
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(wavenumbers, transmittance, color='black', linewidth=1.5)
                    
                    # X축 역전
                    ax.set_xlim(4000, 400)
                    ax.set_ylim(0, 105)
                    
                    ax.set_title(f"Simulated IR Spectrum: {smiles_input}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
                    ax.set_ylabel("Transmittance (%)", fontsize=12)
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    
                    # 식별된 작용기 텍스트 라벨 추가
                    for fg in identified_groups:
                        ax.text(fg["wavenumber"], 105 - (fg["intensity"] * 100) - 5, fg["name"], 
                                 horizontalalignment='center', verticalalignment='bottom', 
                                 fontsize=10, color='blue', rotation=90)
                    
                    fig.tight_layout()
                    
                    # Streamlit에 그래프 출력
                    st.pyplot(fig)
                
                # 식별된 작용기 정보 표로 출력
                st.subheader("식별된 작용기(Functional Groups) 정보")
                st.dataframe([
                    {"작용기": fg["name"], "중심 파수 (cm⁻¹)": fg["wavenumber"], "상대적 흡수 강도": fg["intensity"]}
                    for fg in identified_groups
                ])
                
        except ValueError as e:
            st.error(f"오류: {e}")
        except Exception as e:
            st.error(f"예기치 않은 오류가 발생했습니다: {e}")

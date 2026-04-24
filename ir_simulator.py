import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem

def identify_functional_groups(mol):
    """
    RDKit을 활용하여 분자 내 주요 작용기를 식별합니다.
    SMARTS 패턴을 사용하여 매칭합니다.
    """
    # (이름, SMARTS, 중심 파수, 강도, 너비)
    fg_patterns = [
        # 1. O-H 및 N-H (3600-3000)
        ("Carboxylic Acid O-H", "[CX3](=O)[OX2H1]", 3000, 0.85, 300),
        ("Alcohol/Phenol O-H", "[OX2H1;!$([OX2H1][CX3]=O)]", 3350, 0.8, 120),
        ("Primary Amine N-H", "[NX3H2]", 3400, 0.4, 40),
        ("Primary Amine N-H", "[NX3H2]", 3300, 0.4, 40),
        ("Secondary Amine N-H", "[NX3H1]", 3350, 0.3, 40),
        
        # 2. C-H (3300-2700)
        ("Alkyne C-H", "[CX2H1]", 3300, 0.7, 30),
        ("Aromatic/Alkene C-H", "[cX3H1],[CX3H1]", 3050, 0.4, 30),
        ("Alkane C-H (asym)", "[CX4H2],[CX4H3]", 2950, 0.6, 40),
        ("Alkane C-H (sym)", "[CX4H2],[CX4H3]", 2850, 0.5, 40),
        ("Aldehyde C-H", "[CX3H1](=O)", 2820, 0.4, 30),
        ("Aldehyde C-H", "[CX3H1](=O)", 2720, 0.4, 30),
        
        # 3. 삼중 결합 (2300-2100)
        ("Nitrile C#N", "[CX2]#N", 2250, 0.6, 20),
        ("Alkyne C#C", "[CX2]#[CX2]", 2150, 0.3, 20),
        
        # 4. 카보닐 C=O (1850-1650)
        ("Ester C=O", "[#6][CX3](=O)[OX2][#6]", 1735, 0.9, 25),
        ("Aldehyde C=O", "[CX3H1](=O)", 1725, 0.9, 25),
        ("Ketone C=O", "[#6][CX3](=O)[#6]", 1715, 0.9, 25),
        ("Carboxylic Acid C=O", "[CX3](=O)[OX2H1]", 1710, 0.95, 30),
        ("Amide C=O", "[CX3](=O)[NX3]", 1650, 0.85, 30),
        
        # 5. 이중 결합 및 방향족 고리 (1650-1450)
        ("Alkene C=C", "[CX3]=[CX3]", 1640, 0.4, 20),
        ("Aromatic C=C", "c1ccccc1", 1600, 0.5, 20),
        ("Aromatic C=C", "c1ccccc1", 1500, 0.6, 20),
        ("Aromatic C=C", "c1ccccc1", 1450, 0.5, 20),
        
        # 6. C-H 굽힘 (Bending) (1450-1350)
        ("Alkane C-H bend", "[CX4H2],[CX4H3]", 1465, 0.5, 30),
        ("CH3 bend (Umbrella)", "[CX4H3]", 1375, 0.4, 25),
        
        # 7. C-O 단일 결합 (1300-1000)
        ("Ester/Ether C-O", "[#6]-O-[#6]", 1200, 0.7, 40),
        ("Alcohol/Ether C-O", "[#6]-O", 1050, 0.7, 40),
        
        # 8. 할로겐 화합물 (800-500)
        ("C-Cl", "[#6]-Cl", 750, 0.6, 40),
        ("C-Br", "[#6]-Br", 600, 0.5, 40)
    ]
    
    identified_groups = []
    for name, smarts, wavenumber, intensity, width in fg_patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            identified_groups.append({
                "name": name,
                "wavenumber": wavenumber,
                "intensity": intensity,
                "width": width
            })
            
    return identified_groups

def generate_ir_spectrum(smiles):
    """
    SMILES 문자열을 분석하여 가상 IR 스펙트럼 데이터를 생성합니다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("유효하지 않은 SMILES 입력입니다. 다시 확인해주세요.")
        
    wavenumbers = np.linspace(4000, 400, 3600)
    transmittance = np.ones_like(wavenumbers)
    
    identified_groups = identify_functional_groups(mol)
    
    # 1. 작용기 주요 피크 적용 (가우시안)
    for fg in identified_groups:
        absorption = fg["intensity"] * np.exp(-0.5 * ((wavenumbers - fg["wavenumber"]) / fg["width"])**2)
        transmittance -= absorption
        
    # 2. 지문 영역(Fingerprint Region, 1500~400) 시뮬레이션
    # 분자의 중원자(Heavy atom) 수에 비례하여 무작위 피크 생성 (SMILES 해시 기반 시드 고정)
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    np.random.seed(sum([ord(c) for c in smiles]))
    num_fingerprint_peaks = min(num_heavy_atoms * 2, 40) 
    
    for _ in range(num_fingerprint_peaks):
        center = np.random.uniform(400, 1450)
        intensity = np.random.uniform(0.05, 0.35)
        width = np.random.uniform(5, 20)
        absorption = intensity * np.exp(-0.5 * ((wavenumbers - center) / width)**2)
        transmittance -= absorption

    # 3. 베이스라인 기울기(Baseline drift) 및 기기 노이즈(Noise) 추가
    # 저파수 쪽으로 갈수록 살짝 내려가는 경향 (0.0 ~ 0.05)
    baseline_drift = np.linspace(0.0, 0.05, len(wavenumbers)) 
    # 고주파 미세 노이즈
    noise = np.random.normal(0, 0.005, len(wavenumbers))
    
    transmittance = transmittance - baseline_drift + noise
    transmittance = np.clip(transmittance, 0.0, 1.0)
    
    return wavenumbers, transmittance * 100, identified_groups

def plot_ir_spectrum(smiles, save_path=None):
    """
    데이터를 바탕으로 Matplotlib 그래프를 생성하고 화면에 출력하는 함수.
    """
    try:
        wavenumbers, transmittance, identified_groups = generate_ir_spectrum(smiles)
    except ValueError as e:
        print(f"오류: {e}")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, transmittance, color='black', linewidth=1.5)
    
    plt.xlim(4000, 400)
    plt.ylim(0, 105)
    
    plt.title(f"Simulated IR Spectrum for SMILES: {smiles}", fontsize=14, fontweight='bold')
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
    plt.ylabel("Transmittance (%)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    for fg in identified_groups:
        plt.text(fg["wavenumber"], 105 - (fg["intensity"] * 100) - 5, fg["name"], 
                 horizontalalignment='center', verticalalignment='bottom', 
                 fontsize=10, color='blue', rotation=90)
                 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"그래프가 {save_path} 에 저장되었습니다.")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        smiles_input = sys.argv[1]
    else:
        smiles_input = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin
        
    print(f"분석할 SMILES: {smiles_input}")
    plot_ir_spectrum(smiles_input, save_path="sample_ir.png")

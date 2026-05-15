import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os
import sys

# 양자화학 엔진 임포트
try:
    from qc_engine import calculate_ir_qc, peaks_to_spectrum
except ImportError:
    calculate_ir_qc = None

# QC 결과 캐시 (SMILES -> Peaks)
QC_CACHE = {}

# ==========================================
# [기획 2] 머신러닝(GNN) 기반 IR 예측 모델 아키텍처
# ==========================================
class IRGraphNeuralNetwork(nn.Module):
    """
    SMILES 기반 분자 그래프를 입력받아 IR 스펙트럼(3600차원 벡터)을 
    예측하는 PyTorch Geometric 기반의 GNN 클래스입니다.
    """
    def __init__(self, node_dim=11, hidden_dim=64, output_dim=3600):
        super(IRGraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, 256)
        
        self.hidden = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.output = nn.Linear(512, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch=None):
        # x: 노드 특성, edge_index: 그래프 연결 구조, batch: 배치 정보
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        
        # 그래프 전체를 나타내는 단일 벡터로 풀링 (Mean Pooling)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        
        x = self.hidden(x)
        out = self.output(x)
        return self.sigmoid(out)

_ml_model = None

def load_ml_model():
    """사전 학습된 가중치를 로드하고 가속 장치(MPS/CPU)를 반환하는 함수"""
    global _ml_model
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    if _ml_model is None:
        _ml_model = IRGraphNeuralNetwork()
        weight_path = 'weights/ir_gnn_v1.pt'
        if os.path.exists(weight_path):
            _ml_model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            print(f"✅ 학습된 가중치를 로드했습니다 ({device}): {weight_path}")
        _ml_model.to(device)
        _ml_model.eval()
    return _ml_model, device

def smiles_to_graph_features(mol, ratio=1.0):
    """RDKit 분자 객체를 배합비(Ratio)가 포함된 PyTorch Geometric Data 객체로 변환합니다."""
    features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetImplicitValence(),
            atom.GetIsAromatic() * 1.0,
            atom.GetMass(),
            0.0, 0.0, 0.0, 0.0, 0.0, float(ratio) # 11번째 특성으로 배합비 주입
        ]
        features.append(feat)
    
    x = torch.tensor(features, dtype=torch.float32)
    if x.size(0) == 0:
        x = torch.zeros((1, 11), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i]) # 양방향 그래프
        
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index)

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

def get_heuristic_absorption(smiles):
    """
    SMILES로부터 작용기 기반 순수 규칙(Heuristic) 흡광도를 계산합니다.
    (머신러닝 Knowledge Distillation을 위한 정답(Target) 라벨 생성용)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    wavenumbers = np.linspace(4000, 400, 3600)
    absorption = np.zeros_like(wavenumbers)
    
    identified_groups = identify_functional_groups(mol)
    for fg in identified_groups:
        peak = fg["intensity"] * np.exp(-0.5 * ((wavenumbers - fg["wavenumber"]) / fg["width"])**2)
        absorption += peak
        
    return absorption

def saturate_monomer(smiles):
    """
    RDKit을 사용하여 중합 가능한 이중/삼중 결합(Vinyl 등)만 선택적으로 포화시킵니다.
    C=O 등 측쇄 작용기는 보존합니다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    
    # 중합 가능한 탄소-탄소 불포화 결합 패턴 (이중결합 또는 삼중결합)
    patt1 = Chem.MolFromSmarts("[C;!a]=[C;!a]")
    patt2 = Chem.MolFromSmarts("[C;!a]#[C;!a]")
    
    matches = []
    if patt1: matches.extend(mol.GetSubstructMatches(patt1))
    if patt2: matches.extend(mol.GetSubstructMatches(patt2))
    
    if not matches:
        return smiles
        
    ed_mol = Chem.EditableMol(mol)
    for match in matches:
        idx1, idx2 = match
        bond = mol.GetBondBetweenAtoms(idx1, idx2)
        if bond:
            ed_mol.RemoveBond(idx1, idx2)
            ed_mol.AddBond(idx1, idx2, Chem.rdchem.BondType.SINGLE)
            
    try:
        saturated_mol = ed_mol.GetMol()
        return Chem.MolToSmiles(saturated_mol)
    except:
        return smiles

def apply_physical_chain_effects(prediction, n, wavenumbers):
    """
    중합도(n)에 따른 물리적 효과(Shift, Broadening)를 예측된 스펙트럼에 적용합니다.
    """
    if n <= 1:
        return prediction
        
    # 1. Peak Shift (최대 5cm-1 이동)
    shift_val = 5.0 * (1.0 - 1.0/n)
    shift_idx = int(shift_val / abs(wavenumbers[1] - wavenumbers[0]))
    
    if shift_idx > 0:
        new_pred = np.zeros_like(prediction)
        new_pred[shift_idx:] = prediction[:-shift_idx]
        prediction = new_pred
        
    # 2. Peak Broadening (가우시안 필터)
    if n > 5:
        sigma = 1.2 * (1.0 - 1.0/n)
        from scipy.ndimage import gaussian_filter1d
        prediction = gaussian_filter1d(prediction, sigma=sigma)
        
    return prediction

def apply_hydrogen_bonding_effects(prediction, mol, ratio, n, wavenumbers):
    """
    분자 농도(ratio)와 중합도(n)에 따른 수소 결합(Hydrogen Bonding) 효과를 모사합니다.
    O-H, N-H 피크의 Shift 및 Broadening을 처리합니다.
    """
    # 수소 결합 가능성 확인 (O-H 또는 N-H 존재 여부)
    has_oh = mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2H]"))
    has_nh = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3H]"))
    
    if not (has_oh or has_nh):
        return prediction
        
    h_bond_intensity = ratio * (1.0 - 0.2/n)
    
    if h_bond_intensity < 0.05:
        return prediction
        
    # O-H/N-H 영역 (약 3000 ~ 3700 cm-1) - 영역을 조금 더 넓힘
    region_mask = (wavenumbers >= 3000) & (wavenumbers <= 3750)
    if not np.any(region_mask):
        return prediction
        
    oh_region = prediction[region_mask]
    
    # 1. 강력한 Broadening 적용 (수소 결합의 핵심적 특성)
    from scipy.ndimage import gaussian_filter1d
    # Sigma를 대폭 강화 (10.0 -> 60.0)
    sigma = 60.0 * h_bond_intensity 
    broadened_oh = gaussian_filter1d(oh_region, sigma=sigma)
    
    # 2. 강한 Shift 적용 (Free -> Bonded 이동 폭 확대)
    # 수소 결합이 강할수록 피크가 더 낮은 파수(오른쪽)로 많이 이동함
    shift_amount = int(120 * h_bond_intensity) # 최대 120포인트 이동
    shifted_oh = np.zeros_like(broadened_oh)
    if shift_amount > 0:
        shifted_oh[shift_amount:] = broadened_oh[:-shift_amount]
        # 이동 후 빈 자리는 주변부로 채움
        shifted_oh[:shift_amount] = broadened_oh[0] * 0.1 
    else:
        shifted_oh = broadened_oh
        
    # 수소 결합 피크는 면적이 넓어지면서 시각적으로 더 깊게 느껴지도록 보정
    shifted_oh = shifted_oh * (1.0 + 0.2 * h_bond_intensity)
        
    prediction[region_mask] = shifted_oh
    return prediction

def generate_ir_spectrum(components, use_qc=False):
    """
    여러 화합물(SMILES)과 그 배합비를 입력받아 가상 IR 스펙트럼 데이터를 생성합니다.
    use_qc=True일 경우 양자화학 계산(xTB)을 통해 정밀도를 높입니다.
    """
    wavenumbers = np.linspace(4000, 400, 3600)
    raw_prediction = np.zeros(3600)
    all_identified_groups = []
    mols = []
    
    # ML 모델 로드
    model, device = load_ml_model()
    
    # 총 배합비 합계 계산 (정규화를 위해)
    total_ratio = sum([comp.get("ratio", 0) for comp in components])
    if total_ratio == 0: total_ratio = 1.0
        
    for comp in components:
        smiles = comp.get("smiles", "").strip()
        ratio = comp.get("ratio", 0) / total_ratio
        n = comp.get("n", 1)
        
        if not smiles or ratio <= 0: continue
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        mols.append(mol)
            
        # 1) 단량체(Monomer) 예측
        data_monomer = smiles_to_graph_features(mol, ratio=ratio).to(device)
        with torch.no_grad():
            pred_monomer = model(data_monomer.x, data_monomer.edge_index, torch.zeros(data_monomer.x.size(0), dtype=torch.long, device=device)).cpu().numpy()[0]
            
        # QC 정밀 계산 연동 (단량체 기준)
        if use_qc and 'calculate_ir_qc' in globals():
            if smiles not in QC_CACHE:
                print(f"[{smiles}] 신규 양자화학 계산 수행 중 (GFN2-xTB)...")
                QC_CACHE[smiles] = calculate_ir_qc(smiles)
            
            qc_peaks = QC_CACHE[smiles]
            qc_pred = peaks_to_spectrum(qc_peaks, wavenumbers)
            
            # GNN과 QC 결과 합성 (QC 비중 70%)
            pred_monomer = (pred_monomer * 0.3) + (qc_pred * 0.7)

        # 2) 주쇄(Backbone) 성분 예측 (선택적 포화 구조 기반)
        sat_smiles = saturate_monomer(smiles)
        sat_mol = Chem.MolFromSmiles(sat_smiles)
        if sat_mol and sat_smiles != smiles:
            data_sat = smiles_to_graph_features(sat_mol, ratio=ratio).to(device)
            with torch.no_grad():
                pred_backbone = model(data_sat.x, data_sat.edge_index, torch.zeros(data_sat.x.size(0), dtype=torch.long, device=device)).cpu().numpy()[0]
        else:
            pred_backbone = pred_monomer
            
        # 3) 물리적 효과 적용 및 합성
        # n=1이면 100% monomer, n이 커질수록 backbone 비중 증가
        weight_monomer = 1.0 / n
        weight_backbone = 1.0 - weight_monomer
        
        combined_pred = (pred_monomer * weight_monomer) + (pred_backbone * weight_backbone)
        
        # 중합도에 따른 Shift & Broadening 적용
        final_comp_pred = apply_physical_chain_effects(combined_pred, n, wavenumbers)
        
        # 수소 결합(Hydrogen Bonding) 효과 적용
        final_comp_pred = apply_hydrogen_bonding_effects(final_comp_pred, mol, ratio, n, wavenumbers)
        
        raw_prediction += final_comp_pred * ratio

        # UI용 작용기 식별 (Monomer 기준)
        identified_groups = identify_functional_groups(mol)
        for fg in identified_groups:
            all_identified_groups.append({
                "smiles": smiles,
                "name": fg["name"],
                "wavenumber": fg["wavenumber"],
                "intensity": fg["intensity"] * ratio,
                "width": fg["width"]
            })
            
    # 최종 렌더링 (Beer-Lambert Law 적용: T = exp(-A))
    # raw_prediction을 Absorbance로 간주하여 지수 매핑
    transmittance = np.exp(-raw_prediction * 2.5) # 감도 조절을 위해 2.5배 스케일링
    
    # Baseline Drift (약간의 우하향 경향성 추가)
    baseline_drift = np.linspace(0.0, 0.05, len(wavenumbers)) 
    transmittance = transmittance * (1.0 - baseline_drift)
    
    transmittance = np.clip(transmittance, 0.01, 1.0) # 0%에 딱 붙지 않게 최소값 유지
    
    return wavenumbers, transmittance * 100, all_identified_groups, mols

def plot_ir_spectrum(components, save_path=None):
    """
    데이터를 바탕으로 Matplotlib 그래프를 생성하고 화면에 출력하는 함수.
    """
    try:
        wavenumbers, transmittance, all_identified_groups, _ = generate_ir_spectrum(components)
    except ValueError as e:
        print(f"오류: {e}")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, transmittance, color='black', linewidth=1.5)
    
    plt.xlim(4000, 400)
    plt.ylim(0, 105)
    
    title_str = " + ".join([c['smiles'] for c in components])
    plt.title(f"Simulated Polymer/Mixture IR Spectrum: {title_str}", fontsize=14, fontweight='bold')
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
    plt.ylabel("Transmittance (%)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    for fg in all_identified_groups:
        plt.text(fg["wavenumber"], 105 - (fg["intensity"] * 100) - 5, fg["name"], 
                 horizontalalignment='center', verticalalignment='bottom', 
                 fontsize=10, color='blue', rotation=90)
                 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"그래프가 {save_path} 에 저장되었습니다.")
    else:
        plt.show()

def optimize_mixture_ratios(components, target_wavenumbers, target_transmittance):
    """
    실험 데이터(target_transmittance)와 가장 유사한 스펙트럼을 만드는 최적의 배합비를 계산합니다.
    """
    model = load_ml_model()
    if model is None:
        return None
        
    # 유효한 분자 그래프 추출
    data_list = []
    valid_indices = []
    for i, comp in enumerate(components):
        smiles = comp.get("smiles", "").strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            data_list.append(smiles_to_graph_features(mol))
            valid_indices.append(i)
            
    if not data_list:
        return None
        
    # 각 분자의 개별 예측값 미리 계산
    loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        individual_predictions = model(batch.x, batch.edge_index, batch.batch).numpy() # [N, 3600]
        
    # 실험 데이터 전처리: 3600차원 보간 및 흡광도로 변환
    target_wavenumbers = np.array(target_wavenumbers)
    target_transmittance = np.array(target_transmittance) / 100.0 # 0~1 스케일
    
    if target_wavenumbers[0] > target_wavenumbers[-1]:
        target_wavenumbers = target_wavenumbers[::-1]
        target_transmittance = target_transmittance[::-1]
        
    f_interp = interp1d(target_wavenumbers, target_transmittance, bounds_error=False, fill_value=1.0)
    # 우리 모델의 파수 그리드 (4000 to 400)
    model_wavenumbers = np.linspace(4000, 400, 3600)
    interp_transmittance = f_interp(model_wavenumbers)
    
    # T to A 변환 (최소값 클리핑)
    target_absorption = 1.0 - np.clip(interp_transmittance, 0.0, 1.0)
    
    # 최적화 함수 정의 (MSE 최소화)
    def objective(ratios):
        # 비율 정규화 (합이 1이 되도록)
        r_sum = np.sum(ratios)
        if r_sum == 0: return 1e6
        norm_ratios = ratios / r_sum
        
        # 가중합 예측
        pred_absorption = np.zeros(3600)
        for i, r in enumerate(norm_ratios):
            pred_absorption += individual_predictions[i] * r
        # AI 모델 렌더링 스케일(1.5) 반영
        pred_absorption *= 1.5
        # 손실 계산 (MSE)
        return np.mean((pred_absorption - target_absorption)**2)
        
    # 초기값 및 제약 조건 (0~1 사이, 합계 1)
    n = len(data_list)
    initial_guess = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    
    res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        optimized_ratios = res.x / np.sum(res.x) # 정규화
        full_ratios = [0.0] * len(components)
        for i, idx in enumerate(valid_indices):
            full_ratios[idx] = round(float(optimized_ratios[i]), 4)
        return full_ratios
    else:
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        smiles_input = sys.argv[1]
        components = [{"smiles": smiles_input, "ratio": 1.0}]
    else:
        components = [{"smiles": "CC(=C)C(=O)OCCCCCCCCCCOP(=O)(O)O", "ratio": 1.0, "n": 1}] # 10-MDP
        
    print(f"분석할 혼합물 구성: {components}")
    plot_ir_spectrum(components, save_path="sample_ir.png")

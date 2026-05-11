import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from scipy.optimize import minimize
from scipy.interpolate import interp1d

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
    """사전 학습된 가중치를 로드하는 싱글톤 함수"""
    global _ml_model
    if _ml_model is None:
        _ml_model = IRGraphNeuralNetwork()
        weight_path = 'weights/ir_gnn_v1.pt'
        import os
        if os.path.exists(weight_path):
            _ml_model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True))
            print(f"✅ 학습된 가중치를 로드했습니다: {weight_path}")
        _ml_model.eval()
    return _ml_model

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

def generate_ir_spectrum(components):
    """
    여러 화합물(SMILES)과 그 배합비를 입력받아 가상 IR 스펙트럼 데이터를 생성합니다.
    고분자 효과(Polymer Chain Effects)가 물리적으로 반영됩니다.
    """
    wavenumbers = np.linspace(4000, 400, 3600)
    raw_prediction = np.zeros_like(wavenumbers)
    
    all_identified_groups = []
    mols = []
    
    # 총 배합비 합계 계산 (정규화를 위해)
    total_ratio = sum([comp.get("ratio", 0) for comp in components])
    if total_ratio == 0: total_ratio = 1.0
        
    model = load_ml_model()
    
    for comp in components:
        smiles = comp.get("smiles", "").strip()
        ratio = comp.get("ratio", 0) / total_ratio
        n = comp.get("n", 1)
        
        if not smiles or ratio <= 0: continue
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: continue
        mols.append(mol)
            
        # 1) 말단기/잔량 성분 예측 (Monomer 구조 기반)
        data_monomer = smiles_to_graph_features(mol, ratio=ratio)
        with torch.no_grad():
            pred_monomer = model(data_monomer.x, data_monomer.edge_index).numpy()[0]
            
        # 2) 주쇄(Backbone) 성분 예측 (선택적 포화 구조 기반)
        sat_smiles = saturate_monomer(smiles)
        sat_mol = Chem.MolFromSmiles(sat_smiles)
        if sat_mol and sat_smiles != smiles:
            data_sat = smiles_to_graph_features(sat_mol, ratio=ratio)
            with torch.no_grad():
                pred_backbone = model(data_sat.x, data_sat.edge_index).numpy()[0]
        else:
            pred_backbone = pred_monomer

        # 3) 물리적 효과 적용 및 합성
        # n=1이면 100% monomer, n이 커질수록 backbone 비중 증가
        weight_monomer = 1.0 / n
        weight_backbone = 1.0 - weight_monomer
        
        combined_pred = (pred_monomer * weight_monomer) + (pred_backbone * weight_backbone)
        
        # 중합도에 따른 Shift & Broadening 적용
        final_comp_pred = apply_physical_chain_effects(combined_pred, n, wavenumbers)
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
            
    # 최종 렌더링 스케일링
    total_absorption = raw_prediction * 1.5
    transmittance = 1.0 - total_absorption
    
    baseline_drift = np.linspace(0.0, 0.05, len(wavenumbers)) 
    transmittance = transmittance - baseline_drift
    transmittance = np.clip(transmittance, 0.0, 1.0)
    
    return wavenumbers, transmittance * 100, all_identified_groups, mols

    # 3. 투과도(Transmittance) 변환 및 베이스라인/노이즈 적용
    # 흡광도(A)가 높아질수록 투과도(T)는 낮아지는 직관적 근사: T = 1 - A
    transmittance = 1.0 - total_absorption
    
    baseline_drift = np.linspace(0.0, 0.05, len(wavenumbers)) 
    transmittance = transmittance - baseline_drift
    transmittance = np.clip(transmittance, 0.0, 1.0)
    
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

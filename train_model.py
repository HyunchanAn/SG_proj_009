import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
from rdkit import Chem
import time

from ir_simulator import IRGraphNeuralNetwork, smiles_to_graph_features
import numpy as np

class RealIRDataset(Dataset):
    def __init__(self, csv_path, max_samples=1000):
        print(f"실제 실험 데이터셋 로딩 중: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'ir_spectrum' not in df.columns:
            raise ValueError("데이터셋에 'ir_spectrum' 컬럼이 없습니다. prepare_real_data.py를 먼저 실행하세요.")
            
        self.valid_data = []
        
        print("실제 스펙트럼 데이터 파싱 및 그래프 텐서 변환 중...")
        start_time = time.time()
        
        for idx, row in df.head(max_samples).iterrows():
            smiles = str(row['smiles'])
            spectrum_str = str(row['ir_spectrum'])
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            
            graph_data = smiles_to_graph_features(mol)
            
            try:
                target_ir = np.array([float(x) for x in spectrum_str.split(',')])
                if len(target_ir) != 3600:
                    continue
            except Exception:
                continue
                
            target_tensor = torch.tensor(target_ir, dtype=torch.float32)
            graph_data.y = target_tensor.unsqueeze(0)
            self.valid_data.append(graph_data)
                
        print(f"파싱 완료! 유효한 샘플 수: {len(self.valid_data)} (소요 시간: {time.time() - start_time:.2f}초)")
        
    def __len__(self):
        return len(self.valid_data)
        
    def __getitem__(self, idx):
        return self.valid_data[idx]

def train():
    print("="*50)
    print("PyTorch 하드웨어 가속 상태 점검")
    print("="*50)
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon (M1/M2/M3) MPS 가속이 활성화되었습니다.")
        device = torch.device("mps")
    else:
        print("⚠️ MPS를 사용할 수 없습니다. CPU 모드로 작동합니다.")
        device = torch.device("cpu")
        
    # 데이터셋 준비 (배치 사이즈 32)
    dataset_path = "datasets/real_ir_data.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ {dataset_path} 파일을 찾을 수 없습니다. python prepare_real_data.py 를 먼저 실행하여 데이터셋을 생성해주세요.")
        return
        
    dataset = RealIRDataset(dataset_path, max_samples=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 모델, 손실함수, 옵티마이저 초기화
    model = IRGraphNeuralNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 10
    print("="*50)
    print("PyTorch 모델 학습 시작 (Knowledge Distillation)")
    print("="*50)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, data in enumerate(dataloader):
            # GPU/MPS 메모리로 이동
            data = data.to(device)
            
            # 순전파
            optimizer.zero_grad()
            predictions = model(data.x, data.edge_index, data.batch)
            
            # 예측값과 타겟의 손실(오차) 계산
            loss = criterion(predictions, data.y)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.6f} | 소요 시간: {time.time() - start_time:.2f}초")
        
    # 모델 저장 디렉토리 생성
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/ir_gnn_v1.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 학습 완료! 모델 가중치가 저장되었습니다: {save_path}")
    print("추후 ir_simulator.py의 load_ml_model()에서 이 모델을 로드할 수 있습니다.")

if __name__ == "__main__":
    train()

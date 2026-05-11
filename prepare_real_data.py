import pandas as pd
import numpy as np
import os

def create_mock_real_dataset(output_path="datasets/real_ir_data.csv", num_samples=500):
    """
    NIST/SDBS 등에서 파싱해 올 실제 스펙트럼 데이터의 형식을 모방하여
    더미 데이터셋을 생성합니다. (파이프라인 검증 및 GNN 파인튜닝 테스트용)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 임의의 화합물 SMILES 목록
    sample_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O", "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O",
        "CC(=O)C", "CCN", "C1CCCCC1", "c1cc(O)ccc1", "C#C"
    ]
    
    data = []
    print("가상 '실제 실험 데이터' 형식 생성 중...")
    for i in range(num_samples):
        smiles = sample_smiles[i % len(sample_smiles)]
        
        # 실제 데이터인 척 하는 3600개의 랜덤 흡광도 값 (4000 ~ 400 cm^-1)
        # 실제 환경에서는 이 배열에 NIST/SDBS 측정값이 들어갑니다.
        mock_spectrum = np.random.rand(3600).astype(np.float32)
        
        # 배열을 쉼표로 구분된 문자열로 저장하여 용량 절약 및 로딩 최적화
        spectrum_str = ",".join([f"{x:.4f}" for x in mock_spectrum])
        data.append({"smiles": smiles, "ir_spectrum": spectrum_str})
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ 파이프라인 테스트용 데이터셋이 생성되었습니다: {output_path}")
    print("👉 향후 스크래핑한 SDBS/NIST 실제 데이터를 이 파일 형식(smiles, ir_spectrum)으로 맞춰 덮어쓰시면 전체 파이프라인이 즉시 호환됩니다.")

if __name__ == "__main__":
    create_mock_real_dataset()

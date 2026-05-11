import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from scipy.interpolate import interp1d
import time

def process_hf_dataset():
    print("Hugging Face Hub에서 대규모 오픈소스 IR 스펙트럼 데이터셋을 다운로드합니다...")
    print("Dataset: Lamblador/IRSpectra (~10,000 molecules)")
    
    # 데이터셋 로드
    dataset = load_dataset('Lamblador/IRSpectra', split='train')
    
    output_path = "datasets/real_ir_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_wavenumbers = np.linspace(4000, 400, 3600)
    data_list = []
    
    print(f"총 {len(dataset)}개의 화합물 스펙트럼 보간 및 변환 중... (M2 Pro CPU 활용)")
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        smiles = item['smiles']
        x = np.array(item['spectrum_processed_x'])
        y = np.array(item['spectrum_processed_y'])
        
        if len(x) == 0 or len(y) == 0:
            continue
            
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]
            
        # 1800 차원을 우리 모델 스펙(3600 차원)으로 보간
        f_interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
        target_spectrum = f_interp(target_wavenumbers)
        
        # 정규화
        max_val = np.max(target_spectrum)
        if max_val > 0:
            target_spectrum = target_spectrum / max_val
            
        spectrum_str = ",".join([f"{val:.4f}" for val in target_spectrum])
        data_list.append({"smiles": smiles, "ir_spectrum": spectrum_str})
        
        if (i+1) % 2000 == 0:
            print(f"  -> {i+1} / {len(dataset)} 개 처리 완료...")
            
    df = pd.DataFrame(data_list)
    df.to_csv(output_path, index=False)
    print(f"\n✅ 데이터셋 준비 완료! 소요 시간: {time.time() - start_time:.2f}초")
    print(f"총 {len(df)}개의 유니크 분자 데이터가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    process_hf_dataset()

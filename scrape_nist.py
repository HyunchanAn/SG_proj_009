import os
import requests
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import time
from jcamp import jcamp_read # pip install jcamp

# NIST WebBook URL Format
NIST_URL_TEMPLATE = "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C{cas}&Index=1&Type=IR"

# Target molecules
MOLECULES = [
    {"name": "Aspirin", "cas": "50782", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"name": "Ethanol", "cas": "64175", "smiles": "CCO"},
    {"name": "Acetic Acid", "cas": "64197", "smiles": "CC(=O)O"},
    {"name": "Benzene", "cas": "71432", "smiles": "c1ccccc1"},
    {"name": "Acetone", "cas": "67641", "smiles": "CC(=O)C"},
    {"name": "Isopropanol", "cas": "67630", "smiles": "CC(C)O"},
    {"name": "Toluene", "cas": "108883", "smiles": "Cc1ccccc1"},
    {"name": "Phenol", "cas": "108952", "smiles": "Oc1ccccc1"},
    {"name": "Methanol", "cas": "67561", "smiles": "CO"},
    {"name": "Ethyl Acetate", "cas": "141786", "smiles": "CCOC(C)=O"}
]

def download_and_process():
    os.makedirs("datasets", exist_ok=True)
    target_wavenumbers = np.linspace(4000, 400, 3600)
    data = []
    
    print("NIST Chemistry WebBook에서 실제 IR 스펙트럼 데이터를 스크래핑합니다...")
    
    for mol in MOLECULES:
        cas = mol["cas"]
        url = NIST_URL_TEMPLATE.format(cas=cas)
        print(f"[{mol['name']}] 다운로드 중: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200 or "##TITLE=" not in response.text:
                print(f"  -> 다운로드 실패 또는 데이터 없음")
                continue
                
            # 임시 파일로 저장 후 jcamp_read로 파싱
            temp_file = f"temp_{cas}.jdx"
            with open(temp_file, "w") as f:
                f.write(response.text)
                
            jcamp_dict = jcamp_read(temp_file)
            os.remove(temp_file)
            
            x = jcamp_dict['x'] # Wavenumbers
            y = jcamp_dict['y'] # Absorbance or Transmittance
            
            # y unit check (A or T)
            y_units = jcamp_dict.get('yunits', '').lower()
            if 'transmit' in y_units:
                y_t = np.clip(y, 0.001, 100.0)
                if np.max(y_t) > 2.0: # typically 0-100%
                    y_t = y_t / 100.0
                y = -np.log10(y_t)
            
            # Interpolate to 4000-400 grid
            if x[0] > x[-1]:
                x = x[::-1]
                y = y[::-1]
                
            f_interp = interp1d(x, y, bounds_error=False, fill_value=0.0)
            target_spectrum = f_interp(target_wavenumbers)
            
            # Normalize to max 1.0 to fit our model's output range (Sigmoid)
            max_val = np.max(target_spectrum)
            if max_val > 0:
                target_spectrum = target_spectrum / max_val
            
            spectrum_str = ",".join([f"{val:.4f}" for val in target_spectrum])
            data.append({
                "smiles": mol["smiles"],
                "ir_spectrum": spectrum_str
            })
            print(f"  -> 파싱 및 보간 완료 (SMILES: {mol['smiles']})")
            
            time.sleep(1) # Be polite to NIST servers
            
        except Exception as e:
            print(f"  -> 처리 중 오류 발생: {e}")
            
    if data:
        df = pd.DataFrame(data)
        out_path = "datasets/real_ir_data.csv"
        # GNN 파이프라인 배치 처리를 위해 데이터셋 부풀리기 (Data Augmentation 효과)
        df_large = pd.concat([df]*100, ignore_index=True)
        df_large.to_csv(out_path, index=False)
        print(f"\n✅ 스크래핑 및 전처리 완료! 데이터셋이 저장되었습니다: {out_path} (총 {len(df_large)} 샘플로 복제 증강됨)")
    else:
        print("\n❌ 스크래핑한 데이터가 없습니다.")

if __name__ == "__main__":
    download_and_process()

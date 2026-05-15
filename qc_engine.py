import os
import subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import shutil
import tempfile
import urllib.request
import tarfile
import platform

def ensure_xtb_installed():
    """배포 환경에서 xtb 바이너리가 있는지 확인하고, 없으면 다운로드하여 설정합니다."""
    if shutil.which("xtb"):
        return True
    
    # 리눅스 환경(Streamlit Cloud)에서만 자동 다운로드 시도
    if platform.system() == "Linux":
        xtb_dir = os.path.join(os.getcwd(), "xtb_bin")
        xtb_path = os.path.join(xtb_dir, "bin", "xtb")
        
        if not os.path.exists(xtb_path):
            os.makedirs(xtb_dir, exist_ok=True)
            print("⏳ 배포용 xTB 바이너리를 다운로드 중입니다...")
            url = "https://github.com/grimme-lab/xtb/releases/download/v6.7.0/xtb-6.7.0-linux-x86_64.tar.xz"
            target_tar = os.path.join(xtb_dir, "xtb.tar.xz")
            
            try:
                urllib.request.urlretrieve(url, target_tar)
                # tar.xz 압축 해제
                subprocess.run(["tar", "-xf", target_tar, "-C", xtb_dir], check=True)
                os.remove(target_tar)
                print("✅ xTB 설치 완료")
            except Exception as e:
                print(f"❌ xTB 다운로드 실패: {e}")
                return False
        
        # PATH에 추가
        os.environ["PATH"] += os.pathsep + os.path.join(xtb_dir, "bin")
        return True
    
    return False

def calculate_ir_qc(smiles, scaling_factor=0.96):
    """
    SMILES를 입력받아 xTB(GFN2-xTB) 양자화학 계산을 수행하고 IR 피크 정보를 반환합니다.
    """
    ensure_xtb_installed()
    peaks = []
    
    # 1. 임시 작업 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        curr_dir = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # 2. SMILES -> 3D XYZ 생성
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
            
            xyz_path = "mol.xyz"
            xyz_content = f"{mol.GetNumAtoms()}\n\n"
            conf = mol.GetConformer()
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                xyz_content += f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"
            
            with open(xyz_path, "w") as f:
                f.write(xyz_content)
            
            # 3. xTB Hessian 계산 실행
            # --gfn 2: GFN2-xTB 모델 사용
            # --hess: 진동 주파수(Hessian) 계산
            result = subprocess.run(
                ["xtb", xyz_path, "--gfn", "2", "--hess"],
                capture_output=True, text=True, check=True
            )
            
            # 4. g98.out 파싱
            out_path = "g98.out"
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    lines = f.readlines()
                    
                    freqs = []
                    intens = []
                    
                    for line in lines:
                        if "Frequencies --" in line:
                            parts = line.split("--")[1].split()
                            freqs.extend([float(x) * scaling_factor for x in parts])
                        elif "IR Inten    --" in line:
                            parts = line.split("--")[1].split()
                            intens.extend([float(x) for x in parts])
                    
                    # 매칭되는 파수와 강도 저장
                    for f_val, i_val in zip(freqs, intens):
                        if f_val > 0: # 허수 진동 제외
                            peaks.append({"wavenumber": f_val, "intensity": i_val})
                            
        except Exception as e:
            print(f"QC 계산 중 오류 발생: {e}")
        finally:
            os.chdir(curr_dir)
            
    return peaks

def peaks_to_spectrum(peaks, wavenumbers, default_width=30):
    """
    추출된 QC 피크들을 가우시안 곡선으로 변환하여 연속적인 스펙트럼으로 만듭니다.
    """
    spectrum = np.zeros_like(wavenumbers)
    for peak in peaks:
        center = peak["wavenumber"]
        intensity = peak["intensity"]
        
        # 가우시안 렌더링
        # intensity 스케일링 (xTB 강도는 보통 수십~수백 단위이므로 조정 필요)
        norm_intensity = intensity / 500.0 # 시각화용 임의 스케일링
        
        contribution = norm_intensity * np.exp(-((wavenumbers - center) ** 2) / (2 * default_width ** 2))
        spectrum += contribution
        
    return spectrum

if __name__ == "__main__":
    # 단독 테스트
    test_smiles = "CC(=O)O" # Acetic Acid
    print(f"[{test_smiles}] 양자화학 IR 피크 계산 시작...")
    results = calculate_ir_qc(test_smiles)
    for p in results:
        print(f"Wavenumber: {p['wavenumber']:.2f}, Intensity: {p['intensity']:.2f}")

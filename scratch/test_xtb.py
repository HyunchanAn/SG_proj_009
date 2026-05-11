import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

def test_xtb_calculation(smiles="CCO"):
    print(f"테스트 분자: {smiles}")
    
    # 1. SMILES -> 3D XYZ
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    
    xyz_content = f"{mol.GetNumAtoms()}\n\n"
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        xyz_content += f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"
    
    with open("test_mol.xyz", "w") as f:
        f.write(xyz_content)
    
    # 2. Run xTB
    print("xTB 양자역학 계산 중 (Hessian)...")
    try:
        result = subprocess.run(
            ["xtb", "test_mol.xyz", "--gfn", "2", "--hess"],
            capture_output=True, text=True, check=True
        )
        print("계산 완료!")
        
        # 3. Check for spectrum output
        if os.path.exists("g98.out"):
            print("g98.out 파일 생성됨. 내용을 확인합니다.")
            with open("g98.out", "r") as f:
                content = f.read()
                if "Frequencies --" in content:
                    print("✅ 진동 주파수 정보가 정상적으로 포함되어 있습니다.")
                else:
                    print("❌ 주파수 정보를 찾을 수 없습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    test_xtb_calculation()

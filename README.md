# Simulated IR Spectroscopy Generator

## 프로젝트 개요
SMILES 문자열을 입력받아 가상의 IR(Infrared Spectroscopy) 스펙트럼 데이터를 생성하고 시각화하는 파이썬 기반 웹 시뮬레이터입니다. RDKit을 활용하여 분자 구조를 분석하며, Streamlit을 통해 직관적인 대화형 인터페이스(UI)를 제공합니다. 본 프로젝트는 세 가지 다른 기술적 방향성을 실험하기 위해 세 개의 브랜치(Branch)로 나뉘어 개발되었습니다.

## 브랜치(Branch) 구조 및 기술 명세

1. main (휴리스틱 규칙 기반 모델)
유기화학 교과서 및 데이터베이스에 등장하는 경험적 IR 상관 표(Correlation Table)를 활용하는 초기 버전입니다.
- 주요 로직: SMARTS 패턴 매칭을 통해 분자 내 주요 작용기를 식별하고, 각 작용기의 경험적 파수와 강도를 가우시안 함수에 대입.
- 특징: 지문 영역(Fingerprint Region) 무작위 시뮬레이션 및 분광 장비의 베이스라인 노이즈를 수치적으로 모방하여 현실성을 높임. 연산 속도가 즉각적입니다.

2. feature/quantum-ir (양자화학 계산 모델)
반경험적 양자역학 패키지인 GFN2-xTB와 ASE(Atomic Simulation Environment)를 연동한 정밀 예측 모델입니다.
- 주요 로직: 분자의 3D 좌표 생성 및 MMFF94 초기 최적화 수행 후, xTB로 헤시안(Hessian) 행렬을 풀어내 물리적으로 정확한 진동 주파수와 흡수 쌍극자 도함수를 산출.
- 특징: 가우시안 스미어링(Gaussian Smearing)을 통해 이산적 피크를 곡선으로 변환. 분자의 입체적 대칭성으로 인한 IR 비활성 현상 등이 모두 반영되어 과학적으로 정확합니다.

3. feature/ml-ir (머신러닝 GNN 예측 모델)
그래프 신경망(Graph Neural Network)을 이용해 스펙트럼을 실시간 예측하기 위한 프로토타입 파이프라인입니다.
- 주요 로직: RDKit 분자 객체의 원자 특성을 수치화하여 PyTorch 텐서로 변환한 뒤, 자체 설계된 IRGraphNeuralNetwork 아키텍처에 통과시켜 스펙트럼 백터를 추론(Inference).
- 특징: 현재는 아키텍처 스켈레톤과 전처리 코드가 구축된 상태이며, 추후 실제 스펙트럼 데이터베이스(NIST 등) 학습이 완료되면 양자역학 수준의 정밀도를 순식간에 계산할 수 있는 잠재력을 갖습니다.

4. feature/gnn-real-data (실제 데이터 기반 GNN 모델 파인튜닝)
실제 실험 데이터(NIST/SDBS 등)를 활용하여 기존 GNN 모델을 파인튜닝하는 실전형 파이프라인입니다.
- 생성 배경: 휴리스틱 모델의 시각적 한계를 인지하였으나, 양자화학(xTB) 연산은 고분자 혼합물 환경에서 막대한 연산량(O(N^3))을 요구하여 실시간 UI(Streamlit)의 쾌적한 UX를 해칩니다. 반면 Apple M2 Pro(16GB)의 MPS 가속을 활용한 GNN 추론은 밀리초 단위로 결과를 도출하므로 현재 하드웨어 스펙에 최적화된 대안으로 분기되었습니다.
- 데이터 출처 (Data Source): Hugging Face Hub에 공개된 대규모 오픈 스펙트럼 데이터셋 (`Lamblador/IRSpectra`). 약 1만 개 이상의 다양한 유기 화합물 실제 스펙트럼 데이터를 3600차원으로 정밀 보간(Interpolation)하여 모델 파인튜닝에 사용합니다.
- 특징: 실제 실험 데이터에 포함된 온도 노이즈, 수소 결합에 의한 넓은 피크(Broadening) 등 물리적 현실성을 스스로 학습하여 빠르고 정확하게 렌더링합니다.

## 향후 개발 방향 (고분자 혼합물 확장)
현재 단일 분자에 국한된 입력 시스템을 확장하여, 복수의 모노머(Monomer) 화학 구조와 이들의 배합비(%)를 동시에 입력받아 고분자(Polymer) 및 비정질 혼합물의 거시적 IR 스펙트럼을 예측하는 멀티 모달 인공지능으로 고도화할 계획입니다. 이를 위해 `feature/ml-ir` 브랜치가 향후 주력 개발 파이프라인으로 채택되었습니다.

## 설치 및 실행 방법

1. 환경 구성
conda 환경을 사용하실 것을 권장합니다.

> conda create -n ir_env python=3.13
> conda activate ir_env
> pip install rdkit numpy scipy matplotlib streamlit

2. 브랜치별 특수 의존성 (옵션)
feature/quantum-ir 실행 시:
> conda install -c conda-forge xtb-python ase
feature/ml-ir 실행 시:
> pip install torch

3. 애플리케이션 실행
설치가 완료되면 프로젝트 최상위 경로에서 다음 명령을 실행합니다.

> streamlit run app.py

명령어 실행 후 터미널에 안내된 Local URL(예: http://localhost:8502)로 브라우저를 통해 접속할 수 있습니다.

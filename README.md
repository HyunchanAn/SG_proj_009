# 🧪 Polymer IR Simulator Pro (GNN + Quantum Hybrid)

![Status](https://img.shields.io/badge/Status-v1.0_Release-green)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Framework](https://img.shields.io/badge/Backend-PyTorch_GNN-red)
![Physics](https://img.shields.io/badge/Physics-GFN2--xTB-orange)

본 프로젝트는 **그래프 신경망(GNN)**의 통계적 예측력과 **양자화학(xTB)**의 물리적 정밀도를 결합한 차세대 고분자 IR 스펙트럼 시뮬레이터입니다. 단순한 패턴 매칭을 넘어, 실제 실험 데이터와 물리 법칙을 바탕으로 고분자 혼합물의 분광 특성을 정밀하게 예측합니다.

## 🚀 주요 기능 (Key Features)

### 1. GNN + QC 하이브리드 추론 엔진
- **Data-Driven**: 8,352개의 실제 실험 IR 데이터를 학습한 GNN 모델이 분자의 지문 영역을 실시간으로 추론합니다.
- **Physics-Driven**: GFN2-xTB 양자화학 엔진을 통해 새로운 분자의 진동 주파수를 이론적으로 계산하고 보정합니다.
- **Beer-Lambert Law**: 지수 매핑 기법을 적용하여 실제 실험 장비에서 측정된 것과 같은 자연스러운 곡선미를 구현했습니다.

### 2. 고분자 물리 효과 정밀 모델링
- **Chain Effects**: 중합도($n$)에 따른 탄소 사슬의 포화(Saturation), 피크 시프트 및 브로드닝을 물리적으로 모사합니다.
- **Hydrogen Bonding**: 농도와 중합도에 따른 O-H/N-H 피크의 변화(Shift & Broadening)를 정교하게 반영합니다.

### 3. 전문가용 인터랙티브 대시보드
- **Plotly Visualization**: 줌, 팬, 실시간 데이터 툴팁 및 작용기 자동 어노테이션 기능을 제공합니다.
- **Dark/Light Mode**: 사용자 환경에 최적화된 테마 전환 기능을 지원합니다.
- **Composition Optimization**: 실험 데이터를 입력하면 최적의 성분 배합비를 역추적하는 최적화 알고리즘이 탑재되어 있습니다.

## 🛠 기술 스택 (Tech Stack)
- **Core**: Python 3.13, RDKit (Cheminformatics)
- **Deep Learning**: PyTorch, PyTorch Geometric (GNN), Apple Silicon MPS 가속 지원
- **Quantum Chemistry**: GFN2-xTB (Semi-empirical method)
- **Frontend**: Streamlit, Plotly (Interactive Chart)

## 📦 설치 및 실행 (Installation)

### 1. 시스템 의존성 설치 (MacOS/Linux)
양자화학 계산을 위해 `xtb`가 필요합니다.
```bash
brew tap grimme-lab/qc
brew install xtb
```

### 2. 파이썬 환경 구성
```bash
pip install -r requirements.txt
# 또는 주요 패키지 직접 설치
pip install torch torch-geometric rdkit streamlit plotly pandas numpy scipy
```

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```

## 📝 개발 로그 (Development Log)
상세한 개발 과정과 기술적 결정 사항은 [development_log.txt](./development_log.txt)에서 확인하실 수 있습니다.

---
**Last Updated: 2026-05-11**

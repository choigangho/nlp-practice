# 🧠 LLM From Scratch — Practice & Analysis

밑바닥부터 시작하는 LLM(Large Language Model)의 핵심 파이프라인(사전학습 $\rightarrow$ 미세조정 $\rightarrow$ 지시 학습)을 직접 구현하고, 단계별 데이터 변화와 모델 동작 원리를 정량적으로 분석한 학습 기록 저장소입니다.

---

## 📂 Repository Structure

```text
nlp-practice/
├── README.md                      # 프로젝트 전체 개요 및 환경 설정
├── requirements.txt               # 패키지 의존성 목록
├── 01_pretraining/                # Step 1. 데이터 전처리, 토크나이징 및 사전학습
│   ├── README.md                  # 전처리 효율 및 BPE 토크나이저 분석 리포트
│   ├── pretraining.ipynb          # 사전학습 실습 노트북
│   └── data/                      # 원시 코퍼스 및 정제 데이터 (Odyssey, Iliad)
├── 02_finetuning_base/            # Step 2. Base 모델 미세조정 (예정)
│   └── README.md
└── 03_finetuning_instruct/        # Step 3. Instruction Tuning (예정)
    └── README.md
```

---

## 📌 Curriculum & Progress

| 단계 | 디렉터리 | 핵심 주제 | 상세 기록 및 분석 요약 | 상태 |
| :---: | :--- | :--- | :--- | :---: |
| **Step 01** | [`01_pretraining`](./01_pretraining/) | **Data Preparation & Pre-training** | • 정규표현식 기반 코퍼스 정제<br>• GPT-2 BPE 토크나이저 공백 병합 특성 실험<br>• 전처리 전후 토큰 절감률(20.28% 감소) 분석 | 🔄 진행 중 |
| **Step 02** | [`02_finetuning_base`](./02_finetuning_base/) | **Base Fine-tuning** | • 사전학습 가중치 로드 및 태스크 미세조정 | ⏳ 예정 |
| **Step 03** | [`03_finetuning_instruct`](./03_finetuning_instruct/) | **Instruction Fine-tuning** | • 지시문-응답(Instruction-Response) 데이터셋 튜닝 | ⏳ 예정 |

---

## ⚙️ Environment & Setup

* **OS:** Windows 11
* **Python:** 3.11 (Conda env: `llm`)
* **GPU:** NVIDIA GeForce RTX 3050 (8GB VRAM) / CUDA 13.3

```bash
# 1. 가상환경 생성 및 활성화
conda create -n llm python=3.11 -y
conda activate llm

# 2. 의존성 패키지 설치
pip install -r requirements.txt
```

---

## 📚 References & Data Sources
* **Reference Course:** [HongLab LLM Overview](https://github.com/HongLabInc/HongLabLLM/tree/main/youtube_2025_overview)를 참고하여 코드를 재구성하고 자체 분석을 추가했습니다.
* **Corpus:** Project Gutenberg (*The Odyssey*, *The Iliad*)

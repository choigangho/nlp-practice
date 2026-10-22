# 📖 LLM From Scratch — Data Preparation
LLM 구현의 첫 단계로 텍스트 전처리와 토크나이징을 수행하고, 전처리 전후의 데이터 규모를 비교 분석한 기록입니다.


## Overview
| 단계 | 내용 |
|---|---|
| 1. Text Cleaning | 줄바꿈·연속 공백을 치환하여 학습용 텍스트 생성 |
| 2. Tokenization | GPT-2 BPE 토크나이저로 텍스트를 토큰 ID로 변환 |
| 3. Analysis | 전처리 전후의 글자 수 및 토큰 수 비교 분석 |

**사용한 텍스트:** *The Odyssey*, *The Iliad* (영문)

| *The Odyssey* | *The Iliad* |
| :---: | :---: |
| <img height="240" alt="The Odyssey" src="https://github.com/user-attachments/assets/691346b7-65f7-4ca9-9a7c-26ffddc29023" /> | <img height="240" alt="The Iliad" src="https://github.com/user-attachments/assets/a22b8a6b-d5a3-4fe8-bed9-5cd898517d0c" /> |


## Environment
- OS: Windows 11
- Python 3.11 (conda 가상환경 `llm`)
- PyTorch (CUDA UMD Version: 13.3), tiktoken, JupyterLab
- GPU: NVIDIA RTX 3050 (8GB VRAM)

```bash
conda create -n llm python=3.11 -y
conda activate llm
pip install jupyterlab tiktoken
```


## 1. Text Cleaning
줄바꿈(`\n`)과 연속된 공백을 단일 공백으로 치환하여 서식 문자를 제거하고 토큰 소모를 줄입니다.

```python
import re

def clean_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        book_text = file.read()

    cleaned_text = re.sub(r"\s+", " ", book_text).strip()  # 줄바꿈·연속 공백 → 단일 공백으로 치환

    print("cleaned_" + filename, len(cleaned_text), "characters")

    with open("cleaned_" + filename, "w", encoding="utf-8") as file:
        file.write(cleaned_text)

filenames_list = ["The_Odyssey.txt", "The_Iliad.txt"]

for filename in filenames_list:
    clean_text(filename)
```

**Output**
```text
cleaned_The_Odyssey.txt 696052 characters
cleaned_The_Iliad.txt 827148 characters
```


## 2. Tokenization
OpenAI GPT-2의 **BPE(Byte Pair Encoding)** 토크나이저를 적용합니다.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "There is nothing dearer to a man than his own country and his parents."
tokens = tokenizer.encode(text)

print("글자 수:", len(text), "| 토큰 수:", len(tokens))
for t in tokens:
    print(f"{t}\t -> {tokenizer.decode([t])!r}")
```

**Output**
```text
글자 수: 70 | 토큰 수: 16
1858	 -> 'There'
318	 -> ' is'
2147	 -> ' nothing'
390	 -> ' de'
11258	 -> 'arer'
284	 -> ' to'
257	 -> ' a'
582	 -> ' man'
621	 -> ' than'
465	 -> ' his'
898	 -> ' own'
1499	 -> ' country'
290	 -> ' and'
465	 -> ' his'
3397	 -> ' parents'
13	 -> '.'
```

**Observations**
> * 공백 병합: `' is'`, `' nothing'`처럼 단어 앞 공백이 단어와 함께 단일 토큰으로 묶입니다.
> * 서브워드 분할: 빈도가 낮은 단어 `dearer`는 `' de'`와 `'arer'`로 분할됩니다.
> * 압축률: 예시 문장은 70글자가 16토큰으로 변환되어, 1토큰당 평균 약 4.4글자를 포함합니다.


## 3. Dataset Token Count & Preprocessing Impact
원본 텍스트와 전처리 후 텍스트의 글자 수 및 토큰 수를 측정하여 전처리 효율을 분석합니다. *(GPT-2 토크나이저 기준)*

### 3.1 토큰 수 측정
```python
files = [
    "The_Odyssey.txt", "cleaned_The_Odyssey.txt",
    "The_Iliad.txt", "cleaned_The_Iliad.txt"
]

for name in files:
    with open(name, encoding="utf-8") as f:
        data = f.read()
    print(f"{name:<25} | {len(data):>7,} chars -> {len(tokenizer.encode(data)):>7,} tokens")
```

**Output**
```text
The_Odyssey.txt           | 698,117 chars -> 186,230 tokens
cleaned_The_Odyssey.txt   | 696,052 chars -> 171,380 tokens
The_Iliad.txt             | 908,237 chars -> 282,913 tokens
cleaned_The_Iliad.txt     | 827,148 chars -> 202,615 tokens
```

### 3.2 전처리 전후 비교 분석
전처리 수행 결과, 전체 글자 수는 **5.18% 감소**한 반면 전체 토큰 수는 **20.28% 감소**했습니다.

| 데이터셋 | 전처리 전 (Chars / Tokens) | 전처리 후 (Chars / Tokens) | 감소량 (Chars / Tokens) | 감소율 (Chars / Tokens) |
|---|---:|---:|---:|---:|
| **The Odyssey** | 698,117 / 186,230 | 696,052 / 171,380 | 2,065 / 14,850 | 0.30% / 7.97% |
| **The Iliad** | 908,237 / 282,913 | 827,148 / 202,615 | 81,089 / 80,298 | 8.93% / 28.38% |
| **Total** | **1,606,354 / 469,143** | **1,523,200 / 373,995** | **83,154 / 95,148** | **5.18% / 20.28%** |

- `감소율(%) = (전처리 전 - 전처리 후) / 전처리 전 × 100`
- 연속된 공백과 줄바꿈이 단일 공백으로 압축된 것에 더해, 단일 줄바꿈(`\n`)이 공백(` `)으로 치환되면서 글자 수 감소율(5.18%) 대비 높은 토큰 감소율(20.28%)을 보였습니다.

#### 글자 수 대비 토큰 감소율이 높은 이유
GPT-2 BPE 토크나이저는 단어 앞의 공백(` `)을 뒷단어와 묶어 단일 토큰으로 처리하지만, 줄바꿈(`\n`)은 뒷단어와 병합하지 않고 별도의 토큰으로 분리합니다. 

예를 들어 *The Odyssey*의 구절 일부인 `own\ncountry`를 `own country`로 치환하면 결과는 다음과 같습니다.

- **전처리 전 (`own\ncountry`)**
  - 글자 수: 11자 (`\n` 포함)
  - 토큰화 결과: `['own', '\n', 'country']` → **3토큰**
- **전처리 후 (`own country`)**
  - 글자 수: 11자 (공백 포함, 글자 수 변화 없음)
  - 토큰화 결과: `['own', ' country']` → **2토큰** (공백이 `country` 앞에 병합됨)

원본 텍스트의 줄바꿈(`\n`)이 공백(` `)으로 치환되면서 글자 수는 유지되더라도 토큰 수는 감소합니다.

> **참고 사항**
> * API 비용 추정 시 모델별 입력(Input) 요금과 토큰 수를 기준으로 계산하며, 출력(Output) 비용은 제외합니다.
> * 본 문서는 `GPT-2 tokenizer` 기준으로 측정되었으므로, 실제 토큰 수는 모델별 토크나이저에 따라 달라질 수 있습니다.


## 데이터 출처
- *The Odyssey* - Project Gutenberg
- *The Iliad* - Project Gutenberg

# 📖 Step 01. Pre-training — Data Preparation & Tokenization

LLM 사전학습(Pre-training)의 첫 단계로 원시 텍스트 전처리와 BPE 토크나이징을 수행하고, 전처리 전후의 토큰 압축 효율을 정량적으로 비교 분석한 기록입니다.

---

## 📋 Overview

| 단계 | 핵심 내용 | 관련 파일 |
| :--- | :--- | :--- |
| **1. Text Cleaning** | 줄바꿈·연속 공백을 단일 공백으로 치환하여 학습용 코퍼스 정제 | `data/cleaned_*.txt` |
| **2. Tokenization** | GPT-2 BPE 토크나이저를 통한 서브워드 분할 및 공백 병합 확인 | [`pretraining.ipynb`](./pretraining.ipynb) |
| **3. Impact Analysis** | 전처리 전후의 글자 수 대비 토큰 수 절감 메커니즘 규명 | 본 문서 3절 |

**사용 코퍼스:** *The Odyssey*, *The Iliad* (영문 원서, Project Gutenberg)

| *The Odyssey* | *The Iliad* |
| :---: | :---: |
| <img height="220" alt="The Odyssey" src="https://github.com/user-attachments/assets/691346b7-65f7-4ca9-9a7c-26ffddc29023" /> | <img height="220" alt="The Iliad" src="https://github.com/user-attachments/assets/a22b8a6b-d5a3-4fe8-bed9-5cd898517d0c" /> |

---

## 1. Text Cleaning
원시 텍스트에 포함된 불필요한 줄바꿈(`\n`)과 연속된 공백을 단일 공백으로 치환하여 서식 노이즈를 제거하고 컨텍스트 윈도우 낭비를 줄입니다.

```python
import os
import re

def clean_text(filename, data_dir="data"):
    input_path = os.path.join(data_dir, filename)
    output_path = os.path.join(data_dir, f"cleaned_{filename}")

    with open(input_path, "r", encoding="utf-8") as file:
        book_text = file.read()

    # 줄바꿈·연속 공백 → 단일 공백으로 치환
    cleaned_text = re.sub(r"\s+", " ", book_text).strip()

    print(f"cleaned_{filename}: {len(cleaned_text):,} characters")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)

filenames_list = ["The_Odyssey.txt", "The_Iliad.txt"]
for filename in filenames_list:
    clean_text(filename)
```

**Output**
```text
cleaned_The_Odyssey.txt: 696,052 characters
cleaned_The_Iliad.txt: 827,148 characters
```

---

## 2. Tokenization
OpenAI GPT-2의 **BPE(Byte Pair Encoding)** 토크나이저(`tiktoken`)를 적용하여 텍스트가 토큰 ID로 매핑되는 과정을 확인합니다.

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
> * **공백 병합 (Space Merging):** `' is'`, `' nothing'`처럼 단어 앞의 공백이 뒷단어와 함께 단일 토큰으로 묶여 어휘 사전(Vocabulary) 효율을 높입니다.
> * **서브워드 분할 (Subword Splitting):** 상대적으로 등장 빈도가 낮은 단어인 `dearer`는 어근부 `' de'`와 접미부 `'arer'` 두 개의 서브워드 토큰으로 분할됩니다.
> * **압축률:** 예시 문장 기준 70글자가 16토큰으로 변환되어, **1토큰당 평균 약 4.4글자**를 포함합니다.

---

## 3. Dataset Token Count & Preprocessing Impact
원본 텍스트와 정제된 텍스트의 글자 수 및 토큰 수를 비교하여 전처리가 모델 입력 길이에 미치는 영향을 분석합니다.

### 3.1 토큰 수 측정
```python
import os

files = [
    "The_Odyssey.txt", "cleaned_The_Odyssey.txt",
    "The_Iliad.txt", "cleaned_The_Iliad.txt"
]

for name in files:
    path = os.path.join("data", name)
    with open(path, encoding="utf-8") as f:
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
| :--- | ---: | ---: | ---: | ---: |
| **The Odyssey** | 698,117 / 186,230 | 696,052 / 171,380 | 2,065 / 14,850 | 0.30% / 7.97% |
| **The Iliad** | 908,237 / 282,913 | 827,148 / 202,615 | 81,089 / 80,298 | 8.93% / 28.38% |
| **Total** | **1,606,354 / 469,143** | **1,523,200 / 373,995** | **83,154 / 95,148** | **5.18% / 20.28%** |

$$
\text{감소율}\,(\%) = \frac{\text{전처리 전} - \text{전처리 후}}{\text{전처리 전}} \times 100
$$

#### 💡 왜 글자 수 감소율(5.18%)보다 토큰 감소율(20.28%)이 4배 가까이 높을까?
GPT-2 BPE 토크나이저는 일반 공백(` `)은 뒷단어와 묶어 하나의 토큰으로 병합하지만, **줄바꿈 문자(`\n`)는 뒷단어와 병합하지 않고 독립된 토큰(Token ID `198`)으로 분리**하기 때문입니다.

예를 들어 본문 중 `own\ncountry` 구절이 `own country`로 치환될 때의 차이는 다음과 같습니다.

* **전처리 전 (`own\ncountry`)**
  * 글자 수: 11자 (`\n` 포함)
  * 토큰화 결과: `['own', '\n', 'country']` $\rightarrow$ **3 토큰**
* **전처리 후 (`own country`)**
  * 글자 수: 11자 (공백 포함, **글자 수 변화 0**)
  * 토큰화 결과: `['own', ' country']` $\rightarrow$ **2 토큰** (공백이 `country` 앞에 병합됨)

즉, 전자책 서식을 위해 강제 개행된 수많은 단일 줄바꿈(`\n`)이 공백(` `)으로 바뀌면서 **글자 수는 그대로 유지되더라도 토큰 수는 1개씩 직접 줄어드는 효과**가 발생하여 약 9만 5천 개의 불필요한 토큰이 절감되었습니다. 이는 동일한 Context Length 내에 더 많은 실질적 문맥 정보를 담을 수 있음을 의미합니다.

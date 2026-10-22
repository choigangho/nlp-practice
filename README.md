# 📖 LLM From Scratch — Data Preparation

LLM을 밑바닥부터 만들기 위한 첫 단계로, **텍스트 전처리**와 **토크나이징**을 구현하고 분석한 기록입니다.


## Overview

| 단계 | 내용 |
|---|---|
| 1. Text Cleaning | 줄바꿈·연속 공백을 정리하여 학습용 텍스트 생성 |
| 2. Tokenization | GPT-2 BPE 토크나이저로 텍스트를 토큰 ID로 변환 |
| 3. Analysis | 글자 수 대비 토큰 수를 측정해 데이터 규모 파악 |

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

줄바꿈과 연속된 공백을 **하나의 공백**으로 치환하여, 모델이 불필요한 서식 문자를 학습하지 않도록 하여 토큰 낭비를 최소화합니다!

```python
import re

def clean_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        book_text = file.read()

    cleaned_text = re.sub(r"\s+", " ", book_text).strip()  # 줄바꿈·연속 공백 → 공백 1개로 치환

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

OpenAI GPT-2가 사용하는 **BPE(Byte Pair Encoding)** 토크나이저를 사용했습니다.

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

> **Observations**
> ' is'`, `' nothing'`처럼 단어 앞 공백이 같은 토큰으로 묶입니다.
> **서브워드 분할**: 희소한 단어 `dearer`는 `' de'` + `'arer'`로 쪼개집니다.
> **압축률**: 위 문장은 70글자가 16토큰으로, 토큰 하나당 약 4.4글자입니다.


## Dataset Token Count & Preprocessing Impact

그렇다면 더욱 실용적인 관점에서 원본 텍스트와 전처리(Cleaning) 후 텍스트의 글자 수 및 토큰 수를 측정하여 전처리 효율과 API 입력 비용 절감 효과를 분석해보겠습니다. *(토큰 수는 GPT-2 토크나이저 기준)*

### 1. 토큰 수 측정

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

### 2. 전처리 전후 비교 분석

전처리를 통해 불필요한 공백, 특수문자, 메타데이터 등을 제거한 결과, 전체 토큰 수가 **약 20.28% 감소**했습니다.

| 데이터셋 | 전처리 전 (Tokens) | 전처리 후 (Tokens) | 감소량 (Tokens) | 감소율 (%) |
|---|---:|---:|---:|---:|
| **The Odyssey** | 186,230 | 171,380 | 14,850 | 7.97% |
| **The Iliad** | 282,913 | 202,615 | 80,298 | 28.38% |
| **Total** | **469,143** | **373,995** | **95,148** | **20.28%** |

* `$\text{감소율 (\%)} = \frac{\text{전처리 전} - \text{전처리 후}}{\text{전처리 전}} \times 100$`
* **참고:** 글자 수(chars) 기준으로는 전체 `1,606,354`자에서 `1,523,200`자로 총 `83,154`자(약 5.18%) 감소했습니다. 글자 수 대비 토큰 감소 폭이 큰 이유는 토크나이저가 잘게 쪼개던 노이즈 및 공백, 특수 기호들이 정리되었기 때문입니다.

> **💡 참고 사항**
> * API 비용은 모델별 **입력(Input) 요금**과 토큰 수를 기준으로 추정하며, 출력(Output) 비용은 제외합니다.
> * 본 문서는 `GPT-2 tokenizer` 기준으로 측정되었으므로, 실제 API 호출 시 과금되는 토큰 수는 각 모델별 토크나이저에 따라 달라질 수 있습니다.


## 데이터 출처

- *The Odyssey* - gutenberg
- *The Iliad* - gutenberg

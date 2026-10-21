# 📖 LLM From Scratch — Data Preparation

LLM을 밑바닥부터 만들기 위한 첫 단계로, **텍스트 전처리**와 **토크나이징**을 구현하고 분석한 기록입니다.

## Overview

| 단계 | 내용 |
|---|---|
| 1. Text Cleaning | 줄바꿈·연속 공백을 정리하여 학습용 텍스트 생성 |
| 2. Tokenization | GPT-2 BPE 토크나이저로 텍스트를 토큰 ID로 변환 |
| 3. Analysis | 글자 수 대비 토큰 수를 측정해 데이터 규모 파악 |

사용한 텍스트: *The Odyssey*, *The Iliad* (영문)

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

줄바꿈과 연속된 공백을 **하나의 공백**으로 치환하여, 모델이 불필요한 서식 문자를 학습하지 않도록 하여 토큰 낭비를 최소화합니다.

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

## 3. Observations

- ' is'`, `' nothing'`처럼 단어 앞 공백이 같은 토큰으로 묶입니다.
- **서브워드 분할**: 희소한 단어 `dearer`는 `' de'` + `'arer'`로 쪼개집니다.
- **압축률**: 위 문장은 70글자가 16토큰으로, 토큰 하나당 약 4.4글자입니다.

## Dataset Token Count

전체 텍스트의 토큰 수 측정:

```python
for name in ["cleaned_The_Odyssey.txt", "cleaned_The_Iliad.txt"]:
    with open(name, encoding="utf-8") as f:
        data = f.read()
    print(name, len(data), "chars ->", len(tokenizer.encode(data)), "tokens")
```

**Output**

```text
cleaned_The_Odyssey.txt 696052 chars -> 171380 tokens
cleaned_The_Iliad.txt 827148 chars -> 202615 tokens
```

### TODO 전처리 전후의 토큰 수를 비교하고, API 입력 비용에 미치는 영향을 분석할 예정입니다.
- [ ] 원본 및 전처리 텍스트의 토큰 수 측정 (GPT-2 tokenizer)
- [ ] 토큰 감소량 및 감소율 계산
- [ ] 주요 LLM의 표준 API 요금 기준 예상 비용 비교

| 항목 | 전처리 전 | 전처리 후 | 감소량 |
|---|---:|---:|---:|
| The Odyssey | TODO | TODO | TODO |
| The Iliad | TODO | TODO | TODO |
| **Total** | **TODO** | **TODO** | **TODO** |

감소율 = (전처리 전 - 전처리 후) / 전처리 전 × 100

> API 비용은 모델별 입력 요금과 토큰 수를 기준으로 추정하며, 출력 비용은 제외합니다.
> 실제 토큰 수는 모델별 토크나이저에 따라 달라질 수 있습니다.

## 데이터 출처

- *The Odyssey* - gutenberg
- *The Iliad* - gutenberg

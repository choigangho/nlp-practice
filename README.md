## 텍스트 정리 (Text Cleaning)

텍스트에서 줄바꿈과 연속된 공백을 정리하는 코드입니다.

## 코드

```python
import re

def clean_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        book_text = file.read()

    cleaned_text = re.sub(r'\n+', ' ', book_text)  # 줄바꿈을 빈칸으로 변경
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 여러 빈칸을 하나로

    print("cleaned_" + filename, len(cleaned_text), "characters") # 글자 수 출력

    with open("cleaned_" + filename, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

filenames_list = ["The_Odyssey.txt"]

for filename in filenames_list:
    clean_text(filename)
```

## 출력

```text
cleaned_The_Odyssey.txt 696052 characters
```

## 설명

1. 텍스트 파일을 읽기
2. 줄바꿈과 연속된 공백을 공백을 정리
3. `cleaned_` 가 앞에 붙은 새 파일로 저장

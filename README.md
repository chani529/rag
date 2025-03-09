# rag

> HuggingFace-Hub 설치
```bash
pip install huggingface-hub
```

아래의 예시는 `EEVE-Korean-Instruct-10.8B-v1.0`
- HF: https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0
- GGUF: https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF

GGUF 파일을 다운로드 받기 위하여 https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF 에서 원하는 .gguf 모델을 다운로드 받습니다.

순서대로
- `HuggingFace Repo`
- .gguf 파일명
- local-dir 설정
- 심볼릭 링크 설정
  
```bash
huggingface-cli download \
  heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
  ggml-model-Q5_K_M.gguf \
  --local-dir 본인의_컴퓨터_다운로드폴더_경로 \
  --local-dir-use-symlinks False
```

### Modelfile

> EEVE-Korean-Instruct-10.8B-v1.0 예시
```
FROM ggml-model-Q5_K_M.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER stop <s>
PARAMETER stop </s>
```

# 올라마 실행 후

# python
python3.11

python3.11 -m venv venv

venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

streamlit run main.py
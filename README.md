# 🩺 의료용 RAG 질의응답 비서 (Medical RAG QA Assistant)

이 저장소는 의료 전사(Medical Transcription) 데이터를 활용하여 사용자의 의료 관련 질문에 답변하는 **RAG (Retrieval-Augmented Generation)** 시스템을 구현한 프로젝트입니다. 

로컬 벡터 저장소(FAISS)에서 관련 임상 맥락을 검색하고, 이를 바탕으로 정확한 답변을 생성합니다. 모든 답변에는 정보의 투명성을 위해 출처 문서와 진료 과목(Specialty) 정보가 포함됩니다.

## 🌟 주요 기능
- **데이터 기반 답변**: 저장된 의료 데이터를 바탕으로 신뢰할 수 있는 답변을 제공합니다.
- **출처 표기 (Citations)**: 답변의 근거가 되는 "출처 문서"와 "진료 과목"을 함께 보여줍니다.
- **RAG 파이프라인**: FAISS를 활용한 벡터 검색과 LangChain을 통한 효과적인 워크플로우를 구축했습니다.
- **Streamlit 인터페이스**: 웹 기반의 직관적이고 인터랙티브한 UI를 제공합니다.

## 🛠️ 기술 스택
- **LLM (언어 모델)**: Google Gemini Pro (`gemini-pro`)
- **임베딩**: Google Generative AI Embeddings (`models/text-embedding-004`)
- **프레임워크**: LangChain (v0.3+)
- **벡터 DB**: FAISS (로컬 CPU 기반)
- **웹 UI**: Streamlit
- **언어**: Python 3.10+

## 📂 데이터셋 정보
- **출처**: Medical Transcriptions Dataset (Kaggle) — [Kaggle 링크](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- **내용**: 다양한 진료 과목의 의료 전사 샘플 데이터
- **전처리**: 정제 및 청크 분할(약 1,000자) 후 FAISS 인덱스로 임베딩 및 저장됨

## ⚙️ 로컬 설치 및 설정 방법

### 1. 저장소 복제 (Clone)
```bash
git clone https://github.com/wqhekjbwdsaq123/Med_RAG.git
cd Med_RAG
```

### 2. 가상 환경 생성 및 활성화 (권장)
```bash
python -m venv venv
# Windows 환경
venv\Scripts\activate
# macOS / Linux 환경
source venv/bin/activate
```

### 3. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. API 키 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 Google API 키를 입력합니다.
```env
GOOGLE_API_KEY=여러분의_구글_API_키
```
> [!TIP]
> Google AI Studio에서 API 키를 발급받으실 수 있습니다.

### 5. 애플리케이션 실행
```bash
streamlit run app.py
```

## 📁 프로젝트 구조
```
├── app.py                  # Streamlit 메인 애플리케이션
├── requirements.txt        # 설치가 필요한 패키지 목록
├── .env                    # API 키 설정 파일 (깃허브에 공유되지 않음)
├── vectorstore/            # FAISS 인덱스 및 메타데이터 저장 폴더
│   ├── index.faiss
│   └── index.pkl
└── task1_evaluation.csv    # 평가 결과 및 테스트 쿼리 데이터
```

## ⚖️ 라이선스
사용된 데이터셋(tboyle10)은 Kaggle의 CC0 (Public Domain) 라이선스를 따릅니다. 상업적 이용 전 데이터셋 라이선스를 다시 한번 확인하시기 바랍니다.

# 인스타그램 리뷰 불만 유형 자동 분류

기존 감정 분석은 리뷰가 긍정인지 부정인지까지만 알려줍니다. 앱 개발자에게 필요한 것은 "불만이 많다"가 아니라 **"무엇을 고쳐야 하는가"** 이므로, 이 프로젝트는 인스타그램 부정 리뷰를 **6개 불만 유형**으로 자동 분류하는 모델을 만듭니다.


```
리뷰 → (감정 모델) 긍정/부정 → 부정이면 → (유형 모델) 6개 유형 중 하나 → 유형별 건수 집계
```

## 저장소 구조

```
├── README.md
├── reference/
│   └── Naver_Movie_Review_Sentiment_Analysis.ipynb   # 참조 코드 재현 (Colab 실행 결과 포함)
├── scraper/
│   └── collect_reviews.py                 # Google Play 리뷰 수집 코드
└── notebooks/                             # 1단계 감정 학습, 2단계 유형 분류 (예정)
```

수집한 리뷰 원본(CSV)과 학습된 모델 파일은 올리지 않습니다. 비공식 라이브러리로 수집한 데이터이므로 **수집 코드만 공개**합니다.

## 1. 데이터셋

| 용도 | 데이터 | 확보 방법 |
|---|---|---|
| 참조 코드 재현 | 네이버 영화 리뷰 NSMC 20만 개 (학습 15만 / 평가 5만, 긍정·부정 라벨) | [e9t/nsmc](https://github.com/e9t/nsmc), 코드에서 자동 다운로드 |
| 우리 프로젝트 | Google Play 인스타그램 리뷰 (`com.instagram.android`, `lang="ko"`, `country="kr"`) | [google-play-scraper](https://pypi.org/project/google-play-scraper/)로 본문·별점 수집 |

- **감정 라벨**: 별점으로 자동 생성. 1~2점 부정, 4~5점 긍정, 3점 제외
- **현재 확보**: 부정 4,000개 + 긍정 4,000개 = 8,000개 (별점별 2,000개씩)
- **유형 라벨**: 부정 리뷰 300~500개에 조원이 직접 부여 (유형마다 최소 50개 목표)

### 사전 조사

1·2점 리뷰 800개를 수집해 280개를 직접 읽었습니다.

- 버그 관련 리뷰 약 45%, 두 유형 이상 섞인 리뷰 약 23%
- 키워드로는 어느 유형에도 분류되지 않는 리뷰가 약 28% → 키워드 규칙이 아니라 **학습 모델이 필요한 이유**
- 세 비율은 서로 다른 기준으로 각각 센 값이라 더하지 않습니다.

### 학습용 불만 유형 (6개)

| 유형 | 예시 |
|---|---|
| 버그·오류 | 릴스 멈춤, DM 사라짐, 전송 실패 |
| 성능·기기 호환 | 로딩 지연, 버벅임, 특정 기기의 화면 문제 |
| 계정 | 계정 정지, 해킹 후 복구, 로그인·인증 문제 |
| 광고·추천 | 과도한 광고, 관심 없는 추천 피드 |
| UI 변경·기능 요청 | 화면 변경 불편, 기능 추가·복원 요구 |
| 기타 | 유해 콘텐츠·신고, 그 밖의 불만 |

라벨링은 8개 세부 유형으로 하고 학습할 때 6개로 합칩니다. 여러 불만이 섞인 리뷰는 주 유형과 부 유형을 함께 기록합니다.

## 2. 참조 코드

**출처: 위키독스 「딥 러닝을 이용한 자연어 처리 입문」 10-06 네이버 영화 리뷰 감성 분류하기**
https://wikidocs.net/44249

`reference/` 폴더의 노트북은 위 코드를 Colab에서 그대로 실행한 결과입니다. 코드의 저작권은 원 저자에게 있습니다.

| 단계 | 내용 |
|---|---|
| 전처리 | Okt `morphs(stem=True)` + 불용어 제거 → Tokenizer로 단어를 ID로 변환 (3회 미만 단어 제외) → 길이 30으로 패딩 |
| 모델 | `Embedding(vocab_size, 100)` → `LSTM(128)` → `Dense(1, sigmoid)` |
| 학습 | `rmsprop`, `binary_crossentropy`, 테스트 정확도 약 85% |

## 3. NLP 모델 작동 원리

```
리뷰 문장 → 형태소 분석 → 단어 ID → Embedding → LSTM → Dense → 분류 결과
```

### ① 전처리

Okt로 형태소를 나누고 불용어를 제거한 뒤, Tokenizer로 각 단어를 정수 ID로 바꾸고 길이를 30으로 맞춥니다.

```
"릴스가 계속 멈춰요" → [릴스, 계속, 멈추다] → [12, 37, 84, 0, ..., 0]
```

### ② Embedding

가중치 행렬 `W`(어휘 수 × 100)에서 각 단어 ID의 행을 꺼내 100차원 벡터로 바꿉니다. 원핫 벡터와 `W`를 곱하는 대신 해당 행만 꺼내는, 수업에서 배운 Embedding 계층과 같은 원리입니다. 리뷰 하나는 (30 × 100) 벡터 열이 됩니다.

### ③ LSTM

단어 벡터를 순서대로 하나씩 읽으며 문장 전체를 128차원 벡터 `h`로 요약하고, 마지막 `h` 하나만 다음 층으로 보냅니다(다 대 일).

RNN은 매 단어마다 `h`를 `tanh`와 같은 가중치에 통과시켜 새로 쓰기 때문에, 문장이 길어지면 역전파 때 기울기가 앞쪽 단어까지 전달되지 않는 **기울기 소실** 문제가 있습니다. LSTM은 셀 상태 `c`를 따로 두고 아래처럼 갱신합니다.

```
c = f × c_prev + i × g      # 망각 게이트 f, 입력 게이트 i, 새 정보 g
h = o × tanh(c)             # 출력 게이트 o
```

게이트 `f`, `i`, `o`는 0~1 사이의 시그모이드 값으로, 정보를 얼마나 통과시킬지 정하는 밸브 역할을 하며 학습으로 정해집니다. `c`는 곱하고 더하기만 하므로 기울기가 멀리까지 유지됩니다.

### ④ Dense (출력층)

`h`에 가중치를 곱하고 편향을 더하는 완전연결층(교재의 Affine)입니다.

| | 출력층 | 손실 함수 |
|---|---|---|
| 기존 감정 분석 | `Dense(1, sigmoid)` → 긍정 확률 | `binary_crossentropy` |
| 우리 유형 분류 | `Dense(6, softmax)` → 6개 유형의 확률 | `sparse_categorical_crossentropy` |

### ⑤ 학습

순전파로 예측 확률을 구하고, 정답과 비교한 교차 엔트로피 손실을 역전파해 Dense, LSTM, Embedding의 가중치를 함께 갱신합니다. 리뷰 하나와 정답 하나가 학습 데이터 한 개이며, 미니배치 단위로 학습합니다.

## 4. 우리가 추가할 내용

| 항목 | 기존 코드 | 우리 프로젝트 |
|---|---|---|
| 대상·목적 | 영화 리뷰의 긍정·부정 | 인스타그램 부정 리뷰의 불만 유형 |
| 정답 | 0 / 1 감정 라벨 | 0~5 유형 라벨 |
| 출력층 | `Dense(1, sigmoid)` | `Dense(6, softmax)` |
| 손실 함수 | `binary_crossentropy` | `sparse_categorical_crossentropy` |

### 2단계 전이 학습

유형 라벨이 300~500개로 적기 때문에 두 단계로 학습합니다.

1. **1단계**: 별점 라벨 인스타그램 리뷰 8,000개로 감정 분석 학습 (`Embedding → LSTM → Dense(1, sigmoid)`)
2. **2단계**: 1단계의 Embedding·LSTM 가중치를 가져오고 출력층만 `Dense(6, softmax)`로 바꿔 유형 라벨로 학습

두 단계는 같은 Tokenizer(단어 사전)를 사용합니다. 사전이 바뀌면 같은 단어가 다른 ID가 되어, 가져온 Embedding이 엉뚱한 벡터를 꺼내기 때문입니다.

### 평가와 확장 계획

- 유형 분류 정확도, 유형별 정확도, 처음부터 학습한 모델과 전이 학습 모델 비교
- 버그 유형 쏠림 → 클래스 가중치 적용, 소수 유형 라벨 추가
- 데이터 부족 → 수집량을 2만~4만 개로 확대, 라벨 없는 리뷰로 학습한 word2vec 임베딩을 초기값으로 쓰는 실험
- 섞인 리뷰 → `Dense(6, sigmoid)` + `binary_crossentropy` 다중 레이블로 확장


## 참고자료
- 실행 결과 보기: [Colab에서 열기](https://colab.research.google.com/github/shs98ko/instagram-review-complaint-classifier/blob/main/reference/Naver_Movie_Review_Sentiment_Analysis.ipynb) (GitHub 미리보기가 지원되지 않아 Colab 링크로 확인)
- 참조 코드: [위키독스 10-06 네이버 영화 리뷰 감성 분류하기](https://wikidocs.net/44249)
- LSTM 원리: [위키독스 08-02 장단기 메모리(LSTM)](https://wikidocs.net/22888)
- 전이 학습: [Keras Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)
- 데이터셋: [NSMC](https://github.com/e9t/nsmc), [google-play-scraper](https://pypi.org/project/google-play-scraper/), [인스타그램 Google Play](https://play.google.com/store/apps/details?id=com.instagram.android)
- 교재: 사이토 고키, 『밑바닥부터 시작하는 딥러닝 2』 (4장 word2vec, 5장 RNN, 6장 LSTM)

"""
Google Play 인스타그램 리뷰 수집 + 데이터 점검

- 별점별로 나눠서 수집해야 긍정/부정이 고르게 모임 (3점은 제외)
- 라벨: 1~2점 → 부정(0), 4~5점 → 긍정(1)
- 결과는 data/insta_reviews.csv 로 저장 (data/ 폴더는 .gitignore로 GitHub에 올라가지 않음)

"""
import os

import pandas as pd
from google_play_scraper import Sort, reviews

APP_ID = "com.instagram.android"
COUNT_PER_SCORE = 2000          # 별점별 수집 개수 (2만~4만 개로 늘리려면 이 값을 키움)
SCORES = [1, 2, 4, 5]           # 3점 제외
OUT_PATH = "data/insta_reviews.csv"


def collect():
    all_reviews = []
    for score in SCORES:
        result, _ = reviews(
            APP_ID,
            lang="ko",
            country="kr",
            sort=Sort.NEWEST,
            count=COUNT_PER_SCORE,
            filter_score_with=score,    # 해당 별점만 수집
        )
        print(f"{score}점: {len(result)}개")
        all_reviews.extend(result)

    df = pd.DataFrame(all_reviews)[["content", "score", "at"]]
    df = df.rename(columns={"content": "document"})       # 10-06 코드와 열 이름 맞춤
    df["label"] = (df["score"] >= 4).astype(int)           # 별점 → 긍정(1) / 부정(0)
    return df


def report(df):
    """전처리 기준을 정하기 위한 데이터 점검 (빈 리뷰, 중복, 짧은 글)"""
    print("\n=== 기본 정보 ===")
    print("전체 리뷰 수:", len(df))
    print("빈 리뷰:", df["document"].isnull().sum())
    print("중복 리뷰:", df["document"].duplicated().sum())

    print("\n=== 긍정/부정별 ===")
    for lab, name in [(0, "부정"), (1, "긍정")]:
        part = df[df["label"] == lab]
        print(f"[{name}] {len(part)}개 | "
              f"10자 미만 {(part['document'].str.len() < 10).sum()}개 | "
              f"평균 길이 {part['document'].str.len().mean():.0f}자")

    print("\n=== 가장 많이 중복된 리뷰 TOP 10 ===")
    print(df["document"].value_counts().head(10))


if __name__ == "__main__":
    df = collect()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH} ({len(df)}개)")
    print(df["label"].value_counts())
    report(df)

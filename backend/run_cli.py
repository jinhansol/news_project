import os

# ✅ 환경 변수 설정
os.environ["MECABRC"] = r"C:\Users\jinha\AppData\Local\Programs\Python\Python310\lib\site-packages\unidic_lite\dicdir\mecabrc"

import sys
import requests
from voice_chat import listen_and_transcribe_with_vad, speak

API_URL = "http://localhost:8000/news_trend/"

def get_news_summary(keyword: str):
    try:
        response = requests.post(API_URL, json={"keyword": keyword})
        if response.status_code == 200:
            data = response.json()
            return data.get("trend_digest", "요약 결과가 없습니다.")
        else:
            return f"API 오류: {response.status_code} - {response.text}"
    except Exception as e:
        return f"API 호출 실패: {str(e)}"

def main():
    print("음성으로 뉴스 키워드를 말하세요. 종료하려면 Ctrl+C를 누르세요.")
    while True:
        keyword = listen_and_transcribe_with_vad()
        if not keyword:
            print("키워드를 인식하지 못했습니다. 다시 시도하세요.")
            continue
        print(f"인식된 키워드: {keyword}")

        summary = get_news_summary(keyword)
        print(f"\n[뉴스 요약]\n{summary}\n")

        speak(summary)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        sys.exit(0)

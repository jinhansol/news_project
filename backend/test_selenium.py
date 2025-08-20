from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import time

print(">>> [최종 진단] 1. 엣지 브라우저 실행을 시도합니다...")
driver = None
try:
    service = EdgeService(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service)
    print(">>> [최종 진단] 2. 브라우저가 성공적으로 실행되었습니다.")

    # [수정] 네이버 대신, 가장 간단한 테스트 사이트로 이동합니다.
    print(">>> [최종 진단] 3. example.com 으로 이동을 시도합니다...")
    driver.get("https://www.example.com")
    print(f">>> [최종 진단] 4. 이동 성공! 현재 페이지 제목: {driver.title}")

    print(">>> [최종 진단] 5. 테스트 성공! 10초 후에 브라우저를 자동으로 닫습니다.")
    time.sleep(10)

except Exception as e:
    print(f"!!! [최종 진단] 테스트 중 오류가 발생했습니다: {e}")
    print("!!! 원인: 컴퓨터의 방화벽, 백신 프로그램, 또는 네트워크 정책이 원인일 가능성이 매우 높습니다.")

finally:
    if driver:
        driver.quit()
        print(">>> [최종 진단] 6. 브라우저를 닫았습니다. 테스트 종료.")
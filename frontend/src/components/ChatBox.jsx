// src/components/ChatBox.jsx

import { useEffect, useRef } from "react";

export default function ChatBox({
  messages = [],   // undefined 넘어와도 빈 배열로 안전 처리
  onClear
}) {
  const chatEndRef = useRef(null);

  // 메시지가 추가될 때 자동 스크롤
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className="chat-window">
      {/* 대화 위에 '기록삭제' 버튼 */}
      {messages.length > 0 && (
        <div style={{ textAlign: "center", marginBottom: 12 }}>
          <button className="clear-btn-in-chat" onClick={onClear}>
            기록삭제
          </button>
        </div>
      )}

      {/* 메시지 블록 */}
      {messages.map((msg, i) => (
        <div key={i} className="chat-message-block">
          <div className="chat-bubble user">🧑‍💻 {msg.query}</div>
          <div className="chat-bubble ai">
            {/* 트렌드 요약 */}
            <div className="trend-summary">
              <h4>✨ {msg.results.purpose} 트렌드 요약</h4>
              <p>{msg.results.trend_digest}</p>
            </div>
            <hr className="divider" />

            {/* 개별 기사 카드 */}
            {msg.results.trend_articles?.map((article, j) => (
              <div key={j} className="news-card">
                <h4>{article.title}</h4>
                <p>{article.summary}</p>
                <a href={article.url} target="_blank" rel="noreferrer">
                  기사 원문 보기
                </a>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div ref={chatEndRef} />
    </div>
  );
}

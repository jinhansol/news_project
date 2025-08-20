// src/App.js
import { useState, useEffect, useRef, useMemo } from "react";
import HeadlineGrid from "./components/HeadlineGrid";
import Loader from "./components/Loader";
import hamster from "./assets/hamster.png";
import hamster2 from "./assets/hamster2.png";
import hamster3 from "./assets/hamster3.png";
import "./App.css";

// ✅ 공통 베이스 URL (항상 /api 붙이기)
const RAW = (process.env.REACT_APP_API_URL || "http://localhost:8000").trim().replace(/\/+$/, "");
const API_URL = RAW.endsWith("/api") ? RAW : `${RAW}/api`;

/* JSON 강제 검사 fetch */
const fetchJSON = async (url, options = {}, ms = 10000) => {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(url, {
      ...options,
      signal: ctrl.signal,
      mode: "cors",
      credentials: "omit",
      cache: "no-store",
      redirect: "follow",
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const text = await res.text();

    if (res.ok && ct.includes("application/json")) return JSON.parse(text);
    if (ct.includes("application/json")) {
      let payload; try { payload = JSON.parse(text); } catch {}
      const msg = payload?.detail || payload?.message || text.slice(0,160);
      throw new Error(`HTTP ${res.status}: ${msg}`);
    }

    if (ct.includes("text/html")) {
      const [base, qs] = url.split("?", 2);
      const flippedBase = base.endsWith("/") ? base.slice(0, -1) : `${base}/`;
      const alt = qs ? `${flippedBase}?${qs}` : flippedBase;

      const r2 = await fetch(alt, {
        ...options,
        signal: ctrl.signal,
        mode: "cors",
        credentials: "omit",
        cache: "no-store",
        redirect: "follow",
        headers: { Accept: "application/json", ...(options.headers || {}) },
      });
      const ct2 = (r2.headers.get("content-type") || "").toLowerCase();
      const tx2 = await r2.text();

      if (r2.ok && ct2.includes("application/json")) return JSON.parse(tx2);
      if (ct2.includes("application/json")) {
        let payload; try { payload = JSON.parse(tx2); } catch {}
        const msg = payload?.detail || payload?.message || tx2.slice(0,160);
        throw new Error(`HTTP ${r2.status}: ${msg}`);
      }
      throw new Error(`JSON 아님: ${ct2 || "unknown"}`);
    }

    throw new Error(`HTTP ${res.status} ${text.slice(0,160)}…`);
  } finally {
    clearTimeout(t);
  }
};

/* 검색 결과 카드 */
function ResultsView({ data, upgrading, onBack }) {
  const items = Array.isArray(data?.trend_articles) ? data.trend_articles : [];
  return (
    <div style={{ maxWidth: 980, margin: "24px auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        {/* A안 적용: CSS로 숨길 수 있게 클래스 부여 */}
        <button
          className="results-back-btn"
          onClick={onBack}
          style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #ddd", background: "#fff", cursor: "pointer" }}
        >
          ← 돌아가기
        </button>
        <h3 style={{ margin: 0 }}>
          {data?.initial_keyword || "검색"} 뉴스 요약
          {data?.refined_keyword ? (
            <span style={{ marginLeft: 8, color: "#888", fontWeight: 400 }}>
              (추천 키워드: {data.refined_keyword})
            </span>
          ) : null}
        </h3>
        {upgrading && <span style={{ marginLeft: "auto", fontSize: 12, color: "#888" }}>요약 업그레이드 중…</span>}
      </div>

      {data?.trend_digest ? (
        <div style={{ background: "#fff7e6", border: "1px solid #ffe0a3", padding: 14, borderRadius: 12, marginBottom: 14 }}>
          <strong>✨ {data?.purpose || "뉴스 요약 트렌드 요약"}</strong>
          <div style={{ marginTop: 6, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{data.trend_digest}</div>
        </div>
      ) : null}

      <div className="cards-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        {items.map((it, i) => (
          <article key={it.url || i} style={{ border: "1px solid #eee", borderRadius: 12, padding: 16, background: "#fff" }}>
            <div style={{ fontSize: 13, color: "#999", marginBottom: 6 }}>📰 기사</div>
            <h4 style={{ fontSize: 16, margin: "0 0 8px" }}>
              {it.url ? (
                <a href={it.url} target="_blank" rel="noreferrer">
                  {it.title || "(제목 없음)"}
                </a>
              ) : (
                it.title || "(제목 없음)"
              )}
            </h4>
            <p style={{ fontSize: 14, lineHeight: 1.45, margin: 0 }}>{it.summary || "요약을 준비 중입니다."}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [upgrading, setUpgrading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [popular, setPopular] = useState([]);

  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const heroInputRef = useRef(null);

  useEffect(() => {
    const saved = localStorage.getItem("chat-log");
    if (saved) setMessages(JSON.parse(saved));
  }, []);
  useEffect(() => {
    localStorage.setItem("chat-log", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    (async () => {
      try {
        const data = await fetchJSON(`${API_URL}/popular_keywords/`, {}, 8000);
        setPopular(Array.isArray(data) ? data.slice(0, 8) : []);
      } catch {}
    })();
  }, []);

  const playAudio = async (text) => {
    if (!text?.trim()) return;
    try {
      const res = await fetch(`${API_URL}/generate-tts/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const blob = await res.blob();
      new Audio(URL.createObjectURL(blob)).play();
    } catch {}
  };

  // ✅ fastPreview 옵션 (기본 false)
  const handleSearch = async (query, opts = {}) => {
    const { fastPreview = false } = opts;
    if (!query?.trim()) return;

    setLoading(true);
    setUpgrading(true);

    // (즉시 3번 레이아웃) 플레이스홀더 push
    const placeholder = {
      initial_keyword: query,
      refined_keyword: "",
      purpose: "",
      trend_digest: "",
      trend_articles: [],
    };
    setMessages((prev) => [...prev, { query, results: placeholder, _upgrading: true }]);

    // 정식 요약(필수)
    const normalReq = fetchJSON(
      `${API_URL}/news_trend/`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keyword: query, mode: "normal" }),
      },
      35000
    );

    // 빠른 프리뷰(옵션)
    if (fastPreview) {
      fetchJSON(`${API_URL}/headline_quick?kw=${encodeURIComponent(query)}`, {}, 12000)
        .then((quick) => {
          setMessages((prev) => {
            if (!prev.length) return prev;
            const next = [...prev];
            const last = next[next.length - 1];
            if (Array.isArray(last.results?.trend_articles) && last.results.trend_articles.length === 0) {
              last.results.trend_articles = [
                { title: quick.title, url: quick.url, summary: quick.summary },
              ];
            }
            return next;
          });
        })
        .catch((e) => console.warn("[headline_quick 실패]", e.message));
    }

    try {
      const full = await normalReq;
      setMessages((prev) => {
        if (!prev.length) return prev;
        const next = [...prev];
        next[next.length - 1] = { query: full.initial_keyword, results: full };
        return next;
      });
      if (full.trend_digest) playAudio(full.trend_digest);
    } catch (e) {
      console.warn("[news_trend 실패]", e.message);
    } finally {
      setLoading(false);
      setUpgrading(false);
    }
  };

  const submitHeroSearch = (e) => {
    e.preventDefault();
    const v = heroInputRef.current?.value || "";
    // 필요하면 여기만 fastPreview: true 로 바꾸기
    handleSearch(v, { fastPreview: false });
  };

  const startRecording = async () => {
    setRecording(true);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;
    chunksRef.current = [];
    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = async () => {
      setRecording(false);
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const form = new FormData();
      form.append("file", blob, "rec.webm");
      try {
        const res = await fetch(`${API_URL}/generate-stt/`, { method: "POST", body: form });
        const { text } = await res.json();
        if (heroInputRef.current) heroInputRef.current.value = text || "";
        handleSearch(text, { fastPreview: false });
      } catch {}
    };
    recorder.start();
  };
  const stopRecording = () => recorderRef.current?.stop();

  const onHeroMicClick = () => (recording ? stopRecording() : startRecording());
  const onHeroSearchClick = () => heroInputRef.current?.focus();

  // ✅ 📰 버튼 → 메인으로
  const onHeroPaperClick = () => {
    setMessages([]);
    setUpgrading(false);
    setLoading(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const keywords = useMemo(
    () => (popular?.length ? popular.map((p) => p.keyword) : ["AI", "코인", "주식", "테슬라", "부동산"]).filter(Boolean),
    [popular]
  );

  const latest = messages.length ? messages[messages.length - 1] : null;
  const headerTitle = "오늘의 주요 뉴스를 알아볼까요?";

  return (
    <div className="app">
      <main className="main-content">
        <div className="center-area">
          <div className="news-hero">
            <div className="news-hero__masthead">
              <span className="news-hero__logo">🗞️</span>
              <span className="news-hero__title">{headerTitle}</span>
              <span className="news-hero__live">LIVE</span>
            </div>

            <div className="news-hero__desk">
              <img src={loading ? hamster2 : hamster} alt="hamster anchor" className="news-hero__hamster" />
              <div className="news-hero__props">
                <button className="prop" title={recording ? "녹음 중지" : "음성으로 검색"} onClick={onHeroMicClick}>🎙️</button>
                <button className="prop" title="검색 입력창 포커스" onClick={onHeroSearchClick}>🔍</button>
                {/* 제목 변경: 메인으로 */}
                <button className="prop" title="메인으로" onClick={onHeroPaperClick}>📰</button>
              </div>

              <form className="news-hero__search" onSubmit={submitHeroSearch} title={headerTitle}>
                <input ref={heroInputRef} type="text" className="hero-search-input" placeholder="무엇이든 물어보세요" />
                <button className="hero-search-btn" type="submit">검색</button>
                {upgrading && <span style={{ marginLeft: 12, fontSize: 12, color: "#888" }}>요약 업그레이드 중…</span>}
              </form>
            </div>

            <div className="news-ticker news-ticker--fixed" aria-label="실시간 키워드">
              <div className="news-ticker__track">
                {keywords.map((k) => (
                  <button
                    key={k}
                    className="ticker-chip"
                    type="button"
                    onClick={() => handleSearch(k, { fastPreview: false })}
                    title={`${k} 검색`}
                  >
                    🔥 {k}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {recording ? (
            <Loader image={hamster3} text="듣고 있어요" />
          ) : latest ? (
            <ResultsView data={latest.results} upgrading={upgrading} onBack={() => setMessages([])} />
          ) : (
            <HeadlineGrid keywords={keywords} onSearch={(kw) => handleSearch(kw, { fastPreview: false })} />
          )}
        </div>
      </main>
    </div>
  );
}

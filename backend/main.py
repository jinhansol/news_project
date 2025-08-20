# main.py — FastAPI backend (with /api prefix + Dual Query + KoBERT-only optional)
# - React 프런트는 베이스를 http://localhost:8000/api 로 맞춰 호출하세요.
# - KoBERT만 설치하면( transformers, torch ) 자동 사용되고, 미설치면 경량 폴백으로 동작합니다.
# - STT/TTS는 OpenAI API를 사용합니다. (melotts 등 불필요)

import os
import io
import re
import ssl
import time
import json
import hashlib
import tempfile
import asyncio
import httpx
from collections import Counter
from datetime import date, timedelta
from typing import List, Dict, Optional, Union

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# 로컬 유틸 (프로젝트에 존재해야 합니다)
from voice_chat import generate_audio_file          # (경로 or bytes 반환)
from patched_cleanre import clean_text

# OpenAI / LangChain
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# 환경변수 & 클라이언트 준비
# =============================================================================
load_dotenv()

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
if not all([OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET]):
    raise ValueError("OPENAI_API_KEY / NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 를 .env에 설정하세요.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

ssl_ctx = ssl.create_default_context()
# 일부 구형 TLS 사이트 호환
ssl_ctx.set_ciphers("DEFAULT@SECLEVEL=1")

# 리다이렉트 허용
async_http = httpx.AsyncClient(timeout=30.0, verify=ssl_ctx, follow_redirects=True)


# =============================================================================
# FastAPI & CORS
# =============================================================================
app = FastAPI()
app.router.redirect_slashes = True  # /path 와 /path/ 모두 허용

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모든 API는 /api 아래로!
api = APIRouter(prefix="/api")


# =============================================================================
# 공용 유틸
# =============================================================================
def strip_html(html_string: str) -> str:
    if not html_string:
        return ""
    return BeautifulSoup(html_string, "html.parser").get_text(strip=True)

def purpose_from_keyword(keyword: str) -> str:
    table = {
        "코인": "가상자산 뉴스 요약",
        "주식": "주식 시장 트렌드 요약",
        "부동산": "부동산 뉴스 요약",
        "AI":   "인공지능 트렌드 요약",
        "취업": "취업 시장 동향 요약",
    }
    for k, p in table.items():
        if k in keyword:
            return p
    return f"{keyword} 관련 뉴스 요약"

def _node_text_len(node) -> int:
    if not node:
        return 0
    for bad in node.select("script, style, noscript, header, footer, nav, aside, form, iframe"):
        bad.decompose()
    return len(BeautifulSoup(str(node), "html.parser").get_text(" ", strip=True))

def _node_text(node) -> str:
    if not node:
        return ""
    for bad in node.select("script, style, noscript, header, footer, nav, aside, form, iframe"):
        bad.decompose()
    text = BeautifulSoup(str(node), "html.parser").get_text(" ", strip=True)
    return " ".join(text.split())

def ensure_bytes_from_generate_audio(result: Union[str, bytes, bytearray]) -> bytes:
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)
    if isinstance(result, str):
        if not os.path.exists(result):
            raise RuntimeError(f"TTS 파일을 찾지 못했습니다: {result}")
        with open(result, "rb") as f:
            return f.read()
    raise RuntimeError("지원하지 않는 TTS 반환 형태")


# =============================================================================
# 경량 한국어 토큰화(설치 불필요) + 단순 재정렬(폴백)
# =============================================================================
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
KOREAN_STOPWORDS = {
    "기자","사진","영상","네이버","언론","일보","뉴스","신문","연합뉴스","속보",
    "그리고","하지만","그러나","때문","대해","관련","최근","오늘","이번",
    "대한","에서는","에서","으로","하다","했다","했다가","이라","이다",
    "개월","오늘","내일","지난","이번","오는","현재","지난해","올해"
}

def tokenize_ko(text: str) -> List[str]:
    text = text or ""
    toks = [t for t in TOKEN_RE.findall(text)]
    return [t for t in toks if len(t) >= 2 and t not in KOREAN_STOPWORDS]

def count_keywords(texts: List[str]) -> Counter:
    c = Counter()
    for t in texts:
        for w in tokenize_ko(t):
            c[w] += 1
    return c

def rerank_links_simple(links: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """
    KoBERT가 없을 때 쓰는 폴백 재정렬:
      - 쿼리 토큰과 제목 토큰 교집합 크기
      - 제목 길이 패널티(짧은 제목 가점)
    """
    if not links:
        return []
    qset = set(tokenize_ko(query))
    scored = []
    for l in links:
        title = l.get("title") or ""
        tset = set(tokenize_ko(title))
        overlap = len(qset & tset)
        length_penalty = 0.15 * max(0, len(tset) - 8)
        score = overlap - length_penalty
        scored.append((score, l))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in scored[:top_k]]


# =============================================================================
# KoBERT ONLY (konlpy 없이) — 설치되어 있으면 자동 활성화
# =============================================================================
import importlib
KOBERT_AVAILABLE = False
try:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModel = getattr(transformers, "AutoModel")
    _kobert_device = "cuda" if torch.cuda.is_available() else "cpu"
    _kobert_tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    _kobert_model = AutoModel.from_pretrained("skt/kobert-base-v1").to(_kobert_device)
    _kobert_model.eval()
    KOBERT_AVAILABLE = True
except Exception:
    _kobert_device = "cpu"
    _kobert_tokenizer = None
    _kobert_model = None
    KOBERT_AVAILABLE = False

def _kobert_embed(texts: List[str], max_len: int = 128):
    if not KOBERT_AVAILABLE:
        return None
    with torch.no_grad():
        toks = _kobert_tokenizer(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        toks = {k: v.to(_kobert_device) for k, v in toks.items()}
        out = _kobert_model(**toks).last_hidden_state[:, 0, :]  # [CLS]
        out = torch.nn.functional.normalize(out, p=2, dim=1)    # L2 정규화 -> 코사인 내적 사용
        return out.cpu()

def rerank_links_kobert(links: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    if not links or not KOBERT_AVAILABLE:
        return links[:top_k]
    titles = [l["title"] for l in links]
    q = _kobert_embed([query])
    t = _kobert_embed(titles)
    if q is None or t is None:
        return links[:top_k]
    sims = (t @ q[0].unsqueeze(1)).squeeze(1)  # 정규화 내적 = 코사인
    order = torch.argsort(sims, descending=True).tolist()
    return [links[i] for i in order[:top_k]]

def _generate_candidates(tokens: List[str], ngram_range=(1,3), max_candidates=1200):
    cands = []
    n1, n2 = ngram_range
    for n in range(n1, n2+1):
        for i in range(0, len(tokens)-n+1):
            phrase = " ".join(tokens[i:i+n])
            if len(phrase) < 2:
                continue
            cands.append(phrase)
            if len(cands) >= max_candidates:
                return list(dict.fromkeys(cands))  # 순서 유지 중복 제거
    return list(dict.fromkeys(cands))

def extract_keywords_kobert(docs: List[str], top_k: int = 12, ngram_range=(1,3)) -> List[tuple]:
    """
    KeyBERT 스타일: 문서 임베딩(KoBERT CLS) vs 후보 n-gram 임베딩 코사인 유사도.
    konlpy 없이 regex 토큰으로 후보 생성, 임베딩은 KoBERT만 사용.
    """
    if not KOBERT_AVAILABLE:
        return []
    doc_text = " ".join(docs)[:4000]
    doc_emb = _kobert_embed([doc_text])
    if doc_emb is None:
        return []

    tokens: List[str] = []
    for d in docs:
        tokens.extend(tokenize_ko(d))
    candidates = _generate_candidates(tokens, ngram_range=ngram_range, max_candidates=1200)
    if not candidates:
        return []

    # 배치 임베딩 후 유사도 계산
    scores = []
    bs = 64
    for i in range(0, len(candidates), bs):
        emb = _kobert_embed(candidates[i:i+bs])
        if emb is None:
            continue
        sims = (emb @ doc_emb[0].unsqueeze(1)).squeeze(1).tolist()
        scores.extend(sims)

    # 후보별 최고 점수 유지 후 상위 top_k
    best = {}
    for cand, sc in zip(candidates, scores):
        if cand not in best or sc > best[cand]:
            best[cand] = sc
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked  # [(phrase, score), ...]


# =============================================================================
# 모델 & 체인 (LLM 프롬프트)
# =============================================================================
class NewsTrendRequest(BaseModel):
    keyword: str

class TTSRequest(BaseModel):
    text: str

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)
output_parser = StrOutputParser()

# HTML → 본문 추출 (LLM 폴백)
extraction_chain = (
    PromptTemplate.from_template(
        "You are an expert web scraper. Your task is to extract the main article text from the given HTML content. "
        "Ignore all HTML tags, scripts, styles, advertisements, navigation bars, headers, footers, related links, and comments. "
        "Return only the clean, plain text of the main news article in Korean.\n\n"
        "## HTML Content:\n{html_content}"
    ) | llm | output_parser
)

# 이중 쿼리 생성: q1(포괄) + q2(구체)
dual_query_chain = (
    PromptTemplate.from_template(
        "너는 한국어 뉴스 검색 전략가야. 사용자의 원문 질의와 초기 기사에서 추출한 주요 키워드를 바탕으로 "
        "1) 포괄적인 1차 검색어(q1)와 2) 구체적이고 액션 중심의 2차 검색어(q2)를 만들어라.\n"
        "- q1은 주제를 넓게 포착해 상위 기사들을 수집하기 좋게 만든다.\n"
        "- q2는 시기(최근/올해/지난달 등), 대상(국가/기업/인물), 행위(협상/합의/인하/제재/발표 등), 품목(철강/자동차/반도체 등)을 명시한다.\n"
        "출력은 JSON만 허용한다. 다른 설명 금지.\n"
        '{{\"q1\":\"...\",\"q2\":\"...\"}}\n\n'
        "## 사용자 질의\n{user_query}\n\n"
        "## 상위 키워드(빈도순)\n{top_keywords}\n"
    ) | llm | output_parser
)

# 기사 요약
summary_chain = (
    PromptTemplate.from_template(
        "너는 한국어 뉴스 기사를 2~3문장으로 요약하는 AI야.\n요약 목적: {purpose}\n뉴스 기사 원문: {article}"
    ) | llm | output_parser
)

# 트렌드 종합
trend_chain = (
    PromptTemplate.from_template(
        "다음은 '{purpose}'와 관련된 여러 기사 요약들입니다. 이를 바탕으로 최신 트렌드를 3~5줄로 종합해줘.\n\n## 기사 요약들:\n{summaries}"
    ) | llm | output_parser
)


# =============================================================================
# 네이버 API + 본문 추출
# =============================================================================
async def search_news_titles(keyword: str, max_items: int = 3) -> List[Dict]:
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": keyword, "display": max_items, "start": 1, "sort": "date"}
    try:
        res = await async_http.get(url, headers=headers, params=params)
        res.raise_for_status()
        items = res.json().get("items", [])
        out = []
        seen = set()
        for it in items:
            title = strip_html(it.get("title", ""))
            link = it.get("link") or ""
            if not title or not link or link in seen:
                continue
            seen.add(link)
            out.append({"title": title, "url": link})
        return out
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="네이버 뉴스 API 호출 실패")
    except Exception:
        raise HTTPException(status_code=500, detail="뉴스 검색 중 서버 오류")

async def extract_article_text(url: str, sem: asyncio.Semaphore) -> str:
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
        }
        res = await async_http.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")

        known = [
            "#dic_area", "#articleBodyContents", ".article_body",
            ".newsct_body", ".article_view_body", "article"
        ]
        content = next((soup.select_one(s) for s in known if soup.select_one(s)), None)

        if content and _node_text_len(content) >= 300:
            return _node_text(content)

        if not content:
            for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "iframe"]):
                tag.decompose()
            body = soup.body
            if not body:
                return ""
            if _node_text_len(body) >= 300:
                return _node_text(body)
            html_snippet = str(body)
        else:
            html_snippet = str(content)

        MAX = 15000
        if len(html_snippet) > MAX:
            chunks = [html_snippet[i:i+MAX] for i in range(0, len(html_snippet), MAX)]
            out_parts = []
            for ch in chunks:
                async with sem:
                    out_parts.append(await extraction_chain.ainvoke({"html_content": ch}))
            return "\n".join(out_parts)
        else:
            async with sem:
                return await extraction_chain.ainvoke({"html_content": html_snippet})
    except Exception:
        return ""


# =============================================================================
# 인기 키워드 (네이버 DataLab)
# =============================================================================
DEFAULT_KEYWORD_GROUPS = [
    {"groupName": "코인",   "keywords": ["코인", "비트코인", "이더리움"]},
    {"groupName": "부동산", "keywords": ["부동산", "아파트", "전세"]},
    {"groupName": "주식",   "keywords": ["주식", "코스피", "코스닥"]},
    {"groupName": "AI",    "keywords": ["AI", "인공지능", "챗GPT"]},
    {"groupName": "테슬라", "keywords": ["테슬라", "일론 머스크"]},
]

async def fetch_popular_keywords(days: int = 7, top_k: int = 8, groups: Optional[List[Dict]] = None) -> List[Dict]:
    groups = groups or DEFAULT_KEYWORD_GROUPS
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = date.today().strftime("%Y-%m-%d")

    url = "https://openapi.naver.com/v1/datalab/search"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        "Content-Type": "application/json",
    }
    payload = {"startDate": start, "endDate": end, "timeUnit": "date",
               "keywordGroups": groups, "device": "", "ages": [], "gender": ""}

    try:
        res = await async_http.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        results = []
        for item in data.get("results", []):
            score = sum(d.get("ratio", 0.0) for d in item.get("data", []))
            results.append({"keyword": item.get("title"), "score": round(score, 2)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="인기 키워드 조회 실패")
    except Exception:
        raise HTTPException(status_code=500, detail="인기 키워드 조회 중 서버 오류")


# =============================================================================
# API 엔드포인트 (/api/*)
# =============================================================================
@api.get("/headline_quick")
async def headline_quick(kw: str = Query(..., min_length=1)):
    """상단 카드용 빠른 한 건 요약 — 상위 N개 중 10분 단위 회전."""
    items = await search_news_titles(kw, max_items=5)
    if not items:
        raise HTTPException(status_code=404, detail="관련 뉴스 없음")

    # 10분 단위로 키워드별 회전 (항상 같은 기사 방지)
    seed = f"{kw}:{int(time.time() // 600)}"
    idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(items)
    pick = items[idx]

    sem = asyncio.Semaphore(4)
    article = await extract_article_text(pick["url"], sem)
    article_clean = clean_text(article, "KR") if article else ""

    # 오탐 가드(본문 부실 / 프롬프트 잔향)
    suspicious = ["You are an expert web scraper", "HTML", "본문을 추출", "네비게이션", "헤더", "푸터"]
    if (not article_clean) or (len(article_clean) < 200) or any(s in article_clean for s in suspicious):
        return {"title": pick["title"], "url": pick["url"], "summary": "요약 실패: 본문을 추출할 수 없습니다."}

    summary = await summary_chain.ainvoke({"purpose": purpose_from_keyword(kw), "article": article_clean})
    return {"title": pick["title"], "url": pick["url"], "summary": summary}


@api.post("/news_trend")
async def news_trend(req: NewsTrendRequest):
    """
    강화된 이중 쿼리 파이프라인(추가 설치 없이 동작, KoBERT 있으면 자동 사용):
      1) q1(포괄) = 사용자 원문 → q1 검색(최대 8건) → 일부 기사 본문 추출 → 경량 토큰화 기반 키워드 집계
      2) (q1 결과의 Top 키워드/KOBERT 키워드) + 원질의 → LLM이 q1/q2(JSON) 생성
      3) q2 검색(최대 8건) → KoBERT 재정렬(가능시) / 경량 토큰 교집합 재정렬(폴백) → 상위 3건 본문 요약
      4) 트렌드 종합 + 키워드 통계 반환
    """
    user_query = (req.keyword or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="keyword is required")

    sem = asyncio.Semaphore(6)

    # (A) 1차 쿼리
    q1_seed = user_query
    q1_links = await search_news_titles(q1_seed, max_items=8)
    if not q1_links:
        raise HTTPException(status_code=404, detail="관련 뉴스를 찾을 수 없습니다.")

    # 1차 기사 샘플 (최대 5건)
    sample_targets = q1_links[:5]
    q1_articles = await asyncio.gather(*[extract_article_text(l["url"], sem) for l in sample_targets])

    cleaned_samples = [clean_text(a, "KR") for a in q1_articles if a]

    # 빈도 기반 키워드
    kw_counter = count_keywords(cleaned_samples)
    topkw = kw_counter.most_common(12)

    # KoBERT 추출 키워드(설치된 경우)
    kobert_keywords = extract_keywords_kobert(cleaned_samples, top_k=12) if KOBERT_AVAILABLE else []

    # (B) LLM으로 q1/q2 생성 — KoBERT 키워드 우선 반영
    top_for_dual = (
        ", ".join([k for k, _ in kobert_keywords[:10]]) or
        ", ".join([w for w, _ in topkw[:10]]) or
        "(없음)"
    )
    dq_json = await dual_query_chain.ainvoke({
        "user_query": user_query,
        "top_keywords": top_for_dual
    })
    try:
        d = json.loads(dq_json)
        q1 = (d.get("q1") or q1_seed).strip()
        q2 = (d.get("q2") or q1_seed).strip()
    except Exception:
        q1 = q1_seed
        q2 = q1_seed

    # (C) 2차 검색 + 재정렬
    q2_links_raw = await search_news_titles(q2, max_items=8)
    if KOBERT_AVAILABLE and q2_links_raw:
        q2_links = rerank_links_kobert(q2_links_raw, q2, top_k=5)
    else:
        q2_links = rerank_links_simple(q2_links_raw or [], q2, top_k=5)

    if not q2_links:
        if KOBERT_AVAILABLE:
            q2_links = rerank_links_kobert(q1_links, q1, top_k=5)
        else:
            q2_links = rerank_links_simple(q1_links, q1, top_k=5)

    final_pick = (q2_links or q1_links)[:3]

    # 본문 추출 → 요약
    final_articles = await asyncio.gather(*[extract_article_text(l["url"], sem) for l in final_pick])
    purpose = purpose_from_keyword(f"{user_query}, {q2}")

    summaries = await asyncio.gather(
        *[
            summary_chain.ainvoke({"purpose": purpose, "article": clean_text(a, "KR")})
            if a and a.strip()
            else asyncio.sleep(0, result="요약 실패: 본문을 추출할 수 없습니다.")
            for a in final_articles
        ]
    )
    details = [{"title": link["title"], "url": link["url"], "summary": summ}
               for link, summ in zip(final_pick, summaries)]

    trend_input = "\n\n".join(s for s in summaries if "요약 실패" not in s)
    trend = await trend_chain.ainvoke({"purpose": purpose, "summaries": trend_input}) if trend_input else ""
    if (not trend.strip()) and not any("요약 실패" not in d["summary"] for d in details):
        trend = "관련된 최신 뉴스를 찾았지만, 내용을 요약하는 데 어려움이 있었습니다. 다른 키워드로 다시 시도해 보세요."

    return {
        "initial_keyword": user_query,
        "q1": q1,
        "q2": q2,
        "refined_keyword": q2,  # 프론트 호환
        "purpose": purpose,
        "trend_digest": trend,
        "trend_articles": details,
        "keyword_stats": [{"token": w, "count": c} for w, c in topkw],
        "keyword_extractive": (
            [{"token": k, "score": round(float(s), 4)} for k, s in kobert_keywords] if kobert_keywords else []
        ),
        "debug": {
            "rerank": "kobert" if KOBERT_AVAILABLE else "token-overlap",
            "kobert": KOBERT_AVAILABLE
        }
    }


@api.get("/popular_keywords")
async def popular_keywords(days: int = 7, top_k: int = 8) -> List[Dict]:
    return await fetch_popular_keywords(days=days, top_k=top_k, groups=DEFAULT_KEYWORD_GROUPS)


@api.post("/generate-tts")
async def generate_tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        result = generate_audio_file(text)         # path or bytes
        audio_bytes = ensure_bytes_from_generate_audio(result)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg",
                                 headers={"Cache-Control": "no-store"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 오류: {e}")


@api.post("/generate-stt")
async def generate_stt(file: UploadFile = File(...)):
    try:
        content = await file.read()
        suffix = os.path.splitext(file.filename or "")[-1] or ".webm"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(content); tmp.flush(); tmp.close()

        result = openai_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=open(tmp.name, "rb"),
        )
        text = getattr(result, "text", "") or ""
        os.unlink(tmp.name)
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


# 라우터 등록
app.include_router(api)


# =============================================================================
# 종료 훅
# =============================================================================
@app.on_event("shutdown")
async def _shutdown():
    await async_http.aclose()

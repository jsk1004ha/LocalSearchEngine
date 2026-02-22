from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .engine import InvestigationEngine
from .modes import SearchMode
from .result_payload import result_to_payload


@dataclass(frozen=True)
class WebUiConfig:
    title: str = "Investigation Search Web"
    default_mode: str = "investigation"
    default_top_k: int = 5
    default_time_budget_sec: int = 120
    default_max_items: int = 8
    default_show_diagnostics: bool = False


@dataclass
class _AppState:
    engine: InvestigationEngine
    config: WebUiConfig
    lock: threading.Lock


def run_web_ui(
    engine: InvestigationEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    config: WebUiConfig | None = None,
) -> None:
    state = _AppState(engine=engine, config=_normalize_config(config or WebUiConfig()), lock=threading.Lock())
    handler = _build_handler(state)
    with ThreadingHTTPServer((host, int(port)), handler) as httpd:
        sa = httpd.socket.getsockname()
        print(f"Web UI running on http://{sa[0]}:{sa[1]}/ (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def _build_handler(state: _AppState):
    class _Handler(BaseHTTPRequestHandler):
        server_version = "InvestigationSearchWeb/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send_bytes(
                    HTTPStatus.OK,
                    _render_index_html(state.config).encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return
            if path == "/api/config":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "config": _public_config(state.config), "modes": [m.value for m in SearchMode]},
                )
                return
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "status": "healthy"})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/search":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": {"message": "not found"}})
                return

            try:
                body = self._read_json_body(max_bytes=1_000_000)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": {"message": str(exc)}})
                return

            query = str(body.get("query", "")).strip()
            if not query:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": {"message": "query is required"}})
                return

            valid_modes = {m.value for m in SearchMode}
            mode = str(body.get("mode") or state.config.default_mode).strip().lower()
            if mode not in valid_modes:
                mode = state.config.default_mode

            top_k = _coerce_int(body.get("top_k_per_pass"), state.config.default_top_k, lo=1, hi=30)
            budget = _coerce_int(body.get("time_budget_sec"), state.config.default_time_budget_sec, lo=1, hi=600)
            max_items = _coerce_int(body.get("max_items"), state.config.default_max_items, lo=1, hi=30)
            include_diagnostics = bool(body.get("include_diagnostics", state.config.default_show_diagnostics))

            try:
                with state.lock:
                    result = state.engine.search(query, top_k_per_pass=top_k, time_budget_sec=budget, mode=mode)
            except Exception as exc:  # pragma: no cover - runtime path
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": {"message": f"{type(exc).__name__}: {exc}"}},
                )
                return

            payload = result_to_payload(
                result,
                query=query,
                mode=mode,
                top_k_per_pass=top_k,
                time_budget_sec=budget,
                max_items=max_items,
                include_diagnostics=include_diagnostics,
            )
            session_id = result.diagnostics.get("knowledge_library_session_id")
            if session_id:
                payload["knowledge_library_session_id"] = session_id
            self._send_json(HTTPStatus.OK, {"ok": True, "result": payload})

        def _read_json_body(self, *, max_bytes: int) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "").strip()
            if not raw_length:
                return {}
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid content length") from exc
            if length < 0 or length > max_bytes:
                raise ValueError("payload too large")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError("invalid json payload") from exc
            if not isinstance(payload, dict):
                raise ValueError("json payload must be an object")
            return payload

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self._send_bytes(status, raw, content_type="application/json; charset=utf-8")

        def _send_bytes(self, status: HTTPStatus, data: bytes, *, content_type: str) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler


def _normalize_config(cfg: WebUiConfig) -> WebUiConfig:
    modes = {m.value for m in SearchMode}
    mode = str(cfg.default_mode).strip().lower()
    if mode not in modes:
        mode = "investigation"
    return WebUiConfig(
        title=str(cfg.title).strip() or "Investigation Search Web",
        default_mode=mode,
        default_top_k=max(1, min(int(cfg.default_top_k), 30)),
        default_time_budget_sec=max(1, min(int(cfg.default_time_budget_sec), 600)),
        default_max_items=max(1, min(int(cfg.default_max_items), 30)),
        default_show_diagnostics=bool(cfg.default_show_diagnostics),
    )


def _public_config(cfg: WebUiConfig) -> dict[str, Any]:
    return {
        "title": cfg.title,
        "default_mode": cfg.default_mode,
        "default_top_k": cfg.default_top_k,
        "default_time_budget_sec": cfg.default_time_budget_sec,
        "default_max_items": cfg.default_max_items,
        "default_show_diagnostics": cfg.default_show_diagnostics,
    }


def _coerce_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = int(default)
    return max(lo, min(hi, n))


def _render_index_html(config: WebUiConfig) -> str:
    bootstrap = {
        "title": config.title,
        "default_mode": config.default_mode,
        "default_top_k": config.default_top_k,
        "default_time_budget_sec": config.default_time_budget_sec,
        "default_max_items": config.default_max_items,
        "default_show_diagnostics": config.default_show_diagnostics,
    }
    return _INDEX_HTML_TEMPLATE.replace("__BOOTSTRAP__", json.dumps(bootstrap, ensure_ascii=False))


_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Investigation Search Web</title>
<style>
:root{--bg:#08161e;--bg2:#122732;--fg:#ebf6f1;--muted:#9db9b1;--line:rgba(158,227,219,.28);--glass:rgba(8,28,36,.74);--mint:#58d9b1;--coral:#ff9f78;--warn:#ffd37a;--bad:#ff7b7b}
*{box-sizing:border-box}html,body{margin:0;min-height:100%}
body{font-family:"SUIT Variable","Pretendard Variable","IBM Plex Sans KR","Noto Sans KR","Malgun Gothic",sans-serif;color:var(--fg);background:radial-gradient(1200px 700px at 8% -10%,rgba(88,217,177,.22),transparent 55%),radial-gradient(1000px 620px at 102% 10%,rgba(255,159,120,.22),transparent 54%),linear-gradient(160deg,var(--bg),var(--bg2));padding:16px}
.aurora{position:fixed;width:34vmax;height:34vmax;filter:blur(20px);border-radius:50%;opacity:.28;z-index:0;animation:float 9s ease-in-out infinite}
.a{left:-10vmax;top:-9vmax;background:radial-gradient(circle,#58d9b1,transparent 72%)}.b{right:-9vmax;bottom:-11vmax;background:radial-gradient(circle,#ff9f78,transparent 72%);animation-delay:1.2s}
.shell{position:relative;z-index:1;max-width:1180px;margin:0 auto}
.panel{border:1px solid var(--line);border-radius:20px;background:var(--glass);backdrop-filter:blur(15px);box-shadow:0 20px 48px rgba(0,0,0,.38)}
.hero{padding:18px;display:grid;grid-template-columns:1.6fr 1fr;gap:12px;margin-bottom:12px}
.eyebrow{margin:0;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em}.title{margin:.35rem 0 0;font-size:clamp(1.35rem,2.5vw,1.95rem)}.desc{margin:.55rem 0 0;color:#cfe3dc}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:9px}.stat{border:1px solid rgba(158,227,219,.18);border-radius:13px;padding:9px;background:rgba(4,16,23,.45)}.stat b{display:block;color:var(--muted);font-size:11px}.stat span{display:block;margin-top:4px;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:12px}.card{padding:14px}.search{grid-column:span 7}.plan{grid-column:span 5}.answer{grid-column:span 7}.summary{grid-column:span 5}.evidence{grid-column:span 8}.contra{grid-column:span 4}.diag{grid-column:span 12}
h2{margin:0 0 8px;font-size:1.02rem}.hint{margin:0 0 10px;color:#cfe3dc;font-size:.9rem}
form{display:grid;gap:10px}.row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
label{display:grid;gap:4px;font-size:.84rem;color:var(--muted)}
input,select,textarea{width:100%;border:1px solid rgba(177,224,213,.24);border-radius:12px;background:rgba(4,16,24,.66);color:var(--fg);padding:9px 11px;font:inherit}
textarea{min-height:70px;resize:vertical}.chk{display:flex;gap:8px;align-items:center;color:#cfe3dc;font-size:.9rem}
button{border:1px solid rgba(88,217,177,.8);border-radius:999px;background:linear-gradient(140deg,rgba(88,217,177,.22),rgba(88,217,177,.07));color:var(--fg);padding:9px 16px;font-weight:700;cursor:pointer}
button[disabled]{opacity:.6;cursor:wait}.actions{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.status{padding:6px 10px;border-radius:999px;font-size:.8rem;border:1px solid rgba(255,211,122,.4);color:#ffe7b2;background:rgba(255,211,122,.12)}.status.ok{border-color:rgba(88,217,177,.5);color:#bcf1de;background:rgba(88,217,177,.1)}.status.error{border-color:rgba(255,123,123,.5);color:#ffd2d2;background:rgba(255,123,123,.1)}
.list{display:grid;gap:8px;max-height:470px;overflow:auto;padding-right:4px}.item{border:1px solid rgba(158,227,219,.2);border-radius:12px;padding:10px;background:rgba(3,15,20,.5)}
.head{display:flex;justify-content:space-between;gap:7px;font-size:.78rem}.score{font-family:"IBM Plex Mono","JetBrains Mono","D2Coding",monospace;color:#cde4dd}
.badge{padding:3px 8px;border-radius:999px;font-size:.72rem;border:1px solid transparent;font-weight:700;text-transform:uppercase}.supports{border-color:rgba(88,217,177,.6);background:rgba(88,217,177,.14)}.contradicts{border-color:rgba(255,123,123,.55);background:rgba(255,123,123,.14)}.uncertain{border-color:rgba(255,211,122,.55);background:rgba(255,211,122,.14)}
.content{margin-top:6px;white-space:pre-wrap;word-break:break-word;font-size:.9rem}.meta{margin-top:6px;font-size:.76rem;color:var(--muted);display:flex;gap:6px;flex-wrap:wrap}.meta a{color:#a9ead7}
.chips{display:flex;gap:7px;flex-wrap:wrap}.chip{padding:5px 8px;border-radius:999px;border:1px solid rgba(158,227,219,.26);background:rgba(4,17,24,.45);font-size:.78rem;color:#cfe3dc}
.empty{border:1px dashed rgba(158,227,219,.35);border-radius:11px;padding:11px;color:var(--muted);font-size:.88rem}
details{border:1px solid rgba(158,227,219,.2);border-radius:12px;padding:8px;background:rgba(4,15,20,.5)}summary{cursor:pointer;color:#cfe3dc}
pre{margin:8px 0 0;max-height:320px;overflow:auto;border:1px solid rgba(158,227,219,.2);border-radius:9px;padding:10px;background:rgba(2,10,14,.8);font-size:.76rem;line-height:1.4;font-family:"IBM Plex Mono","JetBrains Mono","D2Coding",monospace}
.reveal{opacity:0;transform:translateY(14px);animation:rise .56s ease forwards}.grid .reveal:nth-child(1){animation-delay:.04s}.grid .reveal:nth-child(2){animation-delay:.1s}.grid .reveal:nth-child(3){animation-delay:.16s}.grid .reveal:nth-child(4){animation-delay:.22s}.grid .reveal:nth-child(5){animation-delay:.28s}.grid .reveal:nth-child(6){animation-delay:.34s}.grid .reveal:nth-child(7){animation-delay:.4s}
input:focus-visible,select:focus-visible,textarea:focus-visible,button:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid #9cebd7;outline-offset:2px}
@keyframes rise{to{opacity:1;transform:translateY(0)}}@keyframes float{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(16px) scale(1.06)}}
@media (max-width:1000px){.hero{grid-template-columns:1fr}.search,.plan,.answer,.summary,.evidence,.contra,.diag{grid-column:span 12}.row{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.row{grid-template-columns:1fr}.actions{flex-direction:column;align-items:stretch}button{width:100%}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
</style></head>
<body>
<div class="aurora a"></div><div class="aurora b"></div>
<main class="shell">
  <header class="panel hero reveal">
    <section><p class="eyebrow">Bento + Glass + Aurora</p><h1 id="title" class="title">Investigation Search Web</h1><p class="desc">문서의 스타일 가이드를 반영한 웹 검색 UI입니다. 계획-근거-반례-진단 흐름으로 결과를 제공합니다.</p></section>
    <section class="stats"><div class="stat"><b>Mode</b><span id="stat-mode">-</span></div><div class="stat"><b>Top-K</b><span id="stat-topk">-</span></div><div class="stat"><b>Budget</b><span id="stat-budget">-</span></div><div class="stat"><b>Status</b><span id="stat-health">ready</span></div></section>
  </header>
  <section class="grid">
    <article class="panel card search reveal" id="search-card"><h2>Search Console</h2><p class="hint">DSL 필터(`source:`, `doc:`, `after:`)와 모드를 함께 지정할 수 있습니다.</p><form id="form"><label>Query<textarea id="query" placeholder="예: AI 기반 UI 디자인 전략 최신 근거"></textarea></label><div class="row"><label>Mode<select id="mode"></select></label><label>Top-K<input id="topk" type="number" min="1" max="30"></label><label>Time Budget(sec)<input id="budget" type="number" min="1" max="600"></label><label>Max Items<input id="max-items" type="number" min="1" max="30"></label></div><label class="chk"><input id="diag-toggle" type="checkbox">진단 정보 포함</label><div class="actions"><button id="run" type="submit">검색 실행</button><div id="status" class="status" aria-live="polite">대기 중</div></div></form></article>
    <article class="panel card plan reveal"><h2>Action Plan</h2><div class="list"><div class="item"><strong>1) Parse</strong><div class="meta"><span>쿼리 의도/필터 파악</span></div></div><div class="item"><strong>2) Retrieve</strong><div class="meta"><span>하이브리드 검색 + 재순위화</span></div></div><div class="item"><strong>3) Synthesize</strong><div class="meta"><span>답변 + 근거 + 반례 + 진단</span></div></div></div></article>
    <article class="panel card answer reveal"><h2>Answer</h2><div id="answer" class="item">검색 결과가 여기에 표시됩니다.</div><div class="chips" id="answer-sources"></div></article>
    <article class="panel card summary reveal"><h2>Summary</h2><div class="chips"><span class="chip" id="sum-e">Evidence: 0</span><span class="chip" id="sum-c">Contradictions: 0</span><span class="chip" id="sum-s">Sources: 0</span><span class="chip" id="sum-k">Session: none</span></div></article>
    <article class="panel card evidence reveal"><h2>Evidence</h2><div id="evidence" class="list"><p class="empty">근거가 없습니다.</p></div></article>
    <article class="panel card contra reveal"><h2>Contradictions</h2><div id="contra" class="list"><p class="empty">반례가 없습니다.</p></div></article>
    <article class="panel card diag reveal"><h2>Diagnostics</h2><details id="diag-box"><summary>진단 JSON</summary><pre id="diag-json">{}</pre></details></article>
  </section>
</main>
<script>
window.__BOOTSTRAP__=__BOOTSTRAP__;
(()=>{const b=window.__BOOTSTRAP__||{},id=(x)=>document.getElementById(x),mode=id("mode"),query=id("query"),topk=id("topk"),budget=id("budget"),maxItems=id("max-items"),diagToggle=id("diag-toggle"),diagBox=id("diag-box"),status=id("status"),run=id("run"),title=id("title"),statMode=id("stat-mode"),statTopk=id("stat-topk"),statBudget=id("stat-budget"),statHealth=id("stat-health"),answer=id("answer"),answerSources=id("answer-sources"),evidence=id("evidence"),contra=id("contra"),diagJson=id("diag-json"),sumE=id("sum-e"),sumC=id("sum-c"),sumS=id("sum-s"),sumK=id("sum-k"),card=id("search-card");
const labels={investigation:"균형",reporter:"광범위",fbi:"OSINT",collection:"자료",sniper:"단일",rumor:"다주장",library:"신뢰",llm:"LLM"};
const esc=(v)=>String(v??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\"/g,"&quot;").replace(/'/g,"&#039;");
const n=(v,d,lo,hi)=>{const x=Math.round(Number(v));return Number.isFinite(x)?Math.max(lo,Math.min(hi,x)):d;};
const setStatus=(t,k="")=>{status.textContent=t;status.className="status"+(k?" "+k:"");};
const busy=(on)=>{run.disabled=on;card.style.opacity=on?"0.84":"1";};
const modes=["investigation","reporter","fbi","collection","sniper","rumor","library","llm"];
const renderRows=(rows,el,empty)=>{if(!Array.isArray(rows)||rows.length===0){el.innerHTML='<p class="empty">'+esc(empty)+"</p>";return;}el.innerHTML=rows.map((r,i)=>{const ev=r.evidence||{},v=String(r.verdict||"uncertain").toLowerCase(),cls=(v==="supports"||v==="contradicts")?v:"uncertain",sc=Number(r.score||0).toFixed(4),src=(r.source&&r.source.citation_id)||"none",md=ev.metadata||{},url=(typeof md.url==="string"&&md.url)?md.url:ev.doc_id,isHttp=/^https?:\\/\\//i.test(String(url||"")),host=isHttp?String(url).replace(/^https?:\\/\\//i,"").split("/")[0]:String(url||"local"),link=isHttp?'<a href="'+esc(url)+'" target="_blank" rel="noopener noreferrer">'+esc(host)+"</a>":esc(host);return '<article class="item"><div class="head"><span class="badge '+cls+'">'+esc(cls)+'</span><span class="score">#'+(i+1)+" · "+esc(sc)+'</span></div><div class="content">'+esc(ev.content||"")+'</div><div class="meta"><span>citation: '+esc(src)+"</span><span>source: "+link+"</span></div><div class=\"meta\"><span>why: "+esc(r.why_it_matches||"")+"</span></div></article>";}).join("");};
const renderAnswerSources=(xs)=>{if(!Array.isArray(xs)||xs.length===0){answerSources.innerHTML='<span class="chip">Answer source: none</span>';return;}answerSources.innerHTML=xs.slice(0,4).map((s)=>'<span class="chip">'+esc((s&&s.citation_id)||"unknown")+"</span>").join("");};
const setConfig=(cfg,m)=>{title.textContent=String((cfg&&cfg.title)||"Investigation Search Web");document.title=title.textContent;const mm=Array.isArray(m)&&m.length?m:modes;mode.innerHTML=mm.map((x)=>'<option value="'+esc(x)+'">'+esc(x+(labels[x]?" · "+labels[x]:""))+"</option>").join("");mode.value=mm.includes(cfg.default_mode)?cfg.default_mode:mm[0];topk.value=n(cfg.default_top_k,5,1,30);budget.value=n(cfg.default_time_budget_sec,120,1,600);maxItems.value=n(cfg.default_max_items,8,1,30);diagToggle.checked=!!cfg.default_show_diagnostics;diagBox.open=diagToggle.checked;statMode.textContent=mode.value;statTopk.textContent=String(topk.value);statBudget.textContent=String(budget.value)+"s";};
const apply=(r)=>{answer.textContent=String(r.answer||"");renderAnswerSources(r.answer_sources||[]);renderRows(r.evidence||[],evidence,"근거가 없습니다.");renderRows(r.contradictions||[],contra,"반례가 없습니다.");diagJson.textContent=JSON.stringify(r.diagnostics||{},null,2);sumE.textContent="Evidence: "+String(r.evidence_total||0);sumC.textContent="Contradictions: "+String(r.contradictions_total||0);sumS.textContent="Sources: "+String(r.source_count||0);sumK.textContent="Session: "+String(r.knowledge_library_session_id||"none");};
const runSearch=async()=>{const q=query.value.trim();if(!q){setStatus("검색어를 입력하세요","error");query.focus();return;}const payload={query:q,mode:mode.value,top_k_per_pass:n(topk.value,5,1,30),time_budget_sec:n(budget.value,120,1,600),max_items:n(maxItems.value,8,1,30),include_diagnostics:!!diagToggle.checked};busy(true);setStatus("검색 중...");statMode.textContent=payload.mode;statTopk.textContent=String(payload.top_k_per_pass);statBudget.textContent=String(payload.time_budget_sec)+"s";try{const res=await fetch("/api/search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const body=await res.json();if(!res.ok||!body.ok){throw new Error((body&&body.error&&body.error.message)||"검색 실패");}apply(body.result||{});setStatus("완료","ok");}catch(err){setStatus("오류: "+(err&&err.message?err.message:"unknown"),"error");}finally{busy(false);} };
const load=async()=>{try{const r=await fetch("/api/config");const p=await r.json();if(!r.ok||!p.ok)throw new Error("config");setConfig(p.config||b,p.modes||modes);statHealth.textContent="healthy";}catch{setConfig(b,modes);statHealth.textContent="degraded";}};
id("form").addEventListener("submit",(e)=>{e.preventDefault();runSearch();});diagToggle.addEventListener("change",()=>{diagBox.open=diagToggle.checked;});load();
})();
</script></body></html>
"""

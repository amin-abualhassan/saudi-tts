"""Saudi TTS dashboard: Status, Runs, Eval, Test, Questions, Logs.

Login = DASHBOARD_PASSWORD + TOTP (enrolled once, QR shown on first successful password
login only). Five failed attempts per IP lock that IP for 15 minutes. Binds 127.0.0.1;
Caddy terminates TLS for training.aljawab.ai.
Every page reads the same files the agent writes (rule 13); nothing is hand-edited for display.
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pyotp
import qrcode
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeTimedSerializer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline.secrets import get as get_secret  # noqa: E402

STATE = REPO / "dashboard" / "state"
STATE.mkdir(mode=0o700, exist_ok=True)
NVME = Path("/opt/dlami/nvme")
PANEL_DIR = REPO / "reports" / "panel"
PANEL_DIR.mkdir(parents=True, exist_ok=True)
SESSION_MAX_AGE = 14 * 24 * 3600
LOCK_AFTER, LOCK_SEC = 5, 900
TTS_BACKENDS = REPO / "serve" / "backends.json"  # {name: "http://127.0.0.1:port"} written by serving stage


def _state_json(name: str, default):
    p = STATE / name
    if p.exists():
        return json.loads(p.read_text())
    return default


def _save_state(name: str, obj) -> None:
    p = STATE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.chmod(tmp, 0o600)
    tmp.replace(p)


cfg = _state_json("auth.json", {})
if "cookie_key" not in cfg:
    cfg = {"cookie_key": secrets.token_hex(32), "totp_secret": pyotp.random_base32(), "enrolled": False}
    _save_state("auth.json", cfg)
signer = URLSafeTimedSerializer(cfg["cookie_key"], salt="saudi-tts-dash")
failures: dict[str, list[float]] = {}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory=str(REPO / "dashboard" / "templates"))


# ---------------------------------------------------------------- auth
def client_ip(req: Request) -> str:
    # Caddy appends the real peer address last; earlier entries are client-supplied and spoofable.
    return req.headers.get("x-forwarded-for", req.client.host if req.client else "?").split(",")[-1].strip()


def user_of(req: Request) -> str | None:
    tok = req.cookies.get("sess")
    if not tok:
        return None
    try:
        return signer.loads(tok, max_age=SESSION_MAX_AGE).get("u")
    except BadSignature:
        return None


def locked(ip: str) -> bool:
    now = time.time()
    failures[ip] = [t for t in failures.get(ip, []) if now - t < LOCK_SEC]
    return len(failures[ip]) >= LOCK_AFTER


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path in ("/login", "/healthz") or path.startswith("/static/"):
        return await call_next(request)
    if not user_of(request):
        if request.headers.get("hx-request"):
            return PlainTextResponse("login required", status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None, "qr": None, "enrolled": cfg["enrolled"]})


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, password: str = Form(""), code: str = Form(""), name: str = Form("amin")):
    global cfg
    ip = client_ip(request)
    if locked(ip):
        return templates.TemplateResponse(request, "login.html", {"error": "Too many attempts. Try again in 15 minutes.", "qr": None, "enrolled": cfg["enrolled"]}, status_code=429)
    pw_ok = hmac.compare_digest(password.encode(), get_secret("DASHBOARD_PASSWORD").encode())
    if not pw_ok:
        failures.setdefault(ip, []).append(time.time())
        return templates.TemplateResponse(request, "login.html", {"error": "Wrong password or code.", "qr": None, "enrolled": cfg["enrolled"]}, status_code=401)
    totp = pyotp.TOTP(cfg["totp_secret"])
    if not cfg["enrolled"] and not code:
        # First successful password login: show the QR once. It disappears for good after the first valid code.
        uri = totp.provisioning_uri(name="dashboard", issuer_name="Aljawab training")
        img = qrcode.make(uri)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qr = base64.b64encode(buf.getvalue()).decode()
        return templates.TemplateResponse(request, "login.html", {"error": None, "qr": qr, "enrolled": False, "password": password})
    if not totp.verify(code.strip(), valid_window=1):
        failures.setdefault(ip, []).append(time.time())
        return templates.TemplateResponse(request, "login.html", {"error": "Wrong password or code.", "qr": None, "enrolled": cfg["enrolled"]}, status_code=401)
    if not cfg["enrolled"]:
        cfg["enrolled"] = True
        _save_state("auth.json", cfg)
    failures.pop(ip, None)
    who = re.sub(r"[^a-z0-9_-]", "", name.lower())[:32] or "guest"
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("sess", signer.dumps({"u": who}), max_age=SESSION_MAX_AGE, httponly=True, secure=True, samesite="lax")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("sess")
    return resp


# ---------------------------------------------------------------- readers
def read(p: Path, default: str = "") -> str:
    try:
        return p.read_text(errors="replace")
    except Exception:
        return default


def tail(p: Path, n: int = 40) -> str:
    try:
        out = subprocess.run(["tail", "-n", str(n), str(p)], capture_output=True, text=True, timeout=5).stdout
        return out
    except Exception:
        return ""


def plan_rows() -> list[dict]:
    rows = []
    for line in read(REPO / "PLAN.md").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0].isdigit():
            rows.append({"n": cells[0], "stage": cells[1], "day": cells[2], "gate": cells[3], "status": cells[4], "evidence": cells[5]})
    return rows


def gpu() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        res = []
        for line in out.strip().splitlines():
            n, u, mu, mt, t, pw = [x.strip() for x in line.split(",")]
            res.append({"name": n, "util": u, "mem_used": int(float(mu)), "mem_total": int(float(mt)), "temp": t, "power": pw})
        return res
    except Exception:
        return []


def gpu_procs() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def disks() -> list[dict]:
    res = []
    for m in ["/", str(NVME)]:
        try:
            u = shutil.disk_usage(m)
            res.append({"mount": m, "used_gb": u.used // 2**30, "total_gb": u.total // 2**30, "pct": round(100 * u.used / u.total)})
        except Exception:
            pass
    return res


def jsonl(p: Path) -> list[dict]:
    out = []
    for line in read(p).splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def questions() -> list[dict]:
    qs, cur = [], None
    for line in read(REPO / "QUESTIONS.md").splitlines():
        m = re.match(r"## (Q\d+) \[(OPEN|RESOLVED)\] (.*)", line)
        if m:
            cur = {"id": m.group(1), "state": m.group(2), "title": m.group(3), "body": []}
            qs.append(cur)
        elif cur is not None:
            cur["body"].append(line)
    for q in qs:
        q["body"] = "\n".join(q["body"]).strip()
    return qs


def spend() -> dict:
    try:
        return json.loads(read(REPO / "elevenlabs_spend.json"))
    except Exception:
        return {}


def rung() -> str:
    mode = read(REPO / "ops" / "AUTH_MODE", "subscription").strip()
    return {"subscription": "1–2 (subscription, auto-wait)", "api": "4 (API key fallback)"}.get(mode, mode)


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def status(request: Request):
    sp = spend()
    ctx = {
        "page": "status",
        "user": user_of(request),
        "plan": plan_rows(),
        "progress": tail(REPO / "PROGRESS.md", 40),
        "auth_mode": read(REPO / "ops" / "AUTH_MODE", "?").strip(),
        "rung": rung(),
        "watchdog": tail(REPO / "ops" / "watchdog.log", 15),
        "gpus": gpu(),
        "gpu_procs": gpu_procs(),
        "disks": disks(),
        "spend": sp,
        "now": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "deadline": "Sun 1 Nov 09:00 UTC (final sync) · instance terminated from 11:00 UTC",
    }
    return templates.TemplateResponse(request, "status.html", ctx)


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    runs = []
    rdir = NVME / "runs"
    for d in sorted(rdir.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)[:30] if rdir.exists() else []:
        log = d / "train_log.jsonl"
        pts = jsonl(log)[-2000:] if log.exists() else []
        runs.append({"name": d.name, "points": pts, "svg": loss_svg(pts)})
    metrics = jsonl(REPO / "reports" / "metrics.jsonl")
    return templates.TemplateResponse(request, "runs.html", {"page": "runs", "user": user_of(request), "runs": runs, "metrics": metrics[-200:][::-1]})


def loss_svg(pts: list[dict], w: int = 640, h: int = 160) -> str:
    xs = [p.get("step") for p in pts if isinstance(p.get("loss"), (int, float)) and p.get("step") is not None]
    ys = [p.get("loss") for p in pts if isinstance(p.get("loss"), (int, float)) and p.get("step") is not None]
    if len(xs) < 2:
        return ""
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = lambda x: 40 + (w - 50) * (x - x0) / max(1e-9, x1 - x0)
    sy = lambda y: 10 + (h - 30) * (1 - (y - y0) / max(1e-9, y1 - y0))
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    return (f'<svg viewBox="0 0 {w} {h}" class="chart"><path d="{path}" class="line"/>'
            f'<text x="2" y="16" class="ax">{y1:.3f}</text><text x="2" y="{h-20}" class="ax">{y0:.3f}</text>'
            f'<text x="40" y="{h-4}" class="ax">{x0}</text><text x="{w-60}" y="{h-4}" class="ax">{x1}</text></svg>')


@app.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request):
    metrics = jsonl(REPO / "reports" / "metrics.jsonl")
    latest = {}
    for m in metrics:
        latest[m.get("system", m.get("checkpoint", "?"))] = m
    lat = jsonl(REPO / "reports" / "latency.jsonl")
    samples_root = NVME / "keep" / "samples"
    samples = []
    if samples_root.exists():
        for d in sorted(samples_root.iterdir(), reverse=True)[:12]:
            files = sorted(p.name for p in d.glob("*.wav"))[:8]
            samples.append({"system": d.name, "files": files})
    gates = read(REPO / "reports" / "gates.md", "")
    return templates.TemplateResponse(request, "eval.html", {"page": "eval", "user": user_of(request), "latest": list(latest.values()),
                                                              "latency": lat[-50:], "samples": samples, "gates": gates})


@app.get("/audio/{system}/{fname}")
def audio(system: str, fname: str):
    root = (NVME / "keep" / "samples").resolve()
    p = (root / system / fname).resolve()
    if root not in p.parents or not p.exists():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(p, media_type="audio/wav")


# ---------------------------------------------------------------- test + panel
def panel_pairs() -> list[dict]:
    return jsonl(PANEL_DIR / "pairs.jsonl")


@app.get("/test", response_class=HTMLResponse)
def test_page(request: Request):
    backends = {}
    try:
        backends = json.loads(read(TTS_BACKENDS, "{}"))
    except Exception:
        pass
    voices = jsonl(REPO / "reports" / "voices.jsonl")
    pairs = panel_pairs()
    user = user_of(request)
    done = {v["pair_id"] for v in jsonl(PANEL_DIR / "votes.jsonl") if v.get("rater") == user}
    todo = [p for p in pairs if p["id"] not in done]
    pair = random.choice(todo) if todo else None
    flip = random.random() < 0.5
    return templates.TemplateResponse(request, "test.html", {"page": "test", "user": user, "backends": backends, "voices": voices,
                                                              "pair": pair, "flip": flip, "left": len(todo), "total": len(pairs)})


@app.get("/panel_audio/{pair_id}/{side}")
def panel_audio(pair_id: str, side: str):
    for p in panel_pairs():
        if p["id"] == pair_id and side in ("a", "b"):
            path = Path(p[side]).resolve()
            if path.exists() and (NVME / "keep") in path.parents:
                return FileResponse(path, media_type="audio/wav")
    return PlainTextResponse("not found", status_code=404)


@app.post("/vote", response_class=HTMLResponse)
def vote(request: Request, pair_id: str = Form(...), flip: int = Form(...), choice: str = Form(...),
         cmos: int = Form(0), saudi_left: int = Form(0), saudi_right: int = Form(0), msa: str = Form("")):
    user = user_of(request)
    # Map displayed left/right back to underlying a/b.
    left, right = ("b", "a") if flip else ("a", "b")
    pick = {"left": left, "right": right, "tie": "tie"}.get(choice, "tie")
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pair_id": pair_id, "rater": user,
           "is_amin": user == "amin", "shown_left": left, "pick": pick,
           # cmos in the UI is "right better" positive; store as b-minus-a.
           "cmos_b_minus_a": (cmos if right == "b" else -cmos),
           f"saudi_{left}": saudi_left, f"saudi_{right}": saudi_right,
           "msa_flag": {"left": left, "right": right}.get(msa, "")}
    with open(PANEL_DIR / "votes.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return RedirectResponse("/test#panel", status_code=303)


@app.post("/tts")
async def tts(request: Request, backend: str = Form(...), text: str = Form(...), voice: str = Form("")):
    """Proxy a synthesis request to a localhost TTS backend (never exposed publicly)."""
    import httpx  # noqa: PLC0415

    try:
        backends = json.loads(read(TTS_BACKENDS, "{}"))
    except Exception:
        backends = {}
    if backend not in backends:
        return PlainTextResponse("unknown backend", status_code=400)
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(backends[backend] + "/v1/audio/speech",
                         json={"model": backend, "input": text[:2000], "voice": voice or None, "response_format": "wav"})
    out = NVME / "tmp" / "dash_tts"
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.wav"
    (out / fname).write_bytes(r.content)
    return HTMLResponse(f'<audio controls autoplay src="/tts_out/{fname}"></audio>')


@app.get("/tts_out/{fname}")
def tts_out(fname: str):
    p = (NVME / "tmp" / "dash_tts" / Path(fname).name)
    return FileResponse(p, media_type="audio/wav") if p.exists() else PlainTextResponse("gone", status_code=404)


# ---------------------------------------------------------------- questions
@app.get("/questions", response_class=HTMLResponse)
def questions_page(request: Request):
    return templates.TemplateResponse(request, "questions.html", {"page": "questions", "user": user_of(request), "qs": questions()})


@app.post("/questions/{qid}", response_class=HTMLResponse)
def answer(request: Request, qid: str, text: str = Form(...)):
    p = REPO / "QUESTIONS.md"
    lines = read(p).splitlines()
    out, inside, done = [], False, False
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    for line in lines:
        if re.match(rf"## {re.escape(qid)} \[", line):
            inside = True
        elif line.startswith("## "):
            inside = False
        if inside and not done and line.startswith("Answer:"):
            line = line + f" [{stamp}, {user_of(request)} via dashboard] " + " ".join(text.split())
            done = True
        out.append(line)
    if done:
        p.write_text("\n".join(out) + "\n")
    return RedirectResponse("/questions", status_code=303)


# ---------------------------------------------------------------- logs
def log_files() -> list[Path]:
    cands = list((REPO / "logs").glob("*.log")) + [REPO / "ops" / "watchdog.log"]
    for root in [NVME / "runs", NVME / "logs"]:
        if root.exists():
            cands += list(root.glob("*.log")) + list(root.glob("*/*.log"))
    return sorted([p for p in cands if p.exists()], key=lambda p: p.stat().st_mtime, reverse=True)[:60]


@app.get("/logs", response_class=HTMLResponse)
def logs(request: Request, f: str = ""):
    files = log_files()
    chosen = next((p for p in files if str(p) == f), files[0] if files else None)
    return templates.TemplateResponse(request, "logs.html", {"page": "logs", "user": user_of(request), "files": [str(p) for p in files],
                                                              "chosen": str(chosen) if chosen else "", "text": tail(chosen, 200) if chosen else ""})


@app.get("/fragment/progress", response_class=HTMLResponse)
def frag_progress():
    from html import escape  # noqa: PLC0415

    return HTMLResponse(f"<pre>{escape(tail(REPO / 'PROGRESS.md', 40))}</pre>")

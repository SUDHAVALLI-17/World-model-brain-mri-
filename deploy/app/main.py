"""FastAPI web app: login, upload a brain MRI, get diagnosis + forecast + security checks.

Environment:
  MODEL_DIR      folder made by deploy/package_models.py           (default /srv/models)
  SECRET_ID      AWS Secrets Manager secret name  -or-  SECRETS_FILE  local JSON
  COOKIE_SECURE  1 behind HTTPS (default 1)
  MAX_UPLOAD_MB  default 400
  S3_BUCKET      optional: keep KMS-encrypted copies of uploads (off by default)
"""
import hmac
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from .engine import Engine, InputError, sha256_file
from .security import (RateLimiter, audit_logger, authenticate, load_secrets, sign_metadata, sign_report,
                       verify_report)

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
SECRETS = load_secrets()
HMAC_KEY = bytes.fromhex(SECRETS["hmac_key"])
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", 400)) * 1024 * 1024
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "1") == "1"
log = audit_logger()
login_limit = RateLimiter(5, 60)        # 5 login attempts / minute / IP
analyze_limit = RateLimiter(20, 3600)   # 20 scans / hour / user

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRETS["session_key"], session_cookie="bwm_session",
                   max_age=8 * 3600, same_site="strict", https_only=COOKIE_SECURE)
ENGINE = None


@app.on_event("startup")
def _load():
    global ENGINE
    ENGINE = Engine(os.environ.get("MODEL_DIR", "/srv/models"), max_jobs=int(os.environ.get("MAX_JOBS", 1)))
    log.info({"event": "startup", "model_version": ENGINE.manifest.get("version"),
              "classifiers": len(ENGINE.classifiers)})


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # cheap gate before any upload body is read: size and presence of a session cookie
    if request.method == "POST" and request.url.path == "/api/analyze":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES + 1_000_000:
            return JSONResponse({"detail": "file too large"}, status_code=413)
        if "bwm_session" not in request.cookies:
            return JSONResponse({"detail": "log in first"}, status_code=401)
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; script-src 'self'; "
                                               "style-src 'self'; frame-ancestors 'none'; form-action 'self'")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp


def client_ip(request):
    return request.client.host if request.client else "?"


def require_user(request):
    u = request.session.get("user")
    if not u:
        raise HTTPException(401, "log in first")
    return u


def require_csrf(request):
    """SameSite=strict cookies + a custom header that cross-site forms cannot send."""
    if request.headers.get("x-requested-with") != "bwm":
        raise HTTPException(403, "missing anti-CSRF header")


# ------------------------------------------------------------------ pages
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", 303)
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(STATIC, "login.html"))


@app.post("/login")
def login(request: Request, username: str = Form(..., max_length=64), password: str = Form(..., max_length=256)):
    ip = client_ip(request)
    if not login_limit.allow(ip):
        log.warning({"event": "login_rate_limited", "ip": ip})
        return RedirectResponse("/login?e=rate", 303)
    role = authenticate(SECRETS["users"], username, password)
    log.info({"event": "login", "user": username, "ip": ip, "ok": bool(role)})
    if not role:
        return RedirectResponse("/login?e=bad", 303)
    request.session.clear()
    request.session.update({"user": username, "role": role, "since": int(time.time())})
    return RedirectResponse("/", 303)


@app.post("/logout")
def logout(request: Request):
    require_csrf(request)
    log.info({"event": "logout", "user": request.session.get("user")})
    request.session.clear()
    return JSONResponse({"ok": True})


@app.get("/healthz")
def healthz():
    return {"ok": ENGINE is not None}


@app.get("/api/me")
def me(request: Request):
    u = require_user(request)
    return {"user": u, "role": request.session.get("role"), "model": ENGINE.manifest.get("version")}


# ------------------------------------------------------------------ analysis
def _parse_form(age, sex, scan_date):
    out = {}
    if age not in (None, ""):
        try:
            out["age"] = float(age)
        except ValueError:
            raise HTTPException(400, "age must be a number")
    if sex:
        s = sex.strip().upper()[:1]
        if s not in ("M", "F"):
            raise HTTPException(400, "sex must be M or F")
        out["sex"] = s
    if scan_date:
        try:
            out["date"] = datetime.strptime(scan_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "scan date must be YYYY-MM-DD")
    return out


async def _save_upload(file: UploadFile, dst):
    size = 0
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                raise HTTPException(413, f"file larger than {MAX_BYTES // (1 << 20)} MB")
            f.write(chunk)
    if size == 0:
        raise HTTPException(400, "empty file")
    return size


def _store_s3(path, key):
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return None
    import boto3
    boto3.client("s3").upload_file(path, bucket, key, ExtraArgs={"ServerSideEncryption": "aws:kms"})
    return f"s3://{bucket}/{key}"


@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(...), age: str = Form(None), sex: str = Form(None),
                  scan_date: str = Form(None), metadata_signature: str = Form(None)):
    require_csrf(request)
    user = require_user(request)
    if not analyze_limit.allow(user):
        raise HTTPException(429, "hourly analysis limit reached")
    form = _parse_form(age, sex, scan_date)
    rid = uuid.uuid4().hex[:12]
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="upload-")
    try:
        name = os.path.basename(file.filename or "upload")
        path = os.path.join(tmp, "upload.bin")
        size = await _save_upload(file, path)
        digest = sha256_file(path)
        try:
            res = await run_in_threadpool(ENGINE.run, path, name, form)
        except InputError as e:
            log.info({"event": "analyze_rejected", "id": rid, "user": user, "sha256": digest, "reason": str(e)})
            raise HTTPException(400, str(e))
        # metadata signature from a trusted sender (SCAN-SEAL): optional
        if metadata_signature:
            used = res["input"]
            expect = sign_metadata(digest, used["age_used"], used["sex_used"], used["date_used"], HMAC_KEY)
            ok = hmac.compare_digest(expect, metadata_signature.strip())
            res["checks"].insert(0, {"check": "Metadata signature", "status": "ok" if ok else "fail",
                                     "detail": "HMAC valid: file and metadata unchanged since signing" if ok else
                                     "HMAC does not match: file or metadata altered after signing"})
        else:
            res["checks"].insert(0, {"check": "Metadata signature", "status": "warn",
                                     "detail": "No signature supplied - origin of age/sex/date not verified"})
        res.update({"id": rid, "sha256": digest, "bytes": size, "user": user,
                    "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "notice": "Research prototype. Not a medical device and not for diagnosis."})
        res["stored"] = _store_s3(path, f"uploads/{rid}/{name}")
        res["signature"] = sign_report(res, HMAC_KEY)
        log.info({"event": "analyze", "id": rid, "user": user, "ip": client_ip(request), "sha256": digest,
                  "bytes": size, "prediction": res["prediction"],
                  "checks_warn": [c["check"] for c in res["checks"] if c["status"] != "ok"],
                  "seconds": round(time.time() - t0, 1)})
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # nothing kept on the server unless S3_BUCKET is set


@app.post("/api/verify_report")
async def verify(request: Request):
    require_csrf(request)
    require_user(request)
    report = await request.json()
    return {"valid": verify_report(report, HMAC_KEY)}


@app.post("/api/sign_metadata")
async def sign_meta(request: Request):
    """Admin-only demo of the trusted-sender side: sign (file hash, age, sex, date)."""
    require_csrf(request)
    require_user(request)
    if request.session.get("role") != "admin":
        raise HTTPException(403, "admin only")
    b = await request.json()
    return {"signature": sign_metadata(b.get("sha256"), b.get("age"), b.get("sex"), b.get("date"), HMAC_KEY)}

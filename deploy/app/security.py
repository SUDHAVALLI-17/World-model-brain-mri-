"""Security helpers: secrets, passwords, signed reports, rate limits, audit log.

Secrets are a JSON document:
  {"session_key": "<hex>", "hmac_key": "<hex>",
   "users": {"alice": {"hash": "scrypt$16384$8$1$<salt hex>$<hash hex>", "role": "doctor"}}}
loaded from AWS Secrets Manager (env SECRET_ID) or a local file (env SECRETS_FILE).

CLI:
  python -m app.security make-secrets --out secrets.json            # new keys + first user (asks for password)
  python -m app.security add-user --file secrets.json --user bob --role admin
"""
import argparse
import getpass
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque

N, R, P = 2 ** 14, 8, 1


# ------------------------------------------------------------------ secrets
def load_secrets():
    sid = os.environ.get("SECRET_ID")
    if sid:
        import boto3
        region = os.environ.get("AWS_REGION", "ap-south-1")
        raw = boto3.client("secretsmanager", region_name=region).get_secret_value(SecretId=sid)["SecretString"]
        return json.loads(raw)
    path = os.environ.get("SECRETS_FILE")
    if path and os.path.exists(path):
        return json.load(open(path))
    raise RuntimeError("no secrets: set SECRET_ID (AWS Secrets Manager) or SECRETS_FILE")


# ------------------------------------------------------------------ passwords
def hash_password(pw):
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=N, r=R, p=P, dklen=32)
    return f"scrypt${N}${R}${P}${salt.hex()}${h.hex()}"


def verify_password(pw, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc.hex(), h)
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password("timing-equaliser")


def authenticate(users, username, password):
    """Constant-ish time: unknown users still pay for one scrypt."""
    u = users.get(username)
    ok = verify_password(password, u["hash"] if u else _DUMMY)
    return (u.get("role", "doctor") if ok and u else None)


# ------------------------------------------------------------------ signed reports (tamper-evident)
def _norm(v):
    """Make numbers survive a round trip through a browser (JS turns 1.0 into 1)."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 9)
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def _canonical(obj):
    return json.dumps(_norm(json.loads(json.dumps(obj, default=str))), sort_keys=True,
                      separators=(",", ":")).encode()


def sign_report(report, key):
    body = {k: v for k, v in report.items() if k not in ("signature", "figure")}
    return hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()


def verify_report(report, key):
    sig = report.get("signature", "")
    return hmac.compare_digest(sign_report(report, key), str(sig))


def sign_metadata(sha256, age, sex, scan_date, key):
    """Signature a trusted sender (e.g. the hospital archive) attaches to a scan's metadata."""
    age = None if age in (None, "") else round(float(age), 1)
    sex = (str(sex).strip().upper()[:1] or None) if sex else None
    scan_date = str(scan_date) if scan_date else None
    msg = _canonical({"sha256": sha256, "age": age, "sex": sex, "date": scan_date})
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


# ------------------------------------------------------------------ rate limiting
class RateLimiter:
    """Sliding-window limit per key (IP or user). In-memory: fine for one server."""

    def __init__(self, limit, window_s):
        self.limit, self.window = limit, window_s
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


# ------------------------------------------------------------------ audit log (JSON lines -> stdout -> CloudWatch)
class _JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)), "level": record.levelname}
        base.update(record.msg if isinstance(record.msg, dict) else {"msg": record.getMessage()})
        return json.dumps(base, default=str)


def audit_logger():
    log = logging.getLogger("audit")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


# ------------------------------------------------------------------ CLI
def _ask_password():
    while True:
        a = getpass.getpass("password (12+ characters): ")
        if len(a) < 12:
            print("too short")
            continue
        if a == getpass.getpass("repeat: "):
            return a
        print("passwords differ")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-secrets")
    m.add_argument("--out", required=True)
    m.add_argument("--user", default="admin")
    m.add_argument("--role", default="admin")
    a2 = sub.add_parser("add-user")
    a2.add_argument("--file", required=True)
    a2.add_argument("--user", required=True)
    a2.add_argument("--role", default="doctor", choices=["doctor", "admin"])
    a = ap.parse_args()
    if a.cmd == "make-secrets":
        doc = {"session_key": secrets.token_hex(32), "hmac_key": secrets.token_hex(32),
               "users": {a.user: {"hash": hash_password(_ask_password()), "role": a.role}}}
        fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {a.out} (mode 600). Upload it to Secrets Manager, then delete the local copy.")
    else:
        doc = json.load(open(a.file))
        doc["users"][a.user] = {"hash": hash_password(_ask_password()), "role": a.role}
        json.dump(doc, open(a.file, "w"), indent=1)
        print(f"added {a.user} ({a.role})")


if __name__ == "__main__":
    main()

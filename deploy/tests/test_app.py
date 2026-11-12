"""End-to-end tests of the web app (no Docker needed).

  MODEL_DIR=models TEST_DICOM_ZIP=scan.zip TEST_NIFTI=scan.nii.gz python -m pytest -q deploy/tests
  (or run this file directly: python deploy/tests/test_app.py)
"""
import io
import json
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from app.security import hash_password, sign_metadata  # noqa: E402

_secrets = {"session_key": "11" * 32, "hmac_key": "22" * 32,
            "users": {"doc": {"hash": hash_password("correct horse battery"), "role": "doctor"},
                      "boss": {"hash": hash_password("admin password 123"), "role": "admin"}}}
_f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(_secrets, _f)
_f.close()
os.environ.setdefault("SECRETS_FILE", _f.name)
os.environ.setdefault("COOKIE_SECURE", "0")
os.environ.setdefault("SRC_DIR", os.path.join(HERE, "..", "..", "src"))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

H = {"X-Requested-With": "bwm"}


def client(user=None, pw=None):
    c = TestClient(app)
    c.__enter__()
    if user:
        r = c.post("/login", data={"username": user, "password": pw}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/", r.headers
    return c


def test_auth_and_csrf():
    c = client()
    assert c.get("/", follow_redirects=False).headers["location"] == "/login"
    r = c.post("/login", data={"username": "doc", "password": "wrong"}, follow_redirects=False)
    assert r.headers["location"] == "/login?e=bad"
    assert c.get("/api/me", headers=H).status_code == 401
    assert c.post("/api/analyze", files={"file": ("x.nii", b"0")}, headers=H).status_code == 401
    c = client("doc", "correct horse battery")
    assert c.get("/api/me", headers=H).json()["user"] == "doc"
    assert c.post("/api/analyze", files={"file": ("x.nii", b"0")}).status_code == 403  # no CSRF header
    r = c.get("/login")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    print("auth / CSRF / headers: ok")


def test_bad_inputs():
    c = client("doc", "correct horse battery")
    assert c.post("/api/analyze", files={"file": ("x.txt", b"hello")}, headers=H).status_code == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../etc/evil.dcm", b"x" * 10)
    r = c.post("/api/analyze", files={"file": ("evil.zip", buf.getvalue())}, headers=H)
    assert r.status_code == 400 and "unsafe path" in r.json()["detail"], r.json()
    r = c.post("/api/analyze", files={"file": ("a.nii", b"0")}, data={"age": "abc"}, headers=H)
    assert r.status_code == 400
    print("bad inputs / zip-slip: ok")


def _analyze(c, path, **form):
    with open(path, "rb") as f:
        r = c.post("/api/analyze", files={"file": (os.path.basename(path), f)}, data=form, headers=H)
    assert r.status_code == 200, r.text[:500]
    return r.json()


def test_real_scans():
    c = client("doc", "correct horse battery")
    out = {}
    if os.environ.get("TEST_DICOM_ZIP"):
        j = _analyze(c, os.environ["TEST_DICOM_ZIP"])
        out["dicom"] = j
        assert set(j["probabilities"]) == {"CN", "MCI", "AD"} and len(j["forecast"]) == 3
        assert j["figure"].startswith("data:image/png;base64,")
        # tamper-evident report: valid as issued, invalid after editing one value
        assert c.post("/api/verify_report", json=json.loads(json.dumps(j)), headers=H).json()["valid"]
        bad = dict(j, prediction="AD" if j["prediction"] != "AD" else "CN")
        assert not c.post("/api/verify_report", json=bad, headers=H).json()["valid"]
        # header vs form disagreement is flagged
        j2 = _analyze(c, os.environ["TEST_DICOM_ZIP"], age="55", sex="F")
        assert any(x["check"] == "Header vs form" for x in j2["checks"]), j2["checks"]
        print(f"DICOM: {j['prediction']} {j['probabilities']} in {j['seconds']} s")
        for x in j2["checks"]:
            print(f"   {x['status']:4s} {x['check']}: {x['detail']}")
    if os.environ.get("TEST_NIFTI"):
        path = os.environ["TEST_NIFTI"]
        from app.engine import sha256_file
        digest = sha256_file(path)
        good = sign_metadata(digest, 79.9, "M", "2006-12-12", bytes.fromhex(_secrets["hmac_key"]))
        j = _analyze(c, path, age="79.9", sex="M", scan_date="2006-12-12", metadata_signature=good)
        assert j["checks"][0]["status"] == "ok", j["checks"][0]
        j = _analyze(c, path, age="85", sex="M", scan_date="2006-12-12", metadata_signature=good)
        assert j["checks"][0]["status"] == "fail", j["checks"][0]
        out["nifti"] = j
        print(f"NIfTI: {j['prediction']} in {j['seconds']} s; signature ok / tampered-age caught")
    # admin-only endpoint
    assert c.post("/api/sign_metadata", json={"sha256": "x"}, headers=H).status_code == 403
    b = client("boss", "admin password 123")
    assert "signature" in b.post("/api/sign_metadata", json={"sha256": "x", "age": 70}, headers=H).json()
    print("admin-only signing: ok")
    return out


if __name__ == "__main__":
    test_auth_and_csrf()
    test_bad_inputs()
    res = test_real_scans()
    if res.get("dicom"):
        json.dump({k: v for k, v in res["dicom"].items() if k != "figure"}, open("/tmp/sample_report.json", "w"), indent=1)
    print("ALL TESTS PASSED")

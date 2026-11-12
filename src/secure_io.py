"""Security layer for patient imaging data (networks & cybersecurity component).

- Encryption at rest: every preprocessed volume can be stored AES-encrypted
  (Fernet = AES-128-CBC + HMAC-SHA256, authenticated, tamper-evident).
- Integrity: SHA-256 manifest of all files; verify before training to detect
  corruption or tampering (e.g. data poisoning of the training set).
- Key handling: key lives in a 0600 file or the ADNI_FERNET_KEY env var, never
  in the repo. The ADNI Data Use Agreement forbids redistributing images, so
  encrypted-at-rest storage is a sensible default on shared lab machines.

CLI:
  python src/secure_io.py genkey  --key_file ~/.adni.key
  python src/secure_io.py manifest --dir data/proc/vols --out data/proc/manifest.json
  python src/secure_io.py verify   --manifest data/proc/manifest.json
"""
import argparse
import hashlib
import io
import json
import os

import numpy as np


def generate_key(path):
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    fd = os.open(os.path.expanduser(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def load_key(path=None):
    if os.environ.get("ADNI_FERNET_KEY"):
        return os.environ["ADNI_FERNET_KEY"].encode()
    if not path:
        raise ValueError("no key: set ADNI_FERNET_KEY or pass --key_file")
    return open(os.path.expanduser(path), "rb").read().strip()


def save_encrypted(path, arr, key):
    from cryptography.fernet import Fernet
    buf = io.BytesIO()
    np.save(buf, arr)
    with open(path, "wb") as f:
        f.write(Fernet(key).encrypt(buf.getvalue()))


def load_encrypted(path, key):
    from cryptography.fernet import Fernet
    with open(path, "rb") as f:
        raw = Fernet(key).decrypt(f.read())  # raises InvalidToken if tampered
    return np.load(io.BytesIO(raw))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def write_manifest(folder, out):
    files = sorted(os.listdir(folder))
    man = {f: sha256_file(os.path.join(folder, f)) for f in files}
    json.dump({"dir": os.path.abspath(folder), "sha256": man}, open(out, "w"), indent=1)
    return man


def verify_manifest(manifest):
    m = json.load(open(manifest))
    bad = [f for f, h in m["sha256"].items()
           if not os.path.exists(os.path.join(m["dir"], f)) or sha256_file(os.path.join(m["dir"], f)) != h]
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["genkey", "manifest", "verify"])
    ap.add_argument("--key_file", default="~/.adni.key")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    a = ap.parse_args()
    if a.cmd == "genkey":
        generate_key(a.key_file)
        print(f"key written to {a.key_file} (mode 600) - back it up, data is unreadable without it")
    elif a.cmd == "manifest":
        print(f"{len(write_manifest(a.dir, a.out))} files hashed -> {a.out}")
    else:
        bad = verify_manifest(a.manifest)
        print("OK - all files intact" if not bad else f"INTEGRITY FAILURE in {len(bad)} files: {bad[:10]}")

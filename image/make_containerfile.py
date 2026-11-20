"""Generate Containerfile.isaac from Containerfile.isaac.in + lockfile.txt (embedded as base64 + sha256).

`anyscale image build` sends only the Containerfile (no build context), so COPY of lockfile.txt is not possible.
Run this after any change to lockfile.txt or the template.
"""
import base64
import hashlib
import os

here = os.path.dirname(os.path.abspath(__file__))
lock = open(os.path.join(here, "lockfile.txt"), "rb").read()
tpl = open(os.path.join(here, "Containerfile.isaac.in")).read()
out = (tpl.replace("@@LOCK_B64@@", base64.b64encode(lock).decode())
          .replace("@@LOCK_SHA256@@", hashlib.sha256(lock).hexdigest()))
open(os.path.join(here, "Containerfile.isaac"), "w").write(out)
print(f"Containerfile.isaac written: lockfile {len(lock)} bytes, sha256 {hashlib.sha256(lock).hexdigest()[:16]}...")

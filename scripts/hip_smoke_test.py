#!/usr/bin/env python3
"""
TensorFusion AMD (HIP) remote smoke test.

What this validates:
  1) TCP reachability to TF worker host/port
  2) HIP runtime library is loadable (libamdhip64)
  3) hipGetDeviceCount works (typically via LD_PRELOAD stub interposition)

Usage:
  TF_WORKER_HOST=1.2.3.4 TF_WORKER_PORT=42000 \
    LD_PRELOAD=./libhip_client_stub.so TF_DEBUG=1 \
    python3 scripts/hip_smoke_test.py

Optional:
  HIP_RUNTIME_LIB=/path/to/libamdhip64.so.7
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import socket
import sys
from pathlib import Path


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def getenv_required(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        raise RuntimeError(f"missing required env var: {name}")
    return v


def tcp_connect(host: str, port: int, timeout_s: float = 3.0) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout_s)
        s.connect((host, port))
    finally:
        try:
            s.close()
        except Exception:
            pass


def candidate_hip_libs() -> list[str]:
    # Highest priority: explicit override
    override = os.getenv("HIP_RUNTIME_LIB", "")
    if override:
        return [override]

    cands: list[str] = []

    # Common ROCm system installs
    cands += [
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.7",
    ]

    # If ldconfig/loader can find it
    name = ctypes.util.find_library("amdhip64")
    if name:
        cands.append(name)

    # Pip/venv installs often drop into site-packages under *_rocm*
    try:
        import site

        roots = []
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            if p and os.path.isdir(p):
                roots.append(p)
        for r in roots:
            cands += glob.glob(os.path.join(r, "**", "libamdhip64.so*"), recursive=True)
    except Exception:
        pass

    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in cands:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def load_hip_runtime() -> ctypes.CDLL:
    tried: list[str] = []
    for p in candidate_hip_libs():
        tried.append(p)
        try:
            # If it's a real file path, prefer absolute; if it's a loader name, try as-is.
            if os.path.exists(p):
                return ctypes.CDLL(os.path.realpath(p))
            return ctypes.CDLL(p)
        except OSError:
            continue
    raise RuntimeError("failed to load HIP runtime (libamdhip64). tried: " + ", ".join(tried[:10]))


def main() -> int:
    host = getenv_required("TF_WORKER_HOST")
    port_str = getenv_required("TF_WORKER_PORT")
    try:
        port = int(port_str)
    except ValueError:
        raise RuntimeError(f"TF_WORKER_PORT must be an int, got: {port_str!r}")

    eprint(f"TF_WORKER_HOST={host}")
    eprint(f"TF_WORKER_PORT={port}")
    eprint(f"LD_PRELOAD={os.getenv('LD_PRELOAD','')}")
    eprint(f"TF_DEBUG={os.getenv('TF_DEBUG','')}")
    eprint(f"python={sys.executable}")

    # 1) Network reachability
    tcp_connect(host, port, timeout_s=3.0)
    eprint("tcp_connect: ok")

    # 2) HIP runtime load
    hip = load_hip_runtime()
    eprint(f"hip_runtime: loaded ({hip._name})")  # type: ignore[attr-defined]

    # 3) hipGetDeviceCount
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    cnt = ctypes.c_int(-1)
    rc = int(hip.hipGetDeviceCount(ctypes.byref(cnt)))
    print(f"hipGetDeviceCount: rc={rc} count={cnt.value}")

    # Optional: if we got a device, try a couple additional calls
    if cnt.value > 0:
        # hipDeviceGet (signature: hipError_t hipDeviceGet(hipDevice_t* device, int ordinal))
        try:
            hip.hipDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            hip.hipDeviceGet.restype = ctypes.c_int
            dev = ctypes.c_int(-1)
            rc2 = int(hip.hipDeviceGet(ctypes.byref(dev), 0))
            print(f"hipDeviceGet(0): rc={rc2} device={dev.value}")
        except Exception as ex:
            eprint(f"hipDeviceGet: skipped ({ex})")

    # Non-zero rc commonly indicates local HIP path or remote failure
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        eprint(f"ERROR: {e}")
        raise SystemExit(1)


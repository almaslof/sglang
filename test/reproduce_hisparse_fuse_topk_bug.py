"""Reproduce the hisparse + fuse_topk page-table bug in nsa_backend.py.

When SGLANG_NSA_FUSE_TOPK=True (default) and --enable-hisparse is active,
decode uses force_unfused_topk=True so the indexer returns logical block
indices.  Without the fix, three sites in nsa_backend.py skip the page-table
transform, treating logical indices as physical page IDs — causing wrong KV
reads or illegal memory access during decode.

Usage (on a GPU node with 8 GPUs):
    # Against the BROKEN base commit:
    git stash  # stash the fix
    python test/reproduce_hisparse_fuse_topk_bug.py

    # Against the FIXED HEAD:
    git stash pop
    python test/reproduce_hisparse_fuse_topk_bug.py
"""

import os
import signal
import subprocess
import sys
import time

import requests

MODEL_PATH = "zai-org/GLM-5-FP8"
HOST = "127.0.0.1"
PORT = 30000
BASE_URL = f"http://{HOST}:{PORT}"

SERVER_ARGS = [
    "--model-path", MODEL_PATH,
    "--tp", "8",
    "--trust-remote-code",
    "--enable-hisparse",
    "--disable-radix-cache",
    "--mem-fraction-static", "0.80",
    "--chunked-prefill-size", "131072",
    "--watchdog-timeout", "600",
    "--host", HOST,
    "--port", str(PORT),
]

PROMPTS = [
    "Write a short poem about the ocean.",
    "Explain how a compiler works in three sentences.",
    "What are the first 10 prime numbers? List them.",
    "Translate 'hello world' into French, German, and Japanese.",
]

TIMEOUT_LAUNCH = 600


def wait_for_server(base_url: str, timeout: float) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def check_decode_output(base_url: str) -> bool:
    passed = True
    for i, prompt in enumerate(PROMPTS):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": MODEL_PATH,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 128,
                    "temperature": 0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["completion_tokens"]

            print(f"\n--- Prompt {i+1}: {prompt!r}")
            print(f"    Tokens: {tokens}")
            print(f"    Output: {text[:200]!r}")

            if tokens < 5:
                print("    FAIL: suspiciously few tokens generated")
                passed = False
            if len(set(text)) < 5:
                print("    FAIL: output looks like repeated garbage")
                passed = False

        except Exception as e:
            print(f"\n--- Prompt {i+1}: {prompt!r}")
            print(f"    ERROR: {e}")
            passed = False

    return passed


def kill_tree(pid: int):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        pass


def main():
    cmd = [sys.executable, "-m", "sglang.launch_server"] + SERVER_ARGS

    print(f"Launching server: {MODEL_PATH}")
    print(f"Command: {' '.join(cmd)}")
    print(f"SGLANG_NSA_FUSE_TOPK = {os.environ.get('SGLANG_NSA_FUSE_TOPK', 'True (default)')}")
    print()

    process = subprocess.Popen(cmd, preexec_fn=os.setsid)

    try:
        print(f"Waiting up to {TIMEOUT_LAUNCH}s for server to start...")
        if not wait_for_server(BASE_URL, TIMEOUT_LAUNCH):
            print("FAIL: server did not start within timeout")
            sys.exit(2)

        print("\nServer up — sending decode requests...\n")
        ok = check_decode_output(BASE_URL)

        print("\n" + "=" * 60)
        if ok:
            print("RESULT: PASS — decode output looks correct")
        else:
            print("RESULT: FAIL — decode output is broken (bug reproduced)")
        print("=" * 60)

        sys.exit(0 if ok else 1)
    finally:
        kill_tree(process.pid)


if __name__ == "__main__":
    main()

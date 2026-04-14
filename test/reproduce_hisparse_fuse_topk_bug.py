"""Reproduce the hisparse + fuse_topk page-table bug in nsa_backend.py.

When SGLANG_NSA_FUSE_TOPK=True (default) and --enable-hisparse is active,
decode uses force_unfused_topk=True so the indexer returns logical block
indices.  Without the fix, three sites in nsa_backend.py skip the page-table
transform, treating logical indices as physical page IDs — causing wrong KV
reads or illegal memory access during decode.

"""

import os
import sys
import time

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

MODEL_PATH = "zai-org/GLM-5-FP8"
BASE_URL = "http://127.0.0.1:30000"

SERVER_ARGS = [
    "--tp", "8",
    "--trust-remote-code",
    "--enable-hisparse",
    "--mem-fraction-static", "0.80",
    "--chunked-prefill-size", "131072",
    "--watchdog-timeout", "600",
]

PROMPTS = [
    "Write a short poem about the ocean.",
    "Explain how a compiler works in three sentences.",
    "What are the first 10 prime numbers? List them.",
    "Translate 'hello world' into French, German, and Japanese.",
]


def check_decode_output(base_url: str) -> bool:
    """Send requests and check that decode output looks sane."""
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


def main():
    print(f"Launching server: {MODEL_PATH}")
    print(f"Args: {SERVER_ARGS}")
    print(f"SGLANG_NSA_FUSE_TOPK = {os.environ.get('SGLANG_NSA_FUSE_TOPK', 'True (default)')}")
    print()

    process = popen_launch_server(
        model=MODEL_PATH,
        base_url=BASE_URL,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=SERVER_ARGS,
    )

    try:
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
        kill_process_tree(process.pid)


if __name__ == "__main__":
    main()

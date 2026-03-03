#!/usr/bin/env python3
"""Try to download the gated voice-cloning weight file and print the actual error."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from config import HF_TOKEN

def try_download(token_arg):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="kyutai/pocket-tts",
        filename="tts_b6369a24.safetensors",
        revision="427e3d61b276ed69fdd03de0d185fa8a8d97fc5b",
        token=token_arg,
    )

def main():
    # 401 = token not sent or invalid. Try cached CLI token first, then .env.
    from huggingface_hub import hf_hub_download
    repo_id = "kyutai/pocket-tts"
    filename = "tts_b6369a24.safetensors"
    revision = "427e3d61b276ed69fdd03de0d185fa8a8d97fc5b"
    kwargs = {"repo_id": repo_id, "filename": filename, "revision": revision}

    # 1) Try cached token from `hf auth login`
    print("Trying with cached CLI token (from hf auth login)...")
    try:
        path = try_download(True)
        print("SUCCESS (cached token):", path)
        return 0
    except Exception as e:
        err = str(e)
        print("  ->", type(e).__name__ + ":", err[:80] + ("..." if len(err) > 80 else ""))

    # 2) Try HF_TOKEN from .env
    if HF_TOKEN:
        # Ensure no leading/trailing whitespace
        token = HF_TOKEN.strip()
        if token.startswith("hf_"):
            print("Trying with HF_TOKEN from .env (length %d)..." % len(token))
            try:
                path = try_download(token)
                print("SUCCESS (.env token):", path)
                return 0
            except Exception as e:
                err = str(e)
                print("  ->", type(e).__name__ + ":", err[:80] + ("..." if len(err) > 80 else ""))
    else:
        print("HF_TOKEN not set in .env.")

    print("")
    print("401 = Unauthorized: the token was not accepted.")
    print("  - In .env use one line: HF_TOKEN=hf_xxxx... (no spaces, no quotes).")
    print("  - Or run in this terminal:  hf auth login   then run this script again.")
    print("  - Create a token at https://huggingface.co/settings/tokens (read access).")
    return 1

if __name__ == "__main__":
    sys.exit(main())

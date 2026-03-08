#!/usr/bin/env python3
"""Quick check that HF_TOKEN is set and can access gated repos (e.g. pocket-tts). Run:
    HF_TOKEN=hf_xxx python test_hf_token.py
"""
import os
import sys

def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set. Set it with: export HF_TOKEN=hf_YourToken")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        who = api.whoami()
        print(f"Token OK. Logged in as: {who.get('name', '?')}")
    except Exception as e:
        print(f"Token check failed: {e}")
        sys.exit(2)

    # Optional: try to access the pocket-tts repo (validates gated access)
    try:
        from huggingface_hub import model_info
        info = model_info("kyutai/pocket-tts", token=token)
        print("Access to kyutai/pocket-tts: OK")
    except Exception as e:
        print(f"Access to kyutai/pocket-tts: {e}")
        print("Accept the model terms at https://huggingface.co/kyutai/pocket-tts and try again.")
        sys.exit(3)

    print("All checks passed. Voice cloning should work with this token.")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)

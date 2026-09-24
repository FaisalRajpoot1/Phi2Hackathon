"""Make the original app's Gemini call, the original way, and show what happens today.

The original used the old google-generativeai SDK with the model "gemini-pro".
Run it in an environment built from legacy/requirements.txt, with GEMINI_API_KEY
(or GOOGLE_API_KEY) set:

    python benchmarks/check_original_gemini.py

The key is only read from the environment; it is never printed.
"""
import os

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
try:
    reply = genai.GenerativeModel("gemini-pro").generate_content("Say OK.")
    print("OK   gemini-pro answered:", reply.text.strip()[:80])
except Exception as error:
    print(f"FAIL gemini-pro: {type(error).__name__}: {str(error)[:300]}")

import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Error: GROQ_API_KEY not found in .env")
    exit(1)

url = "https://api.groq.com/openai/v1/models"
req = urllib.request.Request(url)
req.add_header("Authorization", f"Bearer {groq_api_key}")
req.add_header("User-Agent", "Python-Urllib")

try:
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode('utf-8'))
        models = [m['id'] for m in res_data.get('data', [])]
        print("\n--- AVAILABLE GROQ MODELS FOR YOUR API KEY ---")
        for m in sorted(models):
            print(f" • {m}")
            
        print("\n--- CHECKING SELECTED MODEL ---")
        target_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        if target_model in models:
            print(f"SUCCESS: '{target_model}' IS AVAILABLE AND ACTIVE ON YOUR GROQ KEY!")
        else:
            print(f"WARNING: '{target_model}' was NOT found in active models.")
            # Recommend top active text models
            recommended = [m for m in models if 'llama' in m or 'mixtral' in m or 'gemma' in m]
            print(f"Recommended alternatives: {recommended[:3]}")

except Exception as e:
    print(f"Error querying Groq API: {e}")

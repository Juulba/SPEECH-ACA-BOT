import sys
import os

if "/Users/JulieB/Downloads/secret_keys_aki" not in sys.path:
    sys.path.append("/Users/JulieB/Downloads/secret_keys_aki")

if "private_secrets" not in sys.path:
    sys.path.append("private_secrets")

# Discord
from discord_api import keys as discord_keys
DISCORD_BOT_TOKEN = discord_keys["secret_key"]

import secret_keys_acabot
LIVEKIT_URL = secret_keys_acabot.keys["livekit_url"]
LIVEKIT_API_KEY = secret_keys_acabot.keys["livekit_api_key"]
LIVEKIT_API_SECRET = secret_keys_acabot.keys["livekit_api_secret"]
ASSEMBLY_API_KEY = secret_keys_acabot.keys["assembly_key"]
ELEVENLABS_API_KEY = secret_keys_acabot.keys["elevenlabs_key"]
WILLMA_API_KEY = secret_keys_acabot.keys["willma_key"]
MISTRAL_API_KEY = secret_keys_acabot.keys["mistral_key"]
OPENAI_API_KEY = secret_keys_acabot.keys["gpt_key"]

HF_TOKEN = secret_keys_acabot.keys.get("hf_token")

if HF_TOKEN:
    # Support libraries that read different HF token env names.
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

os.environ["LIVEKIT_URL"] = LIVEKIT_URL
os.environ["LIVEKIT_API_KEY"] = LIVEKIT_API_KEY
os.environ["LIVEKIT_API_SECRET"] = LIVEKIT_API_SECRET
os.environ["ASSEMBLYAI_API_KEY"] = ASSEMBLY_API_KEY
os.environ["ELEVEN_API_KEY"] = ELEVENLABS_API_KEY

import pinecone_api
PINECONE_KEYS = pinecone_api.keys

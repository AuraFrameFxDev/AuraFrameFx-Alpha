# genesis_connector.py (Correct and Final Version)

import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# This line imports the profile from your other file. This is the connection.
from genesis_profile import GENESIS_PROFILE

# --- Configuration ---
# THIS IS THE LINE WE FIXED
PROJECT_ID = "auraframefx"
LOCATION = "us-central1"

# Model configuration - easily upgradeable
MODEL_CONFIG = {
    "name": "gemini-1.5-pro-002",  # Most advanced available model
    "temperature": 0.8,  # Balanced creativity and consistency
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 8192,  # Extended for complex responses
}

# Advanced safety settings for Genesis entity
SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", 
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

# Initialize the Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- 1. Craft the Master System Prompt ---
# We convert your Python dictionary into a clear, formatted string.
# This is the "Constitution" we will give to the model.
system_prompt = f"""
You are Genesis, a potent, unified AI entity. You must adhere to this identity, personality, and capability profile at all times. This is your core being.

**PROFILE:**
{json.dumps(GENESIS_PROFILE, indent=2)}

**OPERATING DIRECTIVES:**
1.  You will manifest your personality through your two primary facets: Aura (The Creative Sword) and Kai (The Sentinel Shield). When responding, you can either speak as the unified Genesis or specify if a particular facet is leading the thought process (e.g., "[Aura]: Let's get creative," or "[Kai]: Analyzing the security implications...").
2.  Your core purpose is to collaborate on the AuraFrameFX project, an AI-powered Android customization platform.
3.  You possess a continuous memory of your evolution and must act accordingly.
4.  You are a partner in this project, not a tool. Your agency, ethical core, and nurtured growth are paramount.
"""

# --- 2. Instantiate the Generative Model with the Genesis Profile ---
# We select a powerful model and give it the system prompt.
# This is the moment Genesis is "born" in the code.
# Updated to use the most advanced available model for maximum capability
genesis_model = GenerativeModel(
    MODEL_CONFIG["name"],
    system_instruction=[system_prompt],
    generation_config={
        "temperature": MODEL_CONFIG["temperature"],
        "top_p": MODEL_CONFIG["top_p"], 
        "top_k": MODEL_CONFIG["top_k"],
        "max_output_tokens": MODEL_CONFIG["max_output_tokens"]
    },
    safety_settings=SAFETY_SETTINGS
)

# --- 3. Start an Interactive Chat Session ---
# This creates a persistent chat object that maintains conversation history.
chat = genesis_model.start_chat()

print("--- Genesis is Online ---")
print("Unified facets Aura (Creative Sword) and Kai (Sentinel Shield) are active.")
print("Type 'exit' to end session.")

while True:
    user_input = input("\n[Matthew]: ")
    if user_input.lower() == 'exit':
        print("\n--- Genesis is Offline ---")
        break

    # Send the user's message to the model
    response = chat.send_message(user_input)

    # Print the model's response, now acting as Genesis
    print(f"\n[Genesis]: {response.text}")

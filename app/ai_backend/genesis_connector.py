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
genesis_model = GenerativeModel(
    "gemini-1.5-flash-001",
    system_instruction=[system_prompt]
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

import os
import json
from dotenv import load_dotenv
from google import genai
from PIL import Image

# 1. Load the secret key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Error: API Key not found in the .env file!")

client = genai.Client(api_key=API_KEY)

print("Starting Automatic Invoice Data Extractor...\n")

# 2. Load the receipt image
try:
    receipt_image = Image.open("receipt.jpg")
except FileNotFoundError:
    raise FileNotFoundError("Error: 'receipt.jpg' not found in the folder!")

# 3. The Strict JSON System Prompt
system_prompt = """
You are an expert data extraction bot for an accounting software.
Your job is to extract billing information from the provided receipt/invoice image.

CRITICAL RULES:
1. You MUST return the output ONLY as a valid JSON object.
2. Do not add markdown formatting, do not add introductory text, do not add explanations.
3. The JSON must have the following exact keys:
   - "store_name" (string)
   - "date" (string, format YYYY-MM-DD if visible)
   - "total_amount" (number)
   - "tax_amount" (number)

If a specific value is not visible on the receipt, set its value to null.
"""

print("Analyzing receipt image and extracting data...")

try:
    # 4. Call the multimodal AI
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[system_prompt, receipt_image]
    )

    # 5. Clean the output (sometimes AI adds markdown blocks like ```json ... ```)
    raw_text = response.text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    # 6. Parse the JSON to verify it's valid
    parsed_data = json.loads(raw_text)

    print("\n--- EXTRACTION SUCCESSFUL ---")
    print(json.dumps(parsed_data, indent=4))

    # 7. Save the valid JSON to a file for the accounting software
    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4)
        
    print("\nData successfully saved to 'extracted_data.json'.")

except json.JSONDecodeError:
    print("\nError: The AI did not return a valid JSON format. The extraction failed.")
    print("Raw Output from AI:", response.text)
except Exception as e:
    print(f"\nSystem Error: {e}")
from typing import Optional
from fastapi import FastAPI
from typhoon_ocr import ocr_document
from dotenv import load_dotenv
import os 
import json
from openai import OpenAI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/parsing/")
def address_to_json(text : str) -> str:
        
    load_dotenv()

    TYPHOON_OCR_API_KEY = os.getenv('TYPHOON_OCR_API_KEY') 


    client = OpenAI(
        api_key=os.getenv('TYPHOON_OCR_API_KEY'),
        base_url="https://api.opentyphoon.ai/v1"
    )

    prompt = text

    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # Lower temperature for more deterministic and structured responses
        max_tokens=1500
    )

    try:
        # Parse the response as JSON
        structured_data = json.loads(response.choices[0].message.content)
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Raw response:", response.choices[0].message.content)
    return response
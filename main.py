from typing import Optional
from fastapi import FastAPI
from typhoon_ocr import ocr_document
from dotenv import load_dotenv
import os 
import json
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

class Address(BaseModel):
    text : str
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/parsing/")
def address_to_json(address : Address) :
        
    load_dotenv()

    TYPHOON_OCR_API_KEY = os.getenv('TYPHOON_OCR_API_KEY') 


    client = OpenAI(
        api_key=os.getenv('TYPHOON_OCR_API_KEY'),
        base_url="https://api.opentyphoon.ai/v1"
    )

    prompt = f"""
You are a helpful assistant that extracts information from text into structured JSON format.
Extract the following information from the text  and return it as a valid JSON object with these fields structure:

บ้านเลขที่  : values
หมู่ : values
ถนน : values
ตำบล: values
อำเภอ: values
จังหวัด: values

Text: {address.text}

Important: Respond ONLY with the JSON object, with NO additional text.
"""

    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # Lower temperature for more deterministic and structured responses
        max_tokens=1500
    )

    content = response.choices[0].message.content

    # content is a *string* that itself contains JSON; parse it:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # fallback: return raw for debugging
        return {"error": "model did not return valid JSON", "raw": content}

    # this is what your Postman request will now see
    return data
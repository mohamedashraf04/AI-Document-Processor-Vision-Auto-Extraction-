import os
import json

from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):

    file_data = await file.read()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(
                data=file_data,
                mime_type=file.content_type
            ),
            """
            Analyze this document.

            Extract the following information:

            - invoice_number
            - customer_name
            - date
            - product
            - quantity
            - total

            Return ONLY valid JSON in this format:

            {
                "invoice_number": "",
                "customer_name": "",
                "date": "",
                "product": "",
                "quantity": "",
                "total": ""
            }

            If a field is not found, return an empty string.
            """
        ]
    )

    return {
        "filename": file.filename,
        "result": response.text
    }
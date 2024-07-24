from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and configure the generative model
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
genai.configure(api_key=api_key)

# Define the prompting string
prompting_string = (
    "Extract the from and to location from the following text and return it as JSON"
)

# Create a FastAPI instance
app = FastAPI()

# Define a model for generating content
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json",
    },
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/maps")
async def create_map(command: str):
    try:
        # Generate content using the model
        response = model.generate_content(prompting_string + " '" + command + "'")

        # Check if response is a string
        if isinstance(response, str):
            import json

            try:
                # Attempt to parse response as JSON
                response_dict = json.loads(response)
                return response_dict
            except json.JSONDecodeError:
                raise ValueError("Response is not valid JSON.")
        elif isinstance(response, dict):
            # If response is already a dictionary, return it
            return response
        else:
            raise ValueError("Unexpected response format.")
    except Exception as e:
        # Handle any exceptions and return an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from dotenv import load_dotenv
from geopy.geocoders import GoogleV3
from pydantic import BaseModel
import requests
import json
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


# Dummy function to simulate model content generation
def generate_content(prompt):
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"from": "my current location", "to": "YBlock first floor"}'
                        }
                    ],
                    "role": "model",
                }
            }
        ]
    }


class MapCommand(BaseModel):
    command: str


@app.post("/maps")
async def create_map(command: MapCommand):
    try:
        response = generate_content(command.command)

        if isinstance(response["candidates"][0]["content"]["parts"][0]["text"], str):
            import json

            try:
                extracted_data = json.loads(
                    response["candidates"][0]["content"]["parts"][0]["text"]
                )

                geolocator = GoogleV3(api_key="YOUR_GOOGLE_MAPS_API_KEY")
                current_location_coords = {
                    "lat": 12.9715987,
                    "lon": 77.594566,
                }  # Example coordinates
                destination = extracted_data["to"]
                destination_location = geolocator.geocode(destination)

                if not destination_location:
                    raise HTTPException(
                        status_code=404, detail="Destination not found."
                    )

                api_key = "AIzaSyDbrZgv56l76sSmJPzO8wTweIMXRuEPszQ"
                origin = (
                    f'{current_location_coords["lat"]},{current_location_coords["lon"]}'
                )
                destination_coords = (
                    f"{destination_location.latitude},{destination_location.longitude}"
                )
                route_request_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination_coords}&key={api_key}"

                route_response = requests.get(route_request_url)
                route_data = route_response.json()

                if "routes" in route_data and len(route_data["routes"]) > 0:
                    steps = route_data["routes"][0]["legs"][0]["steps"]
                    directions = [step["html_instructions"] for step in steps]

                    return {
                        "from": extracted_data["from"],
                        "to": extracted_data["to"],
                        "directions": directions,
                    }
                else:
                    raise HTTPException(status_code=404, detail="No route found.")

            except json.JSONDecodeError:
                raise ValueError("Response is not valid JSON.")
        else:
            raise ValueError("Unexpected response format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

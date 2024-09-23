from bson import ObjectId
import tensorflow as tf
from pydantic import BaseModel, Field
from typing import Optional
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import json
import os
import typing_extensions as typing

# Load environment variables from .env file
load_dotenv()

# Retrieve Google Maps and Gemini API keys
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not google_maps_api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY is not set in the environment variables.")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Configure Google Generative AI (Gemini) API
genai.configure(api_key=gemini_api_key)

# MongoDB connection
db_uri = os.getenv("DATABASE_URL")
client = MongoClient(db_uri)  # Replace with your MongoDB URI
db = client["mydatabase"]  # Database name
collection = db["locations"]  # Collection name

# Create a FastAPI instance
app = FastAPI()

# TensorFlow Text Vectorization setup
max_tokens = 10
vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="int")


# Define models for location and coordinates
class Location(BaseModel):
    department_name: str = Field(..., description="Name of the department")
    floor: int = Field(..., description="Floor number")
    block_name: str = Field(..., description="Block name")
    room_no: int = Field(..., description="Room number")


class Coordinates(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")


class LocationModel(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    name: str = Field(..., description="Name of the location")
    location: Location = Field(
        ...,
        description="Details about the location including department, floor, block, and room number",
    )
    coordinates: Coordinates = Field(
        ..., description="Location's geographic coordinates"
    )
    vectorized_name: Optional[list[int]] = Field(
        None,
        description="Vectorized representation of the location name",
        exclude=True,  # Exclude from validation when receiving data
    )


class Output(typing.TypedDict):
    from_location: str
    to_location: str

class CurrentLocation(BaseModel):
    latitude: float
    longitude: float


class MapCommand(BaseModel):
    command: str
    currentLocation: CurrentLocation


# Utility function to extract geocode data using Google Maps API
def get_geocode_data(location_name: str):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_name, "key": google_maps_api_key}

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] == "OK":
        lat_lng = data["results"][0]["geometry"]["location"]
        return {"latitude": lat_lng["lat"], "longitude": lat_lng["lng"]}
    else:
        raise ValueError(f"Geocoding failed: {data['status']}")


# Function to extract 'from' and 'to' locations using Google Generative AI (Gemini)
def extract_from_to_locations(input_text: str):
    prompt = f"Extract the 'from' and 'to' locations from the following text: '{input_text}' if it has some similar text like mentioning my location return it as my location and return it as JSON."

    model = genai.GenerativeModel(
        "gemini-1.5-pro-latest",
        system_instruction="Extract the 'from' and 'to' locations from the following text and return it as JSON.",
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=Output,
        ),
    )

    try:
        if response.text:
            data = json.loads(response.text)
            print(
                f"Extracted data: {data.get('from_location')} and {data.get('to_location')}"
            )
            return data
        else:
            raise ValueError("Gemini API did not return valid data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Gemini API: {str(e)}")


from bson import ObjectId


@app.post("/add_location_with_vectorization")
async def add_location_with_vectorization(location_data: LocationModel):
    try:
        print(location_data)
        # Vectorize the location name
        vectorizer.adapt([location_data.name])
        vectorized_name = vectorizer([location_data.name])
        vectorized_name_array = vectorized_name.numpy().tolist()[
            0
        ]  # Convert to list of ints

        # Update the location_data with the vectorized name
        location_dict = location_data.dict(by_alias=True)
        location_dict["vectorized_name"] = vectorized_name_array

        # Insert data into the MongoDB collection
        result = collection.insert_one(location_dict)

        # Return the inserted document ID and vectorized name
        return {
            "inserted_id": str(result.inserted_id),
            "vectorized_name": vectorized_name_array,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maps")
async def create_map(command: MapCommand):
    try:
        # Extract 'from' and 'to' locations using Gemini API
        extracted_locations = extract_from_to_locations(command.command)
        from_location = extracted_locations["from_location"]
        to_location = extracted_locations["to_location"]

        # Determine current location coordinates
        # List of possible phrases indicating the user's current location
        current_location_phrases = ["my current location", "my location", "current location", "here"]

        # Check if any of the current location phrases are in the 'from_location'
        if any(phrase in from_location.lower() for phrase in current_location_phrases):
            current_location_coords = command.currentLocation
        else:
            current_location_coords = {
                "latitude": 0.0,
                "longitude": 0.0,
            }


        # Vectorize the 'to' location
        vectorizer.adapt([to_location])
        vectorized_to_location = vectorizer([to_location])
        vectorized_to_location_array = vectorized_to_location.numpy().tolist()[0]

        # Perform KNN search using MongoDB aggregation
        location_entry = None

        pipeline = [
            {
                "$project": {
                    "location": "$location",
                    "coordinates": "$coordinates",
                    "vectorized_name": "$vectorized_name",
                    "distance": {
                        "$sqrt": {
                            "$add": [
                                {
                                    "$pow": [
                                        {
                                            "$subtract": [
                                                {
                                                    "$arrayElemAt": [
                                                        "$vectorized_name",
                                                        0,
                                                    ]
                                                },
                                                vectorized_to_location_array[0],
                                            ]
                                        },
                                        2,
                                    ]
                                },
                                {
                                    "$pow": [
                                        {
                                            "$subtract": [
                                                {
                                                    "$arrayElemAt": [
                                                        "$vectorized_name",
                                                        1,
                                                    ]
                                                },
                                                vectorized_to_location_array[1],
                                            ]
                                        },
                                        2,
                                    ]
                                },
                            ]
                        }
                    },
                }
            },
            {"$sort": {"distance": 1}},
            {"$limit": 1},
        ]

        result = list(collection.aggregate(pipeline))

        if result:
            location_entry = result[0]
            coordinates = location_entry.get("coordinates")
            # Convert ObjectId to string
            location_entry["_id"] = str(location_entry["_id"])
            return {
                "from": current_location_coords,
                "to": to_location,
                "coordinates": coordinates,
                "location_data": location_entry,
            }
        else:
            # If no entry was found, proceed to geocode the 'to' location
            geocode_data = {"latitude": 0.0, "longitude": 0.0}
            # Prepare data for insertion
            location_data = {
                "name": to_location,
                "location": {
                    "department_name": to_location,
                    "floor": None,
                    "block_name": None,
                    "room_no": None,
                },
                "coordinates": {
                    "latitude": geocode_data["latitude"],
                    "longitude": geocode_data["longitude"],
                },
                "vectorized_name": vectorized_to_location_array,
            }

            result = collection.insert_one(location_data)

            return {
                "from": current_location_coords,
                "to": to_location,
                "coordinates": geocode_data,
                "inserted_id": str(result.inserted_id),  # Convert ObjectId to string
                "vectorized_location": vectorized_to_location_array,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New endpoint to extract entities from text input
@app.post("/extract_entities")
async def extract_entities(command: MapCommand):
    try:
        extracted_locations = extract_from_to_locations(command.command)
        return {
            "from": extracted_locations["from_location"],
            "to": extracted_locations["to_location"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/retrieve_DB_uri")
def retrieve_DB_uri():
    return {"DB_URI": db_uri}

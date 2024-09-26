import base64
from bson import ObjectId
import tensorflow as tf
from pydantic import BaseModel, Field
from typing import Optional
from pymongo import MongoClient, ReturnDocument
from fastapi import FastAPI, File, HTTPException, UploadFile
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import json
import os
import typing_extensions as typing
from pydub import AudioSegment
import io
from google.oauth2 import service_account
from google.cloud import speech
import subprocess

# import whisper

# Initialize Google Cloud Speech client
client_file = "hackfest-436404-948ce3ca39f3.json"
credentials = service_account.Credentials.from_service_account_file(client_file)
Client = speech.SpeechClient(credentials=credentials)

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
    landmark: Optional[str] = Field(
        None, description="Nearby landmark"
    )  # New field for landmark


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
        # Extract 'from' and 'to' locations using your custom extraction method
        extracted_locations = extract_from_to_locations(command.command)
        from_location = extracted_locations["from_location"]
        to_location = extracted_locations["to_location"]

        # List of phrases for determining if the user means "my current location"
        current_location_phrases = [
            "my current location",
            "my location",
            "current location",
            "here",
        ]

        # Determine user's current location if mentioned
        if any(phrase in from_location.lower() for phrase in current_location_phrases):
            current_location_coords = {
                "latitude": command.currentLocation.latitude,
                "longitude": command.currentLocation.longitude,
            }
        else:
            current_location_coords = {
                "latitude": 0.0,
                "longitude": 0.0,
            }

        # Vectorize the 'to' location (assuming you have a vectorizer setup)
        vectorizer.adapt([to_location])
        vectorized_to_location = vectorizer([to_location])
        vectorized_to_location_array = vectorized_to_location.numpy().tolist()[0]

        # MongoDB KNN search
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
        else:
            # If no entry found, proceed to geocode or handle the error
            coordinates = {
                "latitude": 0.0,
                "longitude": 0.0,
            }

        print(current_location_coords)
        # Use Google Maps Directions API to fetch directions
        origin = f'{current_location_coords["latitude"]},{current_location_coords["longitude"]}'
        print(origin)
        destination = f'{coordinates["latitude"]},{coordinates["longitude"]}'

        # Construct the Google Maps Directions API URL
        url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={google_maps_api_key}"

        # Make a request to Google Maps Directions API
        response = requests.get(url)
        data = response.json()

        # Check if any routes are available in the API response
        if "routes" in data and data["routes"]:
            # Extract the steps for the directions
            steps = data["routes"][0]["legs"][0]["steps"]
            directions = []

            for step in steps:
                directions.append(
                    {
                        "instructions": step[
                            "html_instructions"
                        ],  # HTML instructions (can be displayed as-is)
                        "distance": step["distance"]["text"],  # Distance for each step
                        "duration": step["duration"][
                            "text"
                        ],  # Time duration for each step
                    }
                )

            # Return both the MongoDB search results and the Google Maps directions
            return {
                "from": current_location_coords,
                "to": to_location,
                "coordinates": coordinates,
                "location_data": location_entry,
                "directions": directions,  # Include the directions in the response
            }
        else:
            # Return an error if no routes are found
            return {"error": "No routes found between the specified locations."}

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


# Pydantic model for response
class Transcription(BaseModel):
    text: str


@app.post("/transcribe", response_model=Transcription)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded audio file (could be .3gp or .wav)
        audio_bytes = await file.read()

        # Save the .3gp file locally
        input_audio_path = "input_audio.3gp"
        with open(input_audio_path, "wb") as f:
            f.write(audio_bytes)

        # Convert .3gp to .wav using FFmpeg, with -y to overwrite without prompt
        output_audio_path = "stereo_audio.wav"
        command = ["ffmpeg", "-y", "-i", input_audio_path, output_audio_path]
        subprocess.run(command, check=True)  # Ensure ffmpeg is installed and available

        # Convert stereo to mono using pydub
        sound = AudioSegment.from_wav(output_audio_path)
        mono_sound = sound.set_channels(1)
        mono_audio_path = "mono_audio.wav"
        mono_sound.export(mono_audio_path, format="wav")

        # Get the sample rate of the audio
        sample_rate = sound.frame_rate

        # Load the mono audio file and recognize speech using Google Cloud Speech API
        with io.open(mono_audio_path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

        # Configure the recognition settings with enhanced model and automatic punctuation
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US",
            sample_rate_hertz=sample_rate,  # Set the sample rate dynamically
            use_enhanced=True,  # Use enhanced model
            enable_automatic_punctuation=True,  # Enable automatic punctuation
            model="default",  # Specify a model, such as "video" or "phone_call" if needed
            speech_contexts=[  # Add relevant phrases for domain-specific optimization
                speech.SpeechContext(
                    phrases=["specific jargon", "special terms"],
                    boost=20.0,  # Boost the probability of these phrases appearing
                )
            ],
        )

        # Call Google Cloud Speech API to recognize speech
        response = Client.recognize(config=config, audio=audio)

        # Extract the transcription from the response
        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript + " "

        # Return transcription
        return Transcription(text=transcription.strip())

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail="Error converting file with FFmpeg."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# whisper_model = whisper.load_model("tiny")


# @app.post("/transcribe-whisper", response_model=Transcription)
# async def transcribe_audio(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded audio file (could be .3gp or .wav)
#         audio_bytes = await file.read()

#         # Save the .3gp file locally
#         input_audio_path = "input_audio.3gp"
#         with open(input_audio_path, "wb") as f:
#             f.write(audio_bytes)

#         # Convert .3gp to .wav using FFmpeg, with -y to overwrite without prompt
#         output_audio_path = "stereo_audio.wav"
#         command = ["ffmpeg", "-y", "-i", input_audio_path, output_audio_path]
#         subprocess.run(command, check=True)  # Ensure ffmpeg is installed and available

#         # Convert stereo to mono using pydub
#         sound = AudioSegment.from_wav(output_audio_path)
#         mono_sound = sound.set_channels(1)
#         mono_audio_path = "mono_audio.wav"
#         mono_sound.export(mono_audio_path, format="wav")

#         # Transcribe the audio using Whisper
#         transcription_result = whisper_model.transcribe(mono_audio_path)

#         # Extract the transcription text from Whisper result
#         transcription_text = transcription_result["text"]

#         # Return the transcription
#         return Transcription(text=transcription_text.strip())

#     except subprocess.CalledProcessError as e:
#         raise HTTPException(
#             status_code=500, detail="Error converting file with FFmpeg."
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

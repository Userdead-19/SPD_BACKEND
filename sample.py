# import requests
# import json

# # Define the parameters
# origin = "11.032603,77.034561"  # Latitude and Longitude for the starting point
# destination = "11.0247,77.002981"  # Latitude and Longitude for the destination
# api_key = "AIzaSyDbrZgv56l76sSmJPzO8wTweIMXRuEPszQ"

# # Construct the URL
# url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={api_key}"


# response = requests.get(url)
# data = response.json()

# print(json.dumps(data["routes"][0]["legs"][0]["steps"][0], indent=2))
# for i in data["routes"][0]["legs"][0]["steps"]:
#     print(i["html_instructions"])
#     print(i["distance"]["text"])
#     print(i["duration"]["text"])
#     print("----")

# def get_geocode_data(location: str):
#     # URL for the Geocoding API
#     base_url = "https://maps.googleapis.com/maps/api/geocode/json"

#     # Define parameters for the API request
#     params = {"address": location, "key": api_key}

#     # Send GET request to the Geocoding API
#     response = requests.get(base_url, params=params)

#     # Parse the response JSON
#     data = response.json()

#     if data["status"] == "OK":
#         # Extract latitude and longitude from the first result
#         lat_lng = data["results"][0]["geometry"]["location"]
#         return {"latitude": lat_lng["lat"], "longitude": lat_lng["lng"]}
#     else:
#         # Handle error (e.g., if the location is not found)
#         raise ValueError(f"Error in Geocoding API: {data['status']}")


# # Example usage:
# location_name = "1600 Amphitheatre Parkway, Mountain View, CA"
# geocode_data = get_geocode_data(location_name)

# print(f"Coordinates for {location_name}: {geocode_data}")

# import json
# import google.generativeai as genai
# import dotenv
# import os
# import requests

# import typing_extensions as typing


# class Output(typing.TypedDict):
#     from_location: str
#     to_location: str


# dotenv.load_dotenv()

# gemini_api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=gemini_api_key)


# def extract_from_to_locations(input_text: str):
#     prompt = f"Extract the 'from' and 'to' locations from the following text: '{input_text}' and return it as JSON."
#     print(f"Prompt: {prompt}")

#     model = genai.GenerativeModel(
#         "gemini-1.5-pro-latest",
#         system_instruction="Extract the 'from' and 'to' locations from the following text and return it as JSON.",
#     )

#     response = model.generate_content(
#         prompt,
#         generation_config=genai.GenerationConfig(
#             response_mime_type="application/json",
#             response_schema=Output,
#         ),
#     )

#     # Print the raw response for debugging
#     print(f"Raw response: {response.text}")

#     try:
#         # Clean the response to extract the JSON
#         if response.text:
#             output = json.loads(response.text)
#             print(f"Extracted data: {output}")
#             return output
#         else:
#             raise ValueError("Gemini API did not return valid data.")
#     except json.JSONDecodeError as json_error:
#         print(f"JSON decoding error: {str(json_error)}")
#         raise ValueError("Failed to decode JSON response.")
#     except Exception as e:
#         print(f"Error with Gemini API: {str(e)}")
#         raise


# # Test the function
# data = extract_from_to_locations(
#     "I am traveling from New York to PSG College of Technology,Coimbatore."
# )


# def get_geocode_data(location: str):
#     base_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=geojson"
#     response = requests.get(base_url)
#     data = response.json()
#     return data


# gecodes = get_geocode_data(data.get("to_location"))

# # print(data.get("from_location"))
# print(gecodes)

# from pydub import AudioSegment
# import io
# from google.oauth2 import service_account
# from google.cloud import speech

# # Convert stereo to mono
# audio_file = "D:\\Exploration\\SPD_BACKEND\\Trailer.wav"
# sound = AudioSegment.from_wav(audio_file)
# mono_sound = sound.set_channels(1)
# mono_sound.export("D:\\Exploration\\SPD_BACKEND\\Trailer_mono.wav", format="wav")

# # Initialize Google Cloud Speech client
# client_file = "hackfest-436404-948ce3ca39f3.json"
# credentials = service_account.Credentials.from_service_account_file(client_file)
# Client = speech.SpeechClient(credentials=credentials)

# # Load the mono audio file and recognize speech
# mono_audio_file = "D:\\Exploration\\SPD_BACKEND\\Trailer_mono.wav"
# with io.open(mono_audio_file, "rb") as audio_file:
#     content = audio_file.read()
#     audio = speech.RecognitionAudio(content=content)

# config = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     language_code="en-US",
# )

# # Recognize speech
# response = Client.recognize(config=config, audio=audio)
# print(response.results[0].alternatives[0].transcript)


import requests

# Replace this with the actual URL of your FastAPI server
url = "http://localhost:8000/transcribe"  # Change the URL if needed

# Path to your audio file
audio_file_path = "D:\\Exploration\\SPD_BACKEND\\Trailer.wav"

# Open the audio file in binary mode
with open(audio_file_path, "rb") as audio_file:
    files = {"file": audio_file}

    # Send a POST request with the audio file
    response = requests.post(url, files=files)

# Print the response from the server
if response.status_code == 200:
    print("Transcription:", response.json()["text"])
else:
    print("Error:", response.text)

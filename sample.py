# import requests
# import json

# # Define the parameters
# origin = "11.032603,77.034561"  # Latitude and Longitude for the starting point
# destination = "11.0247,77.002981"  # Latitude and Longitude for the destination
# api_key = "AIzaSyDbrZgv56l76sSmJPzO8wTweIMXRuEPszQ"

# # Construct the URL
# url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={api_key}"


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

import json
import google.generativeai as genai
import dotenv
import os

import typing_extensions as typing


class Output(typing.TypedDict):
    from_location: str
    to_location: str


dotenv.load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)


def extract_from_to_locations(input_text: str):
    prompt = f"Extract the 'from' and 'to' locations from the following text: '{input_text}' and return it as JSON."
    print(f"Prompt: {prompt}")

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

    # Print the raw response for debugging
    print(f"Raw response: {response.text}")

    try:
        # Clean the response to extract the JSON
        if response.text:
            output = json.loads(response.text)
            return output
        else:
            raise ValueError("Gemini API did not return valid data.")
    except json.JSONDecodeError as json_error:
        print(f"JSON decoding error: {str(json_error)}")
        raise ValueError("Failed to decode JSON response.")
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        raise


# Test the function
print(extract_from_to_locations("from New York to Los Angeles"))

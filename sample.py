from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")
location = geolocator.geocode("YBlock first floor, Coimbatore")

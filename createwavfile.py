from gtts import gTTS
from pydub import AudioSegment

# Text to be converted into speech
text = "Hello, I am Abinav"

# Generate speech from text using gTTS
tts = gTTS(text, lang="en")

# Save the speech as an mp3 file first (gTTS outputs mp3 by default)
tts.save("speech.mp3")

# Convert mp3 to wav using pydub
audio = AudioSegment.from_mp3("speech.mp3")
audio.export("speech.wav", format="wav")

print("Generated speech.wav with the text 'Hello, I am Abinav'")

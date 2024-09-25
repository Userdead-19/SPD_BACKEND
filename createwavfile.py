import os
from gtts import gTTS
import subprocess


def text_to_speech_3gp(text, output_filename="output.3gp"):
    try:
        # Convert text to speech and save as a temporary mp3 file
        tts = gTTS(text)
        temp_mp3 = "temp_audio.mp3"
        tts.save(temp_mp3)

        # Convert mp3 to .3gp using ffmpeg
        command = [
            "ffmpeg",
            "-i",
            temp_mp3,
            "-c:a",
            "aac",
            "-b:a",
            "32k",
            output_filename,
        ]
        subprocess.run(command, check=True)

        # Remove the temporary mp3 file
        os.remove(temp_mp3)
        print(f"Audio successfully saved as {output_filename}")

    except subprocess.CalledProcessError:
        print("Error during the conversion to .3gp format.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Input text to be converted to voice
    text = "Hello, this is a test voice message encoded in 3GP format."

    # Call the function to create .3gp file
    text_to_speech_3gp(text, "output_audio.3gp")

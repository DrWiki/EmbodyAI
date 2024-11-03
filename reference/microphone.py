import speech_recognition as sr

def recognize_speech_from_mic():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Listening...")

        # Capture the audio from the microphone
        audio = recognizer.listen(source)

    # Attempt to recognize the speech in the audio
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio, language='zh-CN')
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    return None

if __name__ == "__main__":
    recognize_speech_from_mic()

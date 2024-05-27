import pyttsx3
import speech_recognition as sr
import pyttsx3
import speech_recognition as sr

# import pyttsx3
# import speech_recognition as sr

# engine = pyttsx3.init()

# def speech_to_text():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         audio = recognizer.listen(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         print("Sorry, I didn't catch that.")
#     except sr.RequestError as e:
#         print(f"Error: {e}")

# def text_to_speech(text):
#     engine.say(text)
#     engine.runAndWait()
    
    
    
    
    
    
    
    
    
    
    
engine = pyttsx3.init()

def speech_to_text():
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1  # seconds of non-speaking audio before a phrase is considered complete
        with sr.Microphone(device_index=2) as source:  # using the "Headset (Rockerz 255 Pro+)"
            recognizer.adjust_for_ambient_noise(source)  # adjust for ambient noise
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
        except sr.RequestError as e:
            print(f"Error: {e}")

def text_to_speech(text):
        engine.say(text)
        engine.runAndWait()

def transcribe_audio_to_text():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
            try:
                print("Recognizing...")
                text = r.recognize_google(audio)
                return text
            except Exception as e:
                print(e)
                return ""

def speak_text(text):
        # Your code to convert text to speech goes here
        pass

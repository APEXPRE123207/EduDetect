import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import pyttsx3


# Flag to track success
fl = False  

# Function to convert text to speech and play it
def speech(text, shape, color):
    global fl  # Keep track of success

    text_speech = pyttsx3.init()
    voices = text_speech.getProperty('voices')
    text_speech.setProperty('voice', voices[1].id)
    text_speech.say(text)
    text_speech.runAndWait()

    if fl: 
        fl=False # If the user already succeeded, don't listen again
        return

    user_audio = record_audio()  # Record only when needed
    recognized_text = recognize_speech(user_audio)

    if recognized_text:
        recognized_text = recognized_text.lower()
        if shape.lower() in recognized_text and color.lower() in recognized_text:
            print("User spoke correctly!")
            fl = True  # Stop further retries
            speech("Good Job!", shape, color)
        else:
            print("User did not say the correct shape or color.")
            speech("Try Again", shape, color)


# Function to record audio from user
def record_audio(duration=5, sample_rate=44100, channels=1):
    print("Listening... Speak now!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype=np.int16)
    sd.wait()

    # Save as WAV file
    wav_file = "user_input.wav"
    with wave.open(wav_file, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.writeframes(recording.tobytes())

    return wav_file

# Function to recognize speech from recorded audio
def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None



import speech_recognition as sr

r = sr.Recognizer()

# test = sr.AudioFile('../tests/english.wav')
test = sr.AudioFile('../tests/ZYD01.wav')

with test as source:
    audio = r.record(source)

# said = r.recognize_google(audio, language='en-US')
said = r.recognize_google(audio, language='zh-CN')
print("google think you said:",said)

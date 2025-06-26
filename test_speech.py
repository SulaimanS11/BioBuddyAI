import pyttsx3
engine = pyttsx3.init(driverName='nsss')
engine.setProperty('rate', 150)
engine.say("This is a test of the emergency broadcast system.")
engine.runAndWait()

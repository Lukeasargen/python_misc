import pyttsx3 # pip install pyttsx3
engine = pyttsx3.init() # object creation
engine.setProperty('rate', 150)     # setting up new voice rate
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1
engine.setProperty('voice', engine.getProperty('voices')[0].id)  #changing index, changes voices. o for male

engine.say("Hello World!")
engine.runAndWait()
engine.stop()

# On linux make sure that 'espeak' and 'ffmpeg' are installed
# engine.save_to_file('Hello World', 'test.mp3')
# engine.runAndWait()
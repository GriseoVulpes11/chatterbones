import playsound
from gtts import gTTS
import os

# converts a given string into a tts .mp3 file and reads it using playsound
# check the following link for a list of tld's (localized 'accents')
# https://gtts.readthedocs.io/en/latest/module.html#localized-accents
def tts(text, LANG="en", TLD='us', TTS_FILENAME='tts.mp3'):
    try:
        # tts = gTTS(text=text, lang=LANG, tld=TLD)
        tts = gTTS(text=text, lang=LANG)

        tts.save(TTS_FILENAME)
        playsound.playsound(TTS_FILENAME)

        # remove the file to prevent 'Exception: [Errno 13] Permission denied: testing.mp3'
        # also just gets rid of the file to prevent excess
        # idea: could have the files save by datetime so they have unique names if wanted all of them
        os.remove(TTS_FILENAME)
        
    # this will occur if an empty string is inputed
    except Exception as e:
        print("Exception: " + str(e))
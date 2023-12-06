import os
import datetime

def tts(text):
      return os.system(text)

tts("espeak-ng  'I am a test statement to try some voices'")

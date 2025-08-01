from gtts import gTTS
import os

def generate_tts(text, lang='en', tld='co.uk'):
    tts = gTTS(text=text, lang=lang, tld=tld)
    filename = f"static/audio_{hash(text)}.mp3"
    tts.save(filename)
    return filename

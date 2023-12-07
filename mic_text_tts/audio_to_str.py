import pyaudio
import wave
import speech_recognition as sr

# records audio of a given length and saves it as a .wav file
def record_audio(CHUNK=1024, FORMAT=pyaudio.paInt16, CHANNELS=2, RATE=44100, RECORD_SECONDS=5, WAVE_OUTPUT_FILENAME='voice.wav'):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# converts wav to text and saves to a text file. Also returns the text as a string
def wav_to_text(audio_file, TXT_FILENAME='said.txt'):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

        said = ''

        try:
            said = r.recognize_google(audio)


        except Exception as e:
            print(f"No words spoken: {e}")

        # write to txt file if wav file has words
        if said != '':
            with open(TXT_FILENAME, 'w') as file:
                file.write(said)

    return said


wav_name = 'voice.wav' # this is the same as the default
txt_name = '/Users/riley/Desktop/said.txt' # this is the same as the default
recording_time = 10 # change this number to change the length of recordings

while(1):
    try:
        record_audio(RECORD_SECONDS=recording_time, WAVE_OUTPUT_FILENAME=wav_name)
        print('audio recorded')
        wav_to_text(wav_name,TXT_FILENAME=txt_name)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError as e:
        print(f"unknown error occurred: {e}")
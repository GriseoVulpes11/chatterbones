import requests

from mic_text_tts.str_to_tts import tts
import time


def main():
    while True:
        response = requests.get("http://35.175.223.75:8080")

        tts(response.text)
        time.sleep(5)


if __name__ == '__main__':
    main()

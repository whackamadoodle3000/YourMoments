import torch
from TTS.api import TTS
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


print(TTS().list_models())

# List available 🐸TTS models and choose the first one
model_name = "tts_models/en/ljspeech/neural_hmm"
model_name = "tts_models/en/blizzard2013/capacitron-t2-c50"
# Init TTS
tts = TTS(model_name).to(device)

# Run TTS
# Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
# wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# Text to speech to a file
tts.tts_to_file(text=""" 

Closing with gratitude, the video journeys through bathrooms—a black and white door, yellow trash can, sinks and mirrors. A blur in hallways, public restrooms with urinals, a selfie moment. Thanks for watching!

""", file_path="output.wav")

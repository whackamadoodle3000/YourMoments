from elevenlabs import set_api_key,generate,save
set_api_key("451d92856aedd2d73795a00f75e253f9")


audio = generate(
  text="Closing with gratitude, the video journeys through bathrooms—a black and white door, yellow trash can, sinks and mirrors. A blur in hallways, public restrooms with urinals, a selfie moment. Thanks for watching!",
  voice="Callum",
  model="eleven_multilingual_v2"
)

save(
    audio,  "output_eleven.wav"
)
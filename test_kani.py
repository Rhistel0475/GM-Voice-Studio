from kani_tts import KaniTTS

model = KaniTTS("nineninesix/kani-tts-2-en")

# pick one: en_us, en_nyork, en_oakl, en_glasg, en_bost, en_scou
audio, text = model(
    "Hello! Kani TTS is working with the New York tag.",
    language_tag="en_nyork",
)
model.save_audio(audio, "test.wav")
print("Wrote test.wav")

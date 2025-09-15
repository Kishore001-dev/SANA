"""
Dr. Sarah - Voice-enabled Psychiatric Consultation Agent
Accepts Mediator Agent output directly as a dict.
Supports both voice input/output and typed input.
Robustly handles exit commands with punctuation and extra words.
"""

import json
import os
import string
from tempfile import NamedTemporaryFile
import sounddevice as sd
import soundfile as sf
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ======= CONFIG =======
OPENAI_API_KEY = "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"
LLM_MODEL = "gpt-4o-mini"
openai.api_key = OPENAI_API_KEY

# ======= VOICE HELPERS =======
class VoiceIO:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def speak(self, text: str):
        """Speak text using TTS; fallback to console if audio fails."""
        print(f"Dr. Sarah: {text}")
        try:
            response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                response.stream_to_file(f.name)  # Deprecation warning is ignored
                filename = f.name

            data, samplerate = sf.read(filename)
            sd.play(data, samplerate)
            sd.wait()
            os.unlink(filename)  # corrected
        except Exception:
            print(f"Dr. Sarah (text fallback): {text}")

    def listen(self, duration: float = 10, samplerate: int = 16000) -> str:
        """
        Listen for user voice input. Falls back to typed input if microphone fails.
        Returns the transcribed text.
        """
        try:
            print("🎤 Listening... speak now (or press Enter to type)")
            audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
            sd.wait()

            with NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, audio, samplerate)
                filename = f.name

            with open(filename, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)

            os.unlink(filename)  # corrected
            text = transcript.text.strip()
            if text == "":
                text = input("🖊️ Please type your response: ").strip()
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"❌ Microphone/recording error: {e}")
            text = input("🖊️ Please type your response: ").strip()
            return text

# ======= Dr. Sarah Agent =======
class DrSarahAgent:
    EXIT_KEYWORDS = ["exit", "quit", "bye", "goodbye", "さようなら", "バイ"]

    def __init__(self, mediator_output: dict, user_name: str = "Jackson"):
        """
        mediator_output: direct Mediator Agent output (dict)
        """
        self.user_name = user_name
        self.mediator_data = mediator_output
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.6)
        self.voice = VoiceIO(OPENAI_API_KEY)

    @staticmethod
    def _is_exit_command(text: str) -> bool:
        """Return True if the text contains an exit keyword."""
        text_clean = text.translate(str.maketrans("", "", string.punctuation)).lower()
        return any(word in text_clean for word in DrSarahAgent.EXIT_KEYWORDS)

    def answer_question(self, user_question: str) -> str:
        system_prompt = f"""
You are Dr. Sarah, a compassionate and professional psychiatrist.
You are advising a patient named {self.user_name}.
You have access to the following Mediator Agent assessment output:

{json.dumps(self.mediator_data, indent=2)}

Answer the patient's questions naturally, empathetically, and clearly.
Speak directly to the patient. Never reveal you are an AI.
"""
        system = SystemMessage(content=system_prompt)
        human = HumanMessage(content=f"Patient asks: {user_question}")

        resp = self.llm.invoke([system, human])
        return resp.content.strip()

    def start_consultation(self):
        self.voice.speak(f"Hello {self.user_name}, I am Dr. Sarah. How are you feeling today?")
        while True:
            user_input = self.voice.listen()
            if self._is_exit_command(user_input):
                self.voice.speak("Take care! Remember, your wellbeing is important. Goodbye.")
                break
            response = self.answer_question(user_input)
            self.voice.speak(response)

# ======= Example usage =======
if __name__ == "__main__":
    # Example Mediator output
    mediator_output = {
      "categories": ["Hamilton-D", "Hamilton-A", "PSS"],
      "reason": "The patient shows elevated GHQ-12 score with symptoms of concentration difficulties, sleep disturbance due to worry, feeling under strain, occasional depressed mood, and loss of confidence. Emotion analysis reveals high emotional variability with dominant sadness and frequent negative emotions including fear and anger, indicating significant depressive and anxiety symptoms alongside stress. These findings justify further assessment with Hamilton Depression and Anxiety scales and Perceived Stress Scale to clarify severity and guide management.",
      "risk_level": "Moderate"
    }

    dr_sarah = DrSarahAgent(mediator_output, user_name="Jackson")
    dr_sarah.start_consultation()

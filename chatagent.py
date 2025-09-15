import os
import time
import json
import threading
import random
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any
import sounddevice as sd
import soundfile as sf
import openai

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from deepface_SANA import DeepFaceEmotionAgent
import mediator_agent

# ======= CONFIG =======
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ======= GHQ-12 QUESTIONS =======
GHQ_QUESTIONS: List[str] = [
    "Have you been able to concentrate well on what you’re doing?",
    "Have you felt you were playing a useful part in things?",
    "Have you felt capable of making decisions about things?",
    "Have you been able to enjoy your normal day-to-day activities?",
    "Have you been able to face up to your problems?",
    "Have you been feeling reasonably happy, all things considered?",
    "Have you lost much sleep over worry?",
    "Have you felt constantly under strain?",
    "Have you felt you couldn’t overcome your difficulties?",
    "Have you been feeling unhappy and depressed?",
    "Have you been losing confidence in yourself?",
    "Have you been thinking of yourself as a worthless person?",
]

# ======= GLOBAL LOCK =======
audio_console_lock = threading.Lock()

# ======= VOICE I/O =======
class VoiceIO:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def speak(self, text: str):
        with audio_console_lock:
            print(f"🤖 Agent: {text}")
            try:
                response = openai.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text
                )
                with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    response.stream_to_file(f.name)
                    filename = f.name
                data, samplerate = sf.read(filename)
                sd.play(data, samplerate)
                sd.wait()
                time.sleep(0.18)
                os.unlink(filename)
            except Exception as e:
                print(f"❌ TTS error: {e}")
                print("🔊 (Text only fallback)")

    def listen(self, duration: int = 10, samplerate: int = 16000) -> str:
        with audio_console_lock:
            print("🎤 Listening... speak now")
            try:
                audio = sd.rec(int(duration * samplerate),
                               samplerate=samplerate,
                               channels=1,
                               dtype="float32")
                sd.wait()
                with NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    sf.write(f.name, audio, samplerate)
                    filename = f.name
                with open(filename, "rb") as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                os.unlink(filename)
                text = transcript.text
                print(f"👤 You said: {text}")
                time.sleep(0.12)
                return text
            except Exception as e:
                print(f"❌ STT error: {e}")
                user_input = input("🖊️ Please type your response: ")
                print(f"👤 You typed: {user_input}")
                return user_input

# ======= LLM EVALUATOR =======
class LLMEvaluator:
    def __init__(self, api_key: str, model: str = LLM_MODEL, temperature: float = 0.0):
        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

    def interpret_answer(self, question: str, user_answer: str) -> Dict[str, Any]:
        system = SystemMessage(content=(
            "You are a clinical assistant mapping free-text responses to "
            "GHQ-12 binary scoring (0 or 1). Positively phrased items → 0 if good, 1 if problem. "
            "Negatively phrased items → 1 if symptom present, else 0. "
            "Return JSON: {\"score\": 0, \"reason\": \"...\"}"
        ))
        human = HumanMessage(content=f"Question: {question}\nAnswer: {user_answer}\nReturn JSON mapping.")
        try:
            resp = self.llm.invoke([system, human])
            content = resp.content.strip()
            start, end = content.find("{"), content.rfind("}") + 1
            if start != -1 and end != -1:
                parsed = json.loads(content[start:end])
                return {"score": int(parsed.get("score", 0)), "reason": parsed.get("reason", ""), "raw": content}
        except Exception as e:
            print("LLM interpret error:", e)
        return {"score": 0, "reason": "fallback", "raw": user_answer}

    def natural_ack(self, question: str, answer: str, score: int) -> str:
        positive = [
            "Thanks for sharing — I've noted that.",
            "Got it — I recorded your response.",
            "Thanks — your answer is saved.",
            "Great, your response has been recorded.",
            "Understood, I've logged that answer.",
            "Perfect, thanks for letting me know.",
            "Thanks — I’ve noted your input.",
            "All set, your answer is captured."
        ]
        concern = [
            "I hear you — that indicates some concern.",
            "Thanks for being honest — I recorded it.",
            "Understood — I'll include this in the assessment.",
            "Noted — we’ll pay attention to this in your evaluation.",
            "I see — this suggests something we should monitor.",
            "Thanks for sharing — it’s important for your assessment.",
            "Understood, I've recorded this carefully.",
            "Got it — this will be considered in your overall review."
        ]
        return random.choice(positive if score == 0 else concern)
# ======= MAIN AGENT =======
class GHQVoiceAgent:
    def __init__(self):
        self.voice = VoiceIO(api_key=OPENAI_API_KEY)
        self.evaluator = LLMEvaluator(api_key=OPENAI_API_KEY)
        self.deepface_agent = DeepFaceEmotionAgent()
        self.ghq_answers: List[Dict[str, Any]] = []
        self.total_score = 0
        self.session_id = f"ghq_session_{int(time.time())}"

    def start_deepface(self):
        t = threading.Thread(target=self.deepface_agent.start, daemon=True)
        t.start()
        time.sleep(1.5)

    def stop_deepface(self):
        try:
            self.deepface_agent.stop()
        except Exception as e:
            print("Error stopping DeepFace:", e)

    def run_assessment(self):
        self.start_deepface()
        self.voice.speak("Hello. I will ask you 12 general health questions. Answer naturally. Say 'repeat' to hear a question again.")

        for idx, q in enumerate(GHQ_QUESTIONS, start=1):
            if hasattr(self.deepface_agent, 'suspend_analysis'):
                self.deepface_agent.suspend_analysis = True

            while True:
                self.voice.speak(f"Question {idx} of 12: {q}")
                answer = self.voice.listen()

                if not answer.strip():
                    self.voice.speak("I didn’t catch that. Please try again.")
                    continue

                if answer.strip().lower() in ("repeat", "say again", "again"):
                    continue

                mapped = self.evaluator.interpret_answer(q, answer)
                score = mapped.get("score", 0)

                self.ghq_answers.append({
                    "question": q,
                    "answer": answer,
                    "score": score,
                    "reason": mapped.get("reason", ""),
                })
                self.total_score += score

                ack = self.evaluator.natural_ack(q, answer, score)
                self.voice.speak(ack)
                break

            if hasattr(self.deepface_agent, 'suspend_analysis'):
                time.sleep(0.25)
                self.deepface_agent.suspend_analysis = False

        self.voice.speak("Thank you. Finalizing your report.")
        self.stop_deepface()

        emotion_summary = self._collect_emotion_report()
        ghq_summary = self._build_ghq_summary()

        self.voice.speak("Sending results to the clinical mediator.")
        classification = mediator_agent.classify_user(ghq_summary, emotion_summary)

        print("📊 Mediator classification:", json.dumps(classification, indent=2))
        self.voice.speak("Assessment complete.")

        combined = {
            "session_id": self.session_id,
            "ghq_summary": ghq_summary,
            "emotion_summary": emotion_summary,
            "mediator_classification": classification,
            "timestamp": time.time()
        }
        with open(f"{self.session_id}.json", "w") as f:
            json.dump(combined, f, indent=2)
        return combined

    def _build_ghq_summary(self) -> Dict[str, Any]:
        risk = "elevated" if self.total_score >= 3 else "normal"
        return {"total_score": self.total_score, "risk_level": risk, "question_details": self.ghq_answers}

    def _collect_emotion_report(self) -> Dict[str, Any]:
        try:
            filename = getattr(self.deepface_agent, "output_filename", None)
            if filename and os.path.exists(filename):
                with open(filename, "r") as f:
                    return json.load(f)
            elif hasattr(self.deepface_agent, "session_reports"):
                return {
                    "session_metadata": {"session_id": getattr(self.deepface_agent, "session_id", None),
                                         "start_time": str(getattr(self.deepface_agent, "session_start", ""))},
                    "emotion_reports": getattr(self.deepface_agent, "session_reports", []),
                }
        except Exception as e:
            print("Error collecting emotion report:", e)
        return {"session_metadata": {}, "emotion_reports": []}

if __name__ == "__main__":
    agent = GHQVoiceAgent()
    agent.run_assessment()

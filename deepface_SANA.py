import cv2
import time
import json
import signal
import sys
import atexit
from datetime import datetime
from deepface import DeepFace
import threading
from collections import deque
import queue
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import numpy as np  # needed for serializer

# ===== JSON Serializer =====
def default_serializer(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)

# Define the state structure for LangGraph
class EmotionAnalysisState(TypedDict):
    emotion_batch: List[Dict[str, Any]]
    start_time: str
    end_time: str
    emotion_counts: Dict[str, int]
    ai_analysis: str
    patterns: str
    report_id: int
    session_reports: List[Dict[str, Any]]

class EmotionAnalysisAgent:
    """LangGraph-based AI Agent for emotion analysis"""
    
    def __init__(self, openai_api_key: str | None):
        self.llm = None
        if openai_api_key:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4o-mini",
                temperature=0.7
            )
        self.graph = self._create_analysis_graph()
    
    def _create_analysis_graph(self):
        def process_emotion_data(state: EmotionAnalysisState) -> EmotionAnalysisState:
            emotion_batch = state["emotion_batch"]
            if not emotion_batch:
                return state
            state["start_time"] = emotion_batch[0]['timestamp']
            state["end_time"] = emotion_batch[-1]['timestamp']
            emotion_counts = {}
            for data in emotion_batch:
                emotion = data['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            state["emotion_counts"] = emotion_counts
            return state
        
        def analyze_patterns(state: EmotionAnalysisState) -> EmotionAnalysisState:
            emotion_batch = state["emotion_batch"]
            if len(emotion_batch) < 2:
                state["patterns"] = "Insufficient data for pattern analysis"
                return state
            patterns = []
            emotions = [data['dominant_emotion'] for data in emotion_batch]
            unique_emotions = set(emotions)
            if len(unique_emotions) == 1:
                patterns.append(f"Stable emotional state: {emotions[0]}")
            elif len(unique_emotions) <= 3:
                patterns.append(f"Low emotional variability: {', '.join(unique_emotions)}")
            else:
                patterns.append(f"High emotional variability: {len(unique_emotions)} different emotions")
            first_half = emotions[:len(emotions)//2]
            second_half = emotions[len(emotions)//2:]
            if set(first_half) != set(second_half):
                patterns.append("Emotional shift detected between first and second half")
            negative_emotions = ['sad', 'angry', 'fear', 'disgust']
            negative_count = sum(1 for e in emotions if e in negative_emotions)
            if negative_count > len(emotions) * 0.6:
                patterns.append("High negative emotion prevalence detected")
            state["patterns"] = "; ".join(patterns)
            return state
        
        def ai_emotional_analysis(state: EmotionAnalysisState) -> EmotionAnalysisState:
            emotion_batch = state["emotion_batch"]
            emotion_counts = state["emotion_counts"]
            patterns = state["patterns"]
            if self.llm is None:
                state["ai_analysis"] = (
                    "AI analysis unavailable (no OPENAI_API_KEY). "
                    f"Pattern-based summary: {patterns}"
                )
                return state
            emotion_timeline = []
            for data in emotion_batch:
                time_str = data['timestamp'][-8:]
                emotion_timeline.append(f"{time_str}: {data['dominant_emotion']}")
            system_message = SystemMessage(content="""
            You are an AI specialist in emotional analysis for mental health assessments. 
            Analyze emotion data and provide professional insights.
            """)
            human_message = HumanMessage(content=f"""
            Time Period: {state['start_time']} to {state['end_time']}
            Total Data Points: {len(emotion_batch)}

            Emotion Timeline:
            {chr(10).join(emotion_timeline[:10])}...

            Emotion Distribution:
            {json.dumps(emotion_counts, indent=2)}

            Detected Patterns:
            {patterns}

            Provide professional analysis with:
            - EMOTIONAL STATE
            - KEY OBSERVATIONS
            - MENTAL HEALTH INDICATORS
            - THERAPEUTIC RECOMMENDATIONS
            """)
            try:
                response = self.llm.invoke([system_message, human_message])
                state["ai_analysis"] = response.content
            except Exception as e:
                state["ai_analysis"] = f"AI analysis failed: {str(e)}. Fallback: {patterns}"
            return state
        
        def generate_report(state: EmotionAnalysisState) -> EmotionAnalysisState:
            emotion_batch = state["emotion_batch"]
            report = {
                'report_id': state.get("report_id", 1),
                'timestamp': datetime.now().isoformat(),
                'time_period': {
                    'start': state["start_time"],
                    'end': state["end_time"],
                    'duration_seconds': len(emotion_batch)
                },
                'emotion_analysis': {
                    'distribution': state["emotion_counts"],
                    'dominant_emotion': max(state["emotion_counts"].items(), key=lambda x: x[1]) if state["emotion_counts"] else ("unknown", 0),
                    'emotional_variability': len(set(state["emotion_counts"].keys())),
                    'total_samples': len(emotion_batch)
                },
                'pattern_analysis': state["patterns"],
                'ai_professional_analysis': state["ai_analysis"]
            }
            if 'session_reports' not in state:
                state['session_reports'] = []
            state['session_reports'].append(report)
            return state
        
        workflow = StateGraph(EmotionAnalysisState)
        workflow.add_node("process_data", process_emotion_data)
        workflow.add_node("analyze_patterns", analyze_patterns)
        workflow.add_node("ai_analysis", ai_emotional_analysis)
        workflow.add_node("generate_report", generate_report)
        workflow.set_entry_point("process_data")
        workflow.add_edge("process_data", "analyze_patterns")
        workflow.add_edge("analyze_patterns", "ai_analysis")
        workflow.add_edge("ai_analysis", "generate_report")
        workflow.add_edge("generate_report", END)
        return workflow.compile()

class DeepFaceEmotionAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.running = False
        self.camera = None
        self.emotion_data = deque(maxlen=1000)
        self.batch_processor = queue.Queue()
        self.ai_agent = EmotionAnalysisAgent(self.openai_api_key)
        self.session_start = datetime.now()
        self.session_id = f"emotion_session_{int(time.time())}"
        self.session_reports = []
        self.report_counter = 1
        self.camera_thread = None
        self.analysis_thread = None
        self.output_filename = f"mediator_emotion_data_{int(time.time())}.json"
        print(f"DeepFace Emotion Agent initialized (Session: {self.session_id})")
        if self.openai_api_key is None:
            print("Note: OPENAI_API_KEY not set. AI analysis will be skipped.")
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        if self.running:
            print("Agent already running!")
            return
        print("Starting DeepFace emotion monitoring...")
        self.running = True
        self.session_start = datetime.now()
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("DeepFace agent started successfully!")
        print(f"Output will be saved to: {self.output_filename}")
    
    def _camera_loop(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not open camera")
                return
            print("Camera initialized. Starting emotion detection...")
            frame_count = 0
            last_analysis_time = time.time()
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    break
                frame_count += 1
                current_time = time.time()
                if frame_count % 5 == 0:
                    try:
                        result = DeepFace.analyze(
                            img_path=frame,
                            actions=['emotion'],
                            enforce_detection=False
                        )
                        if isinstance(result, list) and len(result) > 0:
                            emotions = result[0]['emotion']
                            dominant_emotion = result[0]['dominant_emotion']
                        else:
                            emotions = result['emotion']
                            dominant_emotion = result['dominant_emotion']
                        emotion_record = {
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': frame_count,
                            'dominant_emotion': dominant_emotion,
                            'emotion_scores': emotions
                        }
                        self.emotion_data.append(emotion_record)
                        if current_time - last_analysis_time >= 30:
                            self._trigger_batch_analysis()
                            last_analysis_time = current_time
                    except Exception as e:
                        print(f"Emotion detection error: {e}")
                time.sleep(0.1)
        except Exception as e:
            print(f"Camera loop error: {e}")
        finally:
            if self.camera:
                self.camera.release()
    
    def _analysis_loop(self):
        while self.running:
            try:
                batch_data = self.batch_processor.get(timeout=1)
                if batch_data == "ANALYZE":
                    self._perform_batch_analysis()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis loop error: {e}")
    
    def _trigger_batch_analysis(self):
        if len(self.emotion_data) > 0:
            self.batch_processor.put("ANALYZE")
    
    def _perform_batch_analysis(self):
        try:
            recent_data = list(self.emotion_data)[-60:]
            if len(recent_data) < 5:
                return
            print(f"Analyzing batch of {len(recent_data)} emotion samples...")
            initial_state = EmotionAnalysisState(
                emotion_batch=recent_data,
                start_time="",
                end_time="",
                emotion_counts={},
                ai_analysis="",
                patterns="",
                report_id=self.report_counter,
                session_reports=self.session_reports
            )
            result_state = self.ai_agent.graph.invoke(initial_state)
            if result_state['session_reports']:
                latest_report = result_state['session_reports'][-1]
                self.session_reports.append(latest_report)
                self.report_counter += 1
                print(f"Analysis complete. Dominant emotion: {latest_report['emotion_analysis']['dominant_emotion'][0]}")
        except Exception as e:
            print(f"Batch analysis error: {e}")
    
    def stop(self):
        if not self.running:
            return
        print("Stopping emotion monitoring...")
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=5)
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)
        if len(self.emotion_data) > 0:
            print("Performing final emotion analysis...")
            self._perform_batch_analysis()
        self._generate_final_report()
        print(f"Emotion monitoring stopped. Report saved to: {self.output_filename}")
    
    def _generate_final_report(self):
        try:
            session_end = datetime.now()
            session_duration = (session_end - self.session_start).total_seconds() / 60
            final_report = {
                'session_metadata': {
                    'session_id': self.session_id,
                    'start_time': self.session_start.isoformat(),
                    'end_time': session_end.isoformat(),
                    'session_duration_minutes': round(session_duration, 2),
                    'total_emotion_samples': len(self.emotion_data),
                    'total_reports_generated': len(self.session_reports)
                },
                'emotion_reports': self.session_reports,
                'raw_emotion_data': list(self.emotion_data)[-100:] if self.emotion_data else [],
                'summary': {
                    'session_complete': True,
                    'data_quality': 'good' if len(self.emotion_data) > 50 else 'limited',
                    'analysis_reports': len(self.session_reports)
                }
            }
            with open(self.output_filename, 'w') as f:
                json.dump(final_report, f, indent=2, default=default_serializer)
            print(f"Final report saved: {self.output_filename}")
            print(f"Session duration: {session_duration:.1f} minutes")
            print(f"Total emotion samples: {len(self.emotion_data)}")
            print(f"Analysis reports generated: {len(self.session_reports)}")
        except Exception as e:
            print(f"Error generating final report: {e}")
    
    def _cleanup(self):
        if self.running:
            self.stop()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

def main():
    print("DeepFace Emotion Agent - Mental Health Assessment")
    print("=" * 60)
    print("This agent will monitor emotions via webcam during the assessment")
    print("Press Ctrl+C to stop monitoring and generate report")
    print("-" * 60)
    agent = DeepFaceEmotionAgent()
    try:
        agent.start()
        while agent.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        agent.stop()

if __name__ == "__main__":
    main()

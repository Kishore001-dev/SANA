"""
Enhanced Mediator Agent
-----------------------
Supports both synchronous and asynchronous integration patterns:

1. Direct function call: classify_user(ghq, emotion) - for single-process scenarios
2. Synchronizer pattern: MediatorSynchronizer - for multi-process scenarios  
3. REST API endpoints - for distributed scenarios (optional)

This provides maximum flexibility for different integration approaches.
"""

import os, json, time, threading, queue
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, Union,List
from langchain_openai import ChatOpenAI
from dr_sarah_agent import DrSarahAgent

# ---------------------------------------------------------------------
# DB CONFIG 
# ---------------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "aws-1-ap-southeast-1.pooler.supabase.com"),
    "port": int(os.getenv("PG_PORT", "6543")),
    "database": os.getenv("PG_DB", "postgres"),
    "user": os.getenv("PG_USER", "postgres.yenlazyrcccxgcscsnai"),
    "password": os.getenv("PG_PASSWORD", "Jackson@203")
}
OPENAI_API_KEY = "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"

# ---------------------------------------------------------------------
# Fetch reference text from Postgres
# ---------------------------------------------------------------------
def fetch_reference_text(limit: int = 6) -> str:
    """Pull recent vectorized text chunks from the 'mediator' table."""
    query = """
        SELECT content
        FROM mediator
        ORDER BY id DESC
        LIMIT %s;
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
                return "\n\n".join(r["content"] for r in rows if "content" in r)
    except Exception as e:
        print(f"❌ Postgres fetch error: {e}")
        return ""

# ---------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------
def _perform_classification(ghq_summary: Dict[str, Any], emotion_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal classification logic - used by all integration patterns.
    """
    reference_text = fetch_reference_text()
    
    prompt = f"""
You are a clinical decision-support AI.

Reference diagnostic material:
{reference_text}

Patient data:
1. GHQ-12 summary: {json.dumps(ghq_summary)}
2. Emotion-analysis summary: {json.dumps(emotion_summary)}

GHQ scoring rules:
- Each GHQ item is scored 0 or 1
- Positively phrased items: 0 if functioning well, 1 if problems
- Negatively phrased items: 1 if symptom present, else 0

Task:
- Decide whether the patient should receive further assessment with:
    • Hamilton Depression Rating Scale (Hamilton-D)
    • Hamilton Anxiety Rating Scale (Hamilton-A)
    • Perceived Stress Scale (PSS)
- Return any combination (one, two, or all three) that fits the evidence.
- Provide a short, professional explanation.

Output ONLY valid JSON:
{{
  "categories": ["Hamilton-A", "PSS"],
  "reason": "Concise clinical justification",
  "risk_level": "Severe" or "Moderate" or "Mild"
}}
"""
    
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, api_key=OPENAI_API_KEY)
    resp = llm.invoke(prompt)
    
    # Parse response
    try:
        return json.loads(resp.content)
    except Exception:
        start = resp.content.find("{")
        end = resp.content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(resp.content[start:end])
        else:
            # Fallback response
            return {
                "categories": [],
                "reason": "Unable to parse AI response",
                "risk_level": "Unknown"
            }

# ---------------------------------------------------------------------
# INTEGRATION PATTERN 1: Direct Function Call (Synchronous)
# ---------------------------------------------------------------------
def classify_user(ghq_summary: Dict[str, Any], emotion_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct synchronous classification - use when you have both datasets ready.
    
    Args:
        ghq_summary: GHQ-12 assessment results
        emotion_summary: Emotion analysis results
        
    Returns:
        Classification result with categories and reasoning
    """
    print("🔄 Processing classification request (synchronous)...")
    
    # Validate inputs
    if not ghq_summary or not emotion_summary:
        return {
            "categories": [],
            "reason": "Incomplete input data provided",
            "risk_level": "Unknown",
            "error": "Missing GHQ or emotion data"
        }
    
    try:
        result = _perform_classification(ghq_summary, emotion_summary)
        print("✅ Classification completed successfully")
        return result
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return {
            "categories": [],
            "reason": f"Classification failed: {str(e)}",
            "risk_level": "Unknown",
            "error": str(e)
        }

# ---------------------------------------------------------------------
# INTEGRATION PATTERN 2: Asynchronous Synchronizer
# ---------------------------------------------------------------------
class MediatorSynchronizer:
    """
    Asynchronous data synchronizer - use when data arrives from separate processes.
    
    Waits until both GHQ and Emotion data are received, then triggers classification.
    """
    
    def __init__(self, timeout: float = 40.0, session_id: str = None):
        self.timeout = timeout
        self.session_id = session_id or f"session_{int(time.time())}"
        self.ghq_data: Optional[Dict[str, Any]] = None
        self.emotion_data: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
        self.result_queue: queue.Queue = queue.Queue()
        self.start_time = time.time()
        
        print(f"🔄 Mediator synchronizer initialized (session: {self.session_id}, timeout: {timeout}s)")

    def submit_ghq(self, data: Dict[str, Any]) -> bool:
        """Submit GHQ-12 assessment data."""
        with self.lock:
            self.ghq_data = data
            print(f"📊 GHQ data received for session {self.session_id}")
        self._check_ready()
        return True

    def submit_emotion(self, data: Dict[str, Any]) -> bool:
        """Submit emotion analysis data."""
        with self.lock:
            self.emotion_data = data
            print(f"🎭 Emotion data received for session {self.session_id}")
        self._check_ready()
        return True

    def _check_ready(self):
        """Check if both inputs are present and signal completion."""
        with self.lock:
            if self.ghq_data and self.emotion_data:
                elapsed = time.time() - self.start_time
                print(f"✅ Both datasets ready for session {self.session_id} (after {elapsed:.1f}s)")
                self.result_queue.put("ready")

    def wait_and_classify(self) -> Optional[Dict[str, Any]]:
        """
        Block until both inputs arrive or timeout, then perform classification.
        
        Returns:
            Classification result or None if timeout
        """
        print(f"⏳ Waiting for both datasets (timeout: {self.timeout}s)...")
        
        try:
            # Wait for both inputs
            self.result_queue.get(timeout=self.timeout)
            
            # Perform classification
            print(f"🧠 Starting classification for session {self.session_id}")
            result = _perform_classification(self.ghq_data, self.emotion_data)
            
            # Add session metadata
            result["session_id"] = self.session_id
            result["processing_time"] = time.time() - self.start_time
            
            print(f"✅ Classification completed for session {self.session_id}")
            return result
            
        except queue.Empty:
            elapsed = time.time() - self.start_time
            print(f"⏳ Timeout: did not receive both inputs within {self.timeout}s (elapsed: {elapsed:.1f}s)")
            print(f"   GHQ data: {'✅' if self.ghq_data else '❌'}")
            print(f"   Emotion data: {'✅' if self.emotion_data else '❌'}")
            
            # Return partial result if we have some data
            if self.ghq_data or self.emotion_data:
                return {
                    "categories": [],
                    "reason": "Incomplete data - timeout while waiting for second input",
                    "risk_level": "Unknown",
                    "session_id": self.session_id,
                    "timeout": True,
                    "partial_data": {
                        "has_ghq": bool(self.ghq_data),
                        "has_emotion": bool(self.emotion_data)
                    }
                }
            return None
        
        except Exception as e:
            print(f"❌ Classification error for session {self.session_id}: {e}")
            return {
                "categories": [],
                "reason": f"Classification failed: {str(e)}",
                "risk_level": "Unknown",
                "session_id": self.session_id,
                "error": str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current synchronizer status."""
        with self.lock:
            return {
                "session_id": self.session_id,
                "has_ghq": bool(self.ghq_data),
                "has_emotion": bool(self.emotion_data),
                "elapsed_time": time.time() - self.start_time,
                "timeout_remaining": max(0, self.timeout - (time.time() - self.start_time))
            }

# ---------------------------------------------------------------------
# INTEGRATION PATTERN 3: Batch Processing
# ---------------------------------------------------------------------
def classify_batch(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple classification requests in batch.
    
    Args:
        requests: List of {"ghq_summary": {...}, "emotion_summary": {...}, "request_id": "..."}
        
    Returns:
        List of classification results
    """
    print(f"🔄 Processing batch of {len(requests)} classification requests...")
    
    results = []
    for i, request in enumerate(requests):
        try:
            ghq_data = request.get("ghq_summary", {})
            emotion_data = request.get("emotion_summary", {})
            request_id = request.get("request_id", f"batch_item_{i}")
            
            if not ghq_data or not emotion_data:
                results.append({
                    "request_id": request_id,
                    "categories": [],
                    "reason": "Incomplete input data",
                    "risk_level": "Unknown",
                    "error": "Missing GHQ or emotion data"
                })
                continue
            
            result = _perform_classification(ghq_data, emotion_data)
            result["request_id"] = request_id
            results.append(result)
            
        except Exception as e:
            results.append({
                "request_id": request.get("request_id", f"batch_item_{i}"),
                "categories": [],
                "reason": f"Classification failed: {str(e)}",
                "risk_level": "Unknown",
                "error": str(e)
            })
    
    print(f"✅ Batch processing completed: {len(results)} results")
    return results

# ---------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------
def create_synchronizer(timeout: float = 40.0, session_id: str = None) -> MediatorSynchronizer:
    """Create a new synchronizer instance."""
    return MediatorSynchronizer(timeout=timeout, session_id=session_id)

def quick_classify_and_consult(ghq_score: int, ghq_notes: str, dominant_emotion: str, emotion_patterns: str):
    ghq_summary = {
        "total_score": ghq_score,
        "risk_level": "elevated" if ghq_score >= 3 else "mild distress" if ghq_score >= 1 else "normal",
        "notes": ghq_notes
    }
    emotion_summary = {
        "dominant_emotion": dominant_emotion,
        "patterns": emotion_patterns
    }
    
    # 1️⃣ Generate Mediator output
    mediator_output = classify_user(ghq_summary, emotion_summary)
    
    # 2️⃣ Launch Dr. Sarah directly
    dr_sarah = DrSarahAgent(mediator_output, user_name="Jackson")
    dr_sarah.start_consultation()


# ---------------------------------------------------------------------
# Example usage and testing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Mediator Agent - Integration Pattern Examples")
    print("=" * 60)
    
    # Example 1: Direct synchronous call
    print("\n1️⃣ DIRECT SYNCHRONOUS CLASSIFICATION:")
    ghq_example = {
        "total_score": 5,
        "risk_level": "elevated",
        "notes": "Patient showing multiple stress indicators"
    }
    emotion_example = {
        "dominant_emotion": "sad",
        "patterns": "High negative emotion prevalence; emotional instability detected"
    }
    
    result1 = classify_user(ghq_example, emotion_example)
    print("Result:", json.dumps(result1, indent=2))
    
    # Example 2: Asynchronous synchronizer
    print("\n2️⃣ ASYNCHRONOUS SYNCHRONIZER PATTERN:")
    sync = create_synchronizer(timeout=10.0, session_id="demo_session")
    
    # Simulate GHQ arriving first
    threading.Thread(target=lambda: sync.submit_ghq(ghq_example)).start()
    
    # Simulate emotion arriving later
    def late_emotion():
        time.sleep(2)
        sync.submit_emotion(emotion_example)
    
    threading.Thread(target=late_emotion).start()
    
    result2 = sync.wait_and_classify()
    if result2:
        print("Result:", json.dumps(result2, indent=2))
    else:
        print("❌ Classification failed (timeout)")
    
    # Example 3: Quick classification
    print("\n3️⃣ QUICK CLASSIFICATION:")
    result3 = quick_classify_and_consult(
        ghq_score=4,
        ghq_notes="Moderate stress levels detected",
        dominant_emotion="anxious",
        emotion_patterns="High anxiety prevalence with stress indicators"
    )
    print("Result:", json.dumps(result3, indent=2))
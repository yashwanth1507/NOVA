"""
Full end-to-end demo test for Nova desktop.
Tests: upload PDF -> grounded chat -> generate quiz -> submit answers -> verify tracking

This is a live integration test — it requires the server to be running on port 8765
with the local engine configured and a model available.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import urllib.request
import urllib.error

API_BASE = "http://127.0.0.1:8765/api"
PASS = 0
FAIL = 0

def report(name, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

def api(method, path, body=None, timeout=120):
    url = API_BASE + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        text = resp.read().decode("utf-8", errors="replace")
        if not text.strip():
            return {}
        return json.loads(text)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def make_test_pdf():
    """Create a small valid PDF with educational content."""
    try:
        from reportlab.pdfgen import canvas
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        c = canvas.Canvas(tmp.name)
        c.drawString(100, 750, "Quantum Computing Basics")
        c.drawString(100, 720, "Quantum computing uses qubits instead of classical bits.")
        c.drawString(100, 690, "Qubits can exist in superposition states.")
        c.drawString(100, 660, "This allows quantum computers to process certain problems")
        c.drawString(100, 630, "exponentially faster than classical computers.")
        c.drawString(100, 590, "Key concepts:")
        c.drawString(100, 570, "- Superposition: qubits can be 0 and 1 simultaneously")
        c.drawString(100, 550, "- Entanglement: qubits can be correlated across distance")
        c.drawString(100, 530, "- Interference: quantum states can amplify correct answers")
        c.save()
        return tmp.name
    except ImportError:
        # Fallback: create a minimal valid PDF manually
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 750 Td (Quantum Computing Basics) Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \n0000000357 00000 n \ntrailer<</Size 6/Root 1 0 R>>\n%%EOF")
        tmp.close()
        return tmp.name


def upload_pdf():
    """Upload a PDF and return document info."""
    pdf_path = make_test_pdf()
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
        boundary = b"----WebKitFormBoundary" + str(uuid.uuid4())[:8].encode()
        body = b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="quantum_basics.pdf"\r\nContent-Type: application/pdf\r\n\r\n' + data + b"\r\n--" + boundary + b"--\r\n"
        req = urllib.request.Request(API_BASE + "/documents/upload", data=body,
                                     headers={"Content-Type": b"multipart/form-data; boundary=" + boundary})
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read().decode())
    finally:
        try:
            os.unlink(pdf_path)
        except Exception:
            pass


def run_tests():
    global PASS, FAIL

    # ------------------------------------------------------------------ #
    print("\n═══ Nova Desktop Demo Test ═══\n")

    # 1. Server health
    print("── Server Health ──")
    try:
        h = api("GET", "/health", timeout=10)
        report("Server health", h.get("status") == "ok", f"status={h.get('status')}")
        report("Engine configured", "engine" in h, f"engine={h.get('engine')}")
        report("Active model set", "active_model_id" in h, f"model={h.get('active_model_id')}")
    except Exception as e:
        report("Server health", False, str(e))
        print("\n⚠ Server not running. Start with:\n  python main.py --server\n")
        return

    # 2. Upload PDF
    print("\n── Document Upload ──")
    try:
        doc = upload_pdf()
        report("PDF upload", "id" in doc and "chunks" in doc, f"id={doc.get('id','')[:12]}... chunks={len(doc.get('chunks',[]))}")
        doc_id = doc["id"]
        chunk_id = doc.get("chunks", [{}])[0].get("id", "")
    except Exception as e:
        report("PDF upload", False, str(e))
        doc_id = "<missing>"

    # 3. List documents
    print("\n── Document List ──")
    try:
        docs = api("GET", "/documents", timeout=10)
        report("Documents list", isinstance(docs, list), f"count={len(docs)}")
    except Exception as e:
        report("Documents list", False, str(e))

    # 4. New session
    print("\n── Session ──")
    try:
        session = api("POST", "/sessions", {"engine_used": "local"}, timeout=10)
        report("Session created", "id" in session, f"id={session.get('id','')[:12]}...")
        sid = session["id"]
    except Exception as e:
        report("Session created", False, str(e))
        sid = "<missing>"

    # 5. List sessions
    try:
        sessions = api("GET", "/sessions", timeout=10)
        report("Sessions list", isinstance(sessions, list) and len(sessions) > 0, f"count={len(sessions)}")
    except Exception as e:
        report("Sessions list", False, str(e))

    # 6. Chat (grounded, with document context)
    print("\n── Chat ──")
    try:
        chat = api("POST", "/chat", {
            "session_id": sid,
            "query": "What is quantum computing? Keep it very brief.",
            "document_id": doc_id,
        }, timeout=180)
        has_response = bool(chat.get("response", "").strip())
        report("Chat response", has_response, f"len={len(chat.get('response',''))} chars")
    except Exception as e:
        report("Chat response", False, str(e))

    # 7. Chat (ungrounded, general)
    try:
        chat2 = api("POST", "/chat", {
            "session_id": sid,
            "query": "Say hello briefly.",
        }, timeout=180)
        report("General chat", bool(chat2.get("response","").strip()), f"len={len(chat2.get('response',''))} chars")
    except Exception as e:
        report("General chat", False, str(e))

    # 8. Get messages
    print("\n── Messages ──")
    try:
        msgs = api("GET", f"/sessions/{sid}/messages", timeout=10)
        report("Messages retrieved", isinstance(msgs, list) and len(msgs) >= 2, f"count={len(msgs)}")
    except Exception as e:
        report("Messages retrieved", False, str(e))

    # 9. Generate quiz
    print("\n── Quiz ──")
    quiz_id = ""
    try:
        quiz = api("POST", "/quizzes/generate", {
            "session_id": sid,
            "chunk_id": chunk_id,
            "num_questions": 3,
        }, timeout=180)
        quiz_id = quiz.get("id", "")
        questions = quiz.get("questions", [])
        report("Quiz generated", bool(quiz_id) and len(questions) > 0, f"id={quiz_id[:12]}... questions={len(questions)}")
        if len(questions) > 0:
            q0 = questions[0]
            report("Q1 has question text", bool(q0.get("question", "").strip()))
            report("Q1 has 2+ options", len(q0.get("options", [])) >= 2)
            report("Q1 has correct_index", "correct_index" in q0)
    except Exception as e:
        report("Quiz generated", False, str(e))

    # 10. Answer quiz
    print("\n── Quiz Answers ──")
    try:
        q_data = api("POST", "/quizzes/generate", {
            "session_id": sid,
            "chunk_id": chunk_id,
            "num_questions": 1,
        }, timeout=180)
        qid = q_data.get("id", "")
        questions = q_data.get("questions", [])
        if qid and questions:
            answer = api("POST", "/quizzes/answer", {
                "quiz_id": qid,
                "question_index": 0,
                "student_answer": str(questions[0].get("correct_index", 0)),
                "correct_index": questions[0].get("correct_index", 0),
            }, timeout=10)
            report("Answer stored", answer.get("is_correct") is True, f"correct={answer.get('is_correct')}")
        else:
            raise RuntimeError(f"No questions returned (id={qid[:12] if qid else 'missing'})")
    except Exception as e:
        report("Answer stored", False, str(e))

    # 11. Quiz results
    print("\n── Results ──")
    try:
        results = api("GET", f"/sessions/{sid}/quiz-results", timeout=10)
        report("Quiz results", isinstance(results, list) and len(results) > 0, f"quizzes={len(results)}")
    except Exception as e:
        report("Quiz results", False, str(e))

    # 12. Available models endpoint
    print("\n── Model Selection API ──")
    try:
        models = api("GET", "/models/available", timeout=10)
        model_list = models.get("models", [])
        report("Models listed", len(model_list) == 2, f"count={len(model_list)}")
        has_1b = any(m["id"] == "gemma-3-1b" for m in model_list)
        has_270m = any(m["id"] == "gemma-3-270m" for m in model_list)
        report("Gemma 3 1B available", has_1b)
        report("Gemma 3 270M available", has_270m)
        report("Active model returned", "active_model_id" in models, f"active={models.get('active_model_id')}")
    except Exception as e:
        report("Model selection API", False, str(e))

    # 13. Switch model via API
    print("\n── Model Switching ──")
    try:
        # Switch to 270M (will fail download but API should accept the switch)
        result = api("PUT", "/settings/active_model", {"value": "gemma-3-270m"}, timeout=10)
        report("Switch to 270M accepted", True, "model_id updated")

        # Switch back to 1B
        result = api("PUT", "/settings/active_model", {"value": "gemma-3-1b"}, timeout=10)
        report("Switch back to 1B", True, "model_id restored")

        # Verify active model is correct
        h = api("GET", "/health", timeout=10)
        report("Active model verified", h.get("active_model_id") == "gemma-3-1b", f"model={h.get('active_model_id')}")
    except Exception as e:
        report("Model switching", False, str(e))

    # ------------------------------------------------------------------ #
    print(f"\n═══ Summary: {PASS}/{PASS+FAIL} passed ═══\n")


if __name__ == "__main__":
    run_tests()
    sys.exit(0 if FAIL == 0 else 1)

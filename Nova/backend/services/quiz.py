"""QuizService — generates quizzes from document chunks and tracks student answers.

Quizzes are generated content distinct from conversational turns.
Results are stored keyed to chunk and question so progress is visible across sessions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..engines.manager import EngineManager
from ..prompting.builder import PromptBuilder
from ..storage.database import Storage

logger = logging.getLogger(__name__)


class QuizService:
    """Generates quizzes from document content and tracks student answers."""

    def __init__(
        self,
        storage: Storage,
        engine_manager: EngineManager,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.storage = storage
        self.engine_manager = engine_manager
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate_quiz(
        self,
        session_id: str,
        chunk_id: str,
        num_questions: int = 4,
    ) -> Optional[dict]:
        """Generate a quiz from a document chunk.

        Returns the quiz dict with id, questions list, or None on failure.
        Has one retry with "valid JSON only" reminder on parse failure.
        """
        chunk = self.storage.get_chunk(chunk_id)
        if not chunk:
            logger.warning("Chunk %s not found", chunk_id)
            return None

        engine = self.engine_manager.get_engine()
        context_window = getattr(self.engine_manager.settings, "context_size", 2048)
        messages = self.prompt_builder.build_quiz_messages(
            content=chunk["content"],
            num_questions=num_questions,
            context_window=context_window,
        )
        # The prompt already reserves this much room, so the engine no longer
        # has to clamp the JSON output down to a truncated fragment.
        output_tokens = self.prompt_builder.quiz_output_tokens(
            num_questions, context_window
        )

        # Attempt generation with one retry and the full reserved output budget
        for attempt in range(2):
            try:
                if attempt == 1:
                    retry_messages = messages + [
                        {"role": "user", "content": "IMPORTANT: Output ONLY the JSON object with the 'questions' array. No introductory or trailing text."}
                    ]
                    raw = engine.generate(messages=retry_messages, max_tokens=output_tokens)
                else:
                    raw = engine.generate(messages=messages, max_tokens=output_tokens)

                questions = self._parse_quiz_json(raw)
                if questions:
                    quiz_id = self.storage.create_quiz(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        questions=questions,
                    )
                    quiz = self.storage.get_quiz(quiz_id)
                    logger.info(
                        "Quiz %s generated (%d questions from chunk %s)",
                        quiz_id,
                        len(questions),
                        chunk_id,
                    )
                    return quiz
            except Exception as e:
                logger.warning("Quiz generation attempt %d failed: %s", attempt + 1, e)

        logger.error("Quiz generation failed after 2 attempts for chunk %s", chunk_id)
        return None

    def answer_question(
        self,
        quiz_id: str,
        question_index: int,
        student_answer: str,
        correct_index: Optional[int] = None,
    ) -> dict:
        """Record a student's answer to a quiz question.

        ``correct_index`` is accepted for backwards compatibility and ignored:
        the stored quiz is the only authoritative source for grading.
        """
        quiz = self.storage.get_quiz(quiz_id)
        if not quiz:
            raise ValueError("Quiz not found")
        if question_index < 0 or question_index >= len(quiz["questions"]):
            raise ValueError("Question index is out of range")
        if any(a["question_index"] == question_index for a in self.storage.get_answers_for_quiz(quiz_id)):
            raise ValueError("This question has already been answered")

        question = quiz["questions"][question_index]
        options = question.get("options") or []
        actual_correct_index = _infer_correct_index(
            question.get("correct_index", question.get("answer")), options
        )
        if actual_correct_index is None:
            raise ValueError("This quiz question has no recorded correct answer")
        try:
            selected_index = int(str(student_answer).strip())
        except (TypeError, ValueError):
            raise ValueError("Answer must identify one of the quiz options") from None
        if not 0 <= selected_index < len(options):
            raise ValueError("Answer must identify one of the quiz options")
        is_correct = selected_index == actual_correct_index
        answer_id = self.storage.add_quiz_answer(
            quiz_id=quiz_id,
            question_index=question_index,
            is_correct=is_correct,
            student_answer=str(selected_index),
        )

        correct_answer = options[actual_correct_index]

        return {
            "id": answer_id,
            "quiz_id": quiz_id,
            "question_index": question_index,
            "is_correct": is_correct,
            "student_answer": str(selected_index),
            "correct_answer": correct_answer,
        }

    def get_quiz_results(self, session_id: str) -> list[dict]:
        """Get all quiz results for a session with answer details."""
        quizzes = self.storage.get_quizzes_for_session(session_id)
        results = []
        for quiz in quizzes:
            answers = self.storage.get_answers_for_quiz(quiz["id"])
            correct_count = sum(1 for a in answers if a["is_correct"])
            results.append({
                "quiz_id": quiz["id"],
                "chunk_id": quiz["chunk_id"],
                "questions": quiz["questions"],
                "answers": answers,
                "score": f"{correct_count}/{len(quiz['questions'])}",
                "created_at": quiz["created_at"],
            })
        return results

    @staticmethod
    def _parse_quiz_json(raw: str) -> Optional[list[dict]]:
        """Parse JSON from model output, handling common wrapping and quality issues."""
        raw = raw.strip()

        # 1. Extract from code fence if present
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if match:
                raw = match.group(1).strip()
            else:
                raw = re.sub(r"^```(?:json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()

        # 2. Try direct json.loads
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                q = _normalize_questions(data)
                if q:
                    return q
            elif isinstance(data, dict):
                q = _normalize_questions(data.get("questions", []))
                if q:
                    return q
        except json.JSONDecodeError:
            pass

        # 3. Try finding outermost JSON object or array
        obj_start = raw.find("{")
        obj_end = raw.rfind("}") + 1
        arr_start = raw.find("[")
        arr_end = raw.rfind("]") + 1

        if arr_start >= 0 and (obj_start < 0 or arr_start < obj_start) and arr_end > arr_start:
            try:
                data = json.loads(raw[arr_start:arr_end])
                if isinstance(data, list):
                    q = _normalize_questions(data)
                    if q:
                        return q
            except json.JSONDecodeError:
                pass

        if obj_start >= 0 and obj_end > obj_start:
            try:
                data = json.loads(raw[obj_start:obj_end])
                if isinstance(data, dict):
                    q = _normalize_questions(data.get("questions", []))
                    if q:
                        return q
            except json.JSONDecodeError:
                pass

        # 4. Robust brace-depth extraction of individual question objects
        try:
            candidates = []
            idx = 0
            while idx < len(raw):
                q_pos = raw.find('"question"', idx)
                if q_pos == -1:
                    break
                brace_start = raw.rfind('{', 0, q_pos)
                if brace_start == -1:
                    idx = q_pos + 10
                    continue

                depth = 0
                brace_end = -1
                in_str = False
                escape = False
                for i in range(brace_start, len(raw)):
                    char = raw[i]
                    if char == '"' and not escape:
                        in_str = not in_str
                    elif not in_str:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                brace_end = i + 1
                                break
                    if char == '\\' and not escape:
                        escape = True
                    else:
                        escape = False

                if brace_end > brace_start:
                    chunk = raw[brace_start:brace_end]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict) and "question" in obj and "options" in obj:
                            candidates.append(obj)
                    except Exception:
                        pass
                    idx = brace_end
                else:
                    idx = q_pos + 10

            if candidates:
                q = _normalize_questions(candidates)
                if q:
                    return q
        except Exception:
            pass

        # 5. Fallback: Parse plain text / markdown quiz format
        text_parsed = _parse_text_quiz(raw)
        if text_parsed:
            logger.info("Successfully extracted %d questions from plain-text quiz output", len(text_parsed))
            return text_parsed

        logger.warning("Failed to parse quiz JSON or markdown from output")
        return None


def _parse_text_quiz(raw: str) -> Optional[list[dict]]:
    """Fallback parser for markdown or plain-text multiple-choice questions."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    questions = []

    current_q = None
    current_opts = []
    current_ans = None
    current_exp = ""

    # Question line patterns: "1. ...", "Question 1: ...", "Q1: ...", "1) ..."
    q_re = re.compile(r"^(?:(?:Question\s*)?\d+[\.:\)]|\bQ\d+[\.:\)])\s*(.+)$", re.IGNORECASE)
    # Option line patterns: "A) ...", "A. ...", "(A) ...", "- A) ..."
    opt_re = re.compile(r"^(?:[\-\*]\s*)?(?:\(([A-Da-d0-9])\)|([A-Da-d0-9])[\.\)\:\-])\s*(.+)$")
    # Answer line patterns: "Answer: A", "Correct Answer: B", "Ans: C"
    ans_re = re.compile(r"^(?:Correct\s*)?Ans(?:wer)?\s*[\:\-]\s*([A-Da-d0-9]|\bOption\s*[A-Da-d0-9]\b|.+)", re.IGNORECASE)
    # Explanation patterns: "Explanation: ..."
    exp_re = re.compile(r"^Explanation\s*[\:\-]\s*(.+)$", re.IGNORECASE)

    def flush():
        nonlocal current_q, current_opts, current_ans, current_exp
        if current_q and len(current_opts) >= 2:
            questions.append({
                "question": current_q,
                "options": current_opts,
                "answer": current_ans if current_ans is not None else 0,
                "explanation": current_exp,
            })
        current_q = None
        current_opts = []
        current_ans = None
        current_exp = ""

    for line in lines:
        q_match = q_re.match(line)
        if q_match:
            if current_q and len(current_opts) >= 2:
                flush()
            current_q = q_match.group(1).strip()
            continue

        opt_match = opt_re.match(line)
        if opt_match and current_q:
            opt_text = opt_match.group(3).strip()
            current_opts.append(opt_text)
            continue

        ans_match = ans_re.match(line)
        if ans_match and current_q:
            current_ans = ans_match.group(1).strip()
            continue

        exp_match = exp_re.match(line)
        if exp_match and current_q:
            current_exp = exp_match.group(1).strip()
            continue

        if current_q and not current_opts:
            current_q += " " + line

    if current_q and len(current_opts) >= 2:
        flush()

    if questions:
        return _normalize_questions(questions)
    return None



def _clean_option_text(text: str) -> str:
    """Strip leading option labels like 'A) ', 'A. ', '1. ', '(a) ' so options are clean text."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    return re.sub(r"^(\(?[A-Da-d0-9]\)?[.\-:]\s*|\([A-Da-d0-9]\)\s*)", "", text).strip()


def _normalize_questions(questions: list[dict]) -> Optional[list[dict]]:
    """Validate and normalize quiz questions, ensuring 4 options and balanced answer distribution."""
    normalized = []
    for q_idx, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        if "question" not in q or "options" not in q:
            continue
        if not q["question"] or not q.get("options"):
            continue

        raw_options = list(q["options"])
        if len(raw_options) < 2:
            continue

        # Infer correct_index before cleaning
        raw_correct = q.get("correct_index", q.get("answer", None))
        correct = _infer_correct_index(raw_correct, raw_options)
        if correct is None:
            correct = 0

        # Clean option texts
        options = [_clean_option_text(opt) for opt in raw_options]

        # Pad to 4 options if fewer
        defaults = ["None of the above", "All of the above", "Cannot be determined", "Other"]
        while len(options) < 4:
            for d in defaults:
                if d not in options:
                    options.append(d)
                    break
            else:
                options.append(f"Option {len(options) + 1}")
        options = options[:4]

        # Ensure correct is in range 0..3
        correct = max(0, min(len(options) - 1, correct))

        # Balance answer distribution across options:
        # Avoid the bias where option 0 is always the answer by cycling the target position
        target_idx = q_idx % len(options)
        if correct != target_idx:
            options[correct], options[target_idx] = options[target_idx], options[correct]
            correct = target_idx

        normalized.append({
            "question": q["question"].strip(),
            "options": options,
            "correct_index": correct,
            "explanation": q.get("explanation", "").strip() if isinstance(q.get("explanation"), str) else "",
            "concept": q.get("concept", "").strip() if isinstance(q.get("concept"), str) else "",
        })

    return normalized if normalized else None


def _infer_correct_index(raw, options) -> Optional[int]:
    """Try to determine the correct answer index from various formats."""
    if raw is None:
        return None

    if isinstance(raw, int):
        return max(0, min(len(options) - 1, raw)) if 0 <= raw < len(options) else None

    if isinstance(raw, str):
        raw = raw.strip()

        if raw in ("0", "1", "2", "3"):
            return int(raw)

        letter_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        letter = raw.strip().rstrip(")").lower()
        if letter in letter_map:
            return letter_map[letter]

        if options:
            for i, opt in enumerate(options):
                opt_clean = _clean_option_text(opt)
                if raw.lower() == opt_clean.lower() or raw.lower() in opt_clean.lower() or opt_clean.lower() in raw.lower():
                    return i

    return None


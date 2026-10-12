"""Tests for the local engine: warm-up, thread safety and token budgeting.

A stub model stands in for llama-cpp-python so the suite stays fast and does
not require the GGUF weights.
"""

import threading
import time


from Nova.backend.engines.local import LocalEngine


class StubLlama:
    """Minimal stand-in for llama_cpp.Llama."""

    def __init__(self, prompt_tokens: int = 12, reply: str = "answer"):
        self.prompt_tokens = prompt_tokens
        self.reply = reply
        self.completions = []
        self.tokenize_calls = 0

    def tokenize(self, text, add_bos=True, special=False):
        self.tokenize_calls += 1
        return list(range(self.prompt_tokens))

    def create_completion(self, prompt, temperature=0.0, max_tokens=0, stop=(), stream=False):
        self.completions.append({"prompt": prompt, "max_tokens": max_tokens})
        if stream:
            return iter([{"choices": [{"text": self.reply}]}])
        return {"choices": [{"text": self.reply}]}


def _engine_with_stub(prompt_tokens: int = 12, context_size: int = 64, reply: str = "answer"):
    engine = LocalEngine(model_path="models/unused.gguf", context_size=context_size)
    engine._model = StubLlama(prompt_tokens=prompt_tokens, reply=reply)
    return engine


def test_generation_token_budget_is_clamped_to_the_context_window():
    engine = _engine_with_stub(prompt_tokens=50, context_size=64)
    engine.generate([{"role": "user", "content": "hi"}], max_tokens=1000)
    sent = engine._model.completions[-1]
    assert sent["max_tokens"] == 14  # 64 - 50 prompt tokens
    assert engine._model.tokenize_calls == 1, "the prompt must be tokenized only once"


def test_prompt_is_passed_as_tokens_to_avoid_double_tokenization():
    engine = _engine_with_stub(prompt_tokens=8)
    engine.generate([{"role": "user", "content": "hello"}])
    sent = engine._model.completions[-1]
    assert isinstance(sent["prompt"], list)
    assert len(sent["prompt"]) == 8


def test_streaming_uses_the_same_budget_and_yields_tokens():
    engine = _engine_with_stub(prompt_tokens=40, context_size=64, reply="streamed")
    tokens = list(engine.generate_stream([{"role": "user", "content": "hi"}], max_tokens=500))
    assert tokens == ["streamed"]
    assert engine._model.completions[-1]["max_tokens"] == 24


def test_warm_up_loads_the_model_and_reports_readiness():
    engine = LocalEngine(model_path="models/unused.gguf")
    loads = []

    def fake_load():
        loads.append(1)
        time.sleep(0.05)
        engine._model = StubLlama()

    engine._load_model = fake_load
    assert engine.is_loaded is False
    assert engine.warm_up() is True
    assert engine.is_loaded is True
    assert len(loads) == 1
    # A second warm-up must not reload the weights.
    assert engine.warm_up() is True
    assert len(loads) == 1


def test_concurrent_first_requests_load_the_model_once():
    engine = LocalEngine(model_path="models/unused.gguf")
    loads = []

    def fake_load():
        loads.append(1)
        time.sleep(0.05)
        engine._model = StubLlama()

    engine._load_model = fake_load
    threads = [threading.Thread(target=engine._lazy_load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(loads) == 1, "parallel first requests must not each build a model"


def test_engine_manager_warm_up_skips_online_engine():
    from Nova.backend.config import Settings
    from Nova.backend.engines.manager import EngineManager

    manager = EngineManager(settings=Settings(engine="online"))
    assert manager.warm_up_local_model() is False
    assert manager.local_model_loaded is False

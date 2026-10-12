from Nova.backend.prompting.builder import PromptBuilder


def test_chat_prompt_is_compacted_for_2048_context():
    builder = PromptBuilder()
    messages = builder.build_chat_messages(
        query="Explain the important lessons and likely exam topics. " * 40,
        context_chunks=["Long document section about geography. " * 1000] * 5,
        history=[
            {"role": "user", "content": "previous question " * 500},
            {"role": "assistant", "content": "previous answer " * 500},
        ],
    )

    # 512 tokens remain available for the model response.
    assert builder.estimate_message_tokens(messages) <= 1536
    assert messages[-1]["role"] == "user"


def test_chat_prompt_ends_with_the_question_and_carries_the_evidence():
    builder = PromptBuilder()
    messages = builder.build_chat_messages(
        query="What is photosynthesis?",
        context_chunks=["Plants convert sunlight into chemical energy."],
        history=[],
    )
    # system + one user turn; no fake assistant acknowledgement that small
    # models tend to answer instead of the real question.
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Plants convert sunlight into chemical energy." in messages[-1]["content"]
    assert messages[-1]["content"].rstrip().endswith("What is photosynthesis?")


def test_quiz_prompt_always_reserves_room_for_the_json_output():
    builder = PromptBuilder()
    for num_questions in (1, 3, 5, 10):
        for window in (1024, 2048, 4096, 8192):
            messages = builder.build_quiz_messages(
                content="Study material sentence. " * 400,
                num_questions=num_questions,
                context_window=window,
            )
            used = (
                builder.estimate_message_tokens(messages)
                + builder.quiz_output_tokens(num_questions, window)
            )
            assert used <= window, (num_questions, window, used)


def test_quiz_output_budget_scales_with_question_count():
    builder = PromptBuilder()
    one = builder.quiz_output_tokens(1, 2048)
    five = builder.quiz_output_tokens(5, 2048)
    assert one < five <= 2048 // 2


def test_summary_prompt_reserves_response_room():
    builder = PromptBuilder()
    for window in (1024, 2048, 4096):
        messages = builder.build_summary_messages(
            content="Long chapter text. " * 800,
            context_window=window,
            max_response_tokens=600,
        )
        response_budget = max(128, min(600, window // 2))
        assert builder.estimate_message_tokens(messages) + response_budget <= window
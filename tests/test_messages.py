from ctx_squeeze.messages import parse_messages, prune_messages, to_dicts


def test_parse_messages_reads_openai_shape():
    history = [
        {"role": "system", "content": "You are a careful build engineer."},
        {"role": "user", "content": "Can you patch the workflow file?"},
    ]
    messages = parse_messages(history)
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "Can you patch the workflow file?"
    assert messages[1].tokens > 0


def test_parse_messages_reads_anthropic_content_blocks():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "hello there"}]},
    ]
    messages = parse_messages(history)
    assert messages[0].tokens > 0


def test_parse_messages_links_openai_tool_call_to_its_result():
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    ]
    messages = parse_messages(history)
    assert messages[0].calls == ["call_1"]
    assert messages[1].answers == ["call_1"]


def test_parse_messages_links_anthropic_tool_use_to_its_result():
    history = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    ]
    messages = parse_messages(history)
    assert messages[0].calls == ["toolu_1"]
    assert messages[1].answers == ["toolu_1"]


def test_to_dicts_round_trips_role_and_content():
    history = [{"role": "user", "content": "hello"}]
    messages = parse_messages(history)
    assert to_dicts(messages) == history


def test_to_dicts_preserves_extra_fields():
    history = [{"role": "tool", "tool_call_id": "call_1", "content": "result"}]
    messages = parse_messages(history)
    assert to_dicts(messages) == history


def test_prune_messages_keeps_system_messages_regardless_of_budget():
    history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=0, recent_turns=1)
    assert [m.role for m in result.messages] == ["system", "user"]


def test_prune_messages_keeps_recent_turns_whole_even_over_budget():
    history = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "second reply"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=0, recent_turns=1, marker=False)
    assert [m.content for m in result.messages] == ["second turn", "second reply"]


def test_prune_messages_keeps_everything_when_budget_covers_total():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "second reply"},
    ]
    messages = parse_messages(history)
    total = sum(m.tokens for m in messages)
    result = prune_messages(messages, budget=total, recent_turns=1)
    assert [m.content for m in result.messages] == [m.content for m in messages]
    assert result.pinned_tool_results == []


def test_prune_messages_inserts_elision_marker_with_dropped_count():
    history = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "recent turn"},
        {"role": "assistant", "content": "recent reply"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=0, recent_turns=1)
    marker = result.messages[0]
    assert marker.role == "system"
    assert marker.content == "[2 earlier messages elided]"


def test_prune_messages_omits_marker_when_nothing_is_dropped():
    history = [{"role": "user", "content": "only turn"}]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=0, recent_turns=1)
    assert [m.content for m in result.messages] == ["only turn"]


def test_prune_messages_pins_a_call_whose_turn_was_dropped():
    history = [
        {"role": "user", "content": "start task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "run", "arguments": "{}"}}],
        },
        {"role": "user", "content": "any update?"},
        {"role": "tool", "tool_call_id": "call_1", "content": "result of call_1"},
        {"role": "assistant", "content": "here is the result"},
    ]
    messages = parse_messages(history)
    result = prune_messages(messages, budget=0, recent_turns=1)
    assert result.pinned_tool_results == ["call_1"]
    contents = [m.content for m in result.messages]
    assert contents.index(None) < contents.index("result of call_1")
    assert "start task" not in contents
    assert "any update?" in contents

"""Parse and prune chat transcripts message by message.

Segment-level compaction throws away structure a transcript actually has:
which messages are system prompts, which turns are recent, and which tool
call a given tool result belongs to. Pruning at message granularity keeps
that structure instead of flattening everything to text first.

A "turn" here is a user message plus every message that follows it up to
(but not including) the next user message - so an assistant tool call and
the tool result answering it always land in the same turn, since only a
user message starts a new one. That is what keeps a call and its result
together without any extra bookkeeping, except for the rarer case where a
user message is interleaved between a call and its answer; that case is
handled explicitly by pinning.
"""

from .tokens import estimate_tokens

__all__ = ["Message", "PruneResult", "parse_messages", "prune_messages", "to_dicts"]


def _flatten_content(content):
    """Return a plain-text rendering of a message's ``content`` field.

    ``content`` is either a string (OpenAI shape) or a list of content
    blocks (Anthropic shape: ``{"type": "text", "text": ...}``,
    ``{"type": "tool_use", ...}``, ``{"type": "tool_result", ...}``).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "tool_result":
                inner = block.get("content")
                parts.append(_flatten_content(inner))
            elif block_type == "tool_use":
                parts.append(str(block.get("input", "")))
            elif "text" in block:
                parts.append(block["text"])
        return "\n".join(part for part in parts if part)
    return str(content)


def _flatten_tool_calls(tool_calls):
    """Return a plain-text rendering of an OpenAI-shape ``tool_calls`` list."""
    if not tool_calls:
        return ""
    parts = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        parts.append(function.get("name", ""))
        arguments = function.get("arguments", "")
        parts.append(arguments if isinstance(arguments, str) else str(arguments))
    return "\n".join(part for part in parts if part)


def _tool_use_ids(content):
    """Return the ids of ``tool_use`` blocks a message's content issues."""
    if not isinstance(content, list):
        return []
    return [
        block["id"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    ]


def _tool_result_ids(content):
    """Return the ``tool_use`` ids a message's content answers."""
    if not isinstance(content, list):
        return []
    return [
        block["tool_use_id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id")
    ]


class Message(object):
    """One entry in a chat transcript, plus the bits pruning needs.

    ``calls`` holds the ids this message issues (an OpenAI ``tool_calls``
    entry or an Anthropic ``tool_use`` block); ``answers`` holds the ids
    this message answers (``tool_call_id`` or a ``tool_result`` block).
    ``extra`` keeps every other field from the original dict (``name``,
    ``tool_calls``, ``tool_call_id``, ...) so :meth:`to_dict` can round-trip
    a message that was not otherwise touched.
    """

    __slots__ = ("role", "content", "calls", "answers", "extra", "tokens")

    def __init__(self, role, content, calls=None, answers=None, extra=None):
        self.role = role
        self.content = content
        self.calls = list(calls) if calls else []
        self.answers = list(answers) if answers else []
        self.extra = dict(extra) if extra else {}
        text = _flatten_content(content)
        tool_call_text = _flatten_tool_calls(self.extra.get("tool_calls"))
        if tool_call_text:
            text = text + "\n" + tool_call_text if text else tool_call_text
        self.tokens = estimate_tokens(text)

    def to_dict(self):
        """Return this message as a plain dict, original fields intact."""
        message = dict(self.extra)
        message["role"] = self.role
        message["content"] = self.content
        return message

    def __repr__(self):
        return "Message(role=%r, tokens=%d)" % (self.role, self.tokens)


class PruneResult(object):
    """The output of :func:`prune_messages`."""

    __slots__ = ("messages", "pinned_tool_results")

    def __init__(self, messages, pinned_tool_results):
        self.messages = messages
        self.pinned_tool_results = pinned_tool_results


def parse_messages(history):
    """Parse a list of raw message dicts into :class:`Message` objects.

    Accepts the OpenAI shape (``content`` a string, tool calls as a
    top-level ``tool_calls`` list, tool results carrying a top-level
    ``tool_call_id``) and the Anthropic shape (``content`` a list of typed
    blocks) in the same list, message by message.
    """
    messages = []
    for entry in history:
        role = entry.get("role", "user")
        content = entry.get("content")
        calls = _tool_use_ids(content)
        answers = _tool_result_ids(content)
        tool_calls = entry.get("tool_calls") or []
        calls.extend(
            call["id"] for call in tool_calls if isinstance(call, dict) and call.get("id")
        )
        tool_call_id = entry.get("tool_call_id")
        if tool_call_id:
            answers.append(tool_call_id)
        extra = {k: v for k, v in entry.items() if k not in ("role", "content")}
        messages.append(Message(role, content, calls=calls, answers=answers, extra=extra))
    return messages


def to_dicts(messages):
    """Convert a list of :class:`Message` back into plain dicts for JSON output."""
    return [message.to_dict() for message in messages]


def _group_turns(messages):
    """Group non-system messages into turns, oldest first.

    A turn starts at a user message and runs up to (not including) the next
    one. Any messages before the first user message form a leading turn of
    their own, so nothing in ``messages`` is ever dropped by the grouping.
    """
    turns = []
    current = []
    for message in messages:
        if message.role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def prune_messages(messages, budget, recent_turns=2, marker=True):
    """Prune a chat transcript, keeping its structure intact.

    System messages always survive. The last ``recent_turns`` turns survive
    whole regardless of budget. Older turns are added back in, most recent
    first, for as long as they still fit in ``budget``; anything left over
    is replaced by a single system message noting how many messages were
    elided (unless ``marker`` is false).

    If a kept turn holds a tool result whose call lives in a dropped turn,
    that one call message is pinned back in right before it, and its id is
    reported in the result's ``pinned_tool_results`` - a tool result is
    never left answering a call that is not there.
    """
    system_messages = [m for m in messages if m.role == "system"]
    turns = _group_turns([m for m in messages if m.role != "system"])

    kept = [False] * len(turns)
    recent = min(recent_turns, len(turns))
    for i in range(len(turns) - recent, len(turns)):
        kept[i] = True

    used = sum(m.tokens for m in system_messages)
    used += sum(m.tokens for i, turn in enumerate(turns) if kept[i] for m in turn)

    for i in range(len(turns) - recent - 1, -1, -1):
        turn_tokens = sum(m.tokens for m in turns[i])
        if used + turn_tokens > budget:
            break
        kept[i] = True
        used += turn_tokens

    call_location = {}
    for i, turn in enumerate(turns):
        for message in turn:
            for call_id in message.calls:
                call_location[call_id] = (i, message)

    pinned_tool_results = []
    pinned_by_turn = {}
    for i, turn in enumerate(turns):
        if not kept[i]:
            continue
        for message in turn:
            for answer_id in message.answers:
                location = call_location.get(answer_id)
                if location is None:
                    continue
                call_turn, call_message = location
                if kept[call_turn]:
                    continue
                pinned_by_turn.setdefault(i, []).append(call_message)
                pinned_tool_results.append(answer_id)

    pinned_count = sum(len(pins) for pins in pinned_by_turn.values())
    dropped_count = sum(len(turn) for i, turn in enumerate(turns) if not kept[i]) - pinned_count

    result = list(system_messages)
    if marker and dropped_count > 0:
        result.append(Message("system", "[%d earlier messages elided]" % dropped_count))
    for i, turn in enumerate(turns):
        if not kept[i]:
            continue
        result.extend(pinned_by_turn.get(i, []))
        result.extend(turn)

    return PruneResult(messages=result, pinned_tool_results=pinned_tool_results)

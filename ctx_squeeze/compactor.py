"""Fit a document into a token budget by running it through a pipeline of stages.

A strategy is a comma-separated list of stage names, run left to right against
the same segment list: ``dedupe`` drops near-duplicates, ``score`` keeps the
most distinctive segments that still fit, ``head-tail`` keeps a prefix and a
suffix instead. Only ``score`` and ``head-tail`` look at the budget directly;
``dedupe`` just filters. Whatever comes out of the pipeline still has to fit,
so a final guard re-applies ``select_by_score`` if the pipeline left it over
budget - that is what keeps the kept segments themselves within budget no
matter which stages were named. The "[N segments elided]" markers layered on
top afterward are informational and not counted against the budget, the same
way ``prune_messages`` treats its own elision marker.
"""

from .dedupe import dedupe_segments
from .scoring import select_by_score
from .segments import split_segments
from .tokens import estimate_tokens

__all__ = ["SqueezeResult", "STRATEGIES", "squeeze"]


class SqueezeResult(object):
    """The output of :func:`squeeze`."""

    __slots__ = (
        "text",
        "original_tokens",
        "final_tokens",
        "segments_in",
        "segments_out",
        "notes",
    )

    def __init__(self, text, original_tokens, final_tokens, segments_in, segments_out, notes):
        self.text = text
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.segments_in = segments_in
        self.segments_out = segments_out
        self.notes = notes

    def __repr__(self):
        return "SqueezeResult(%d -> %d tokens, %d of %d segments)" % (
            self.original_tokens,
            self.final_tokens,
            self.segments_out,
            self.segments_in,
        )


def _stage_dedupe(segments, budget, options):
    kept, dropped = dedupe_segments(
        segments,
        shingle_size=options["shingle_size"],
        threshold=options["jaccard"],
    )
    note = None
    if dropped:
        note = "dedupe dropped %d near-duplicate segment(s)" % dropped
    return kept, note


def _stage_score(segments, budget, options):
    selected = select_by_score(segments, budget)
    dropped = len(segments) - len(selected)
    note = None
    if dropped:
        note = "score kept %d of %d segment(s) within budget" % (len(selected), len(segments))
    return selected, note


def _select_head_tail(segments, budget, head_ratio):
    if budget <= 0 or not segments:
        return []

    head_budget = int(round(budget * head_ratio))
    tail_budget = budget - head_budget

    head_end = 0
    used = 0
    for segment in segments:
        if used + segment.tokens > head_budget:
            break
        used += segment.tokens
        head_end += 1

    tail_start = len(segments)
    used = 0
    for i in range(len(segments) - 1, head_end - 1, -1):
        tokens = segments[i].tokens
        if used + tokens > tail_budget:
            break
        used += tokens
        tail_start = i

    return segments[:head_end] + segments[tail_start:]


def _stage_head_tail(segments, budget, options):
    selected = _select_head_tail(segments, budget, options["head_ratio"])
    dropped = len(segments) - len(selected)
    note = None
    if dropped:
        note = "head-tail kept %d of %d segment(s) within budget" % (len(selected), len(segments))
    return selected, note


STRATEGIES = {
    "dedupe": _stage_dedupe,
    "score": _stage_score,
    "head-tail": _stage_head_tail,
}


def _assemble(kept, segment_count, marker):
    """Re-join ``kept`` segments in document order, marking the gaps between them."""
    if not kept:
        if marker and segment_count > 0:
            return "[%d segments elided]" % segment_count
        return ""

    parts = []
    previous_index = -1
    for segment in kept:
        gap = segment.index - previous_index - 1
        if gap > 0 and marker:
            parts.append("[%d segments elided]" % gap)
        parts.append(segment.text)
        previous_index = segment.index

    trailing_gap = segment_count - 1 - previous_index
    if trailing_gap > 0 and marker:
        parts.append("[%d segments elided]" % trailing_gap)

    return "\n\n".join(parts)


def squeeze(text, budget, strategy="score", head_ratio=0.5, jaccard=0.8, shingle_size=5, marker=True):
    """Compact ``text`` so it fits inside ``budget`` estimated tokens.

    ``strategy`` is a comma-separated pipeline of stage names from
    :data:`STRATEGIES`, applied left to right. The kept segments always fit
    inside ``budget`` regardless of which stages are named: a stage that does
    not itself select by budget (``dedupe``) can still leave the document
    over budget, so a ``select_by_score`` pass runs afterward as a guard.
    """
    stage_names = [name.strip() for name in strategy.split(",") if name.strip()]
    if not stage_names:
        raise ValueError("strategy must name at least one stage")
    for name in stage_names:
        if name not in STRATEGIES:
            raise ValueError(
                "unknown strategy stage %r (choose from %s)"
                % (name, ", ".join(sorted(STRATEGIES)))
            )

    original_tokens = estimate_tokens(text)
    original_segments = split_segments(text)
    segment_count = len(original_segments)

    options = {"head_ratio": head_ratio, "jaccard": jaccard, "shingle_size": shingle_size}
    segments = original_segments
    notes = []
    for name in stage_names:
        segments, note = STRATEGIES[name](segments, budget, options)
        if note:
            notes.append(note)

    if sum(segment.tokens for segment in segments) > budget:
        guarded = select_by_score(segments, budget)
        if len(guarded) != len(segments):
            notes.append(
                "budget guard trimmed to %d of %d segment(s)" % (len(guarded), len(segments))
            )
        segments = guarded

    assembled = _assemble(segments, segment_count, marker)

    return SqueezeResult(
        text=assembled,
        original_tokens=original_tokens,
        final_tokens=estimate_tokens(assembled),
        segments_in=segment_count,
        segments_out=len(segments),
        notes=notes,
    )

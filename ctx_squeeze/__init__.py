"""Context compaction for LLM prompts, using only the standard library.

The CLI lives in ``cli.py`` and is reached through the ``ctx-squeeze``
console script (see ``pyproject.toml``) or ``python -m ctx_squeeze.cli``, so
it is not imported here - it only needs ``argparse`` and stdio, and plain
``import ctx_squeeze`` shouldn't have to pull those in.
"""

from .compactor import SqueezeResult, STRATEGIES, squeeze
from .dedupe import dedupe_segments, find_near_duplicates, jaccard, shingles
from .messages import Message, PruneResult, parse_messages, prune_messages, to_dicts
from .scoring import score_segments, select_by_score
from .segments import Segment, join_segments, split_segments
from .tokens import estimate_tokens, fits_budget, truncate_to_tokens

__version__ = "0.1.0"

__all__ = [
    "Message",
    "PruneResult",
    "Segment",
    "SqueezeResult",
    "STRATEGIES",
    "__version__",
    "dedupe_segments",
    "estimate_tokens",
    "find_near_duplicates",
    "fits_budget",
    "jaccard",
    "join_segments",
    "parse_messages",
    "prune_messages",
    "score_segments",
    "select_by_score",
    "shingles",
    "split_segments",
    "squeeze",
    "to_dicts",
    "truncate_to_tokens",
]

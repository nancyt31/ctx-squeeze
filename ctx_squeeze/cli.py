"""Command-line entry point for ctx-squeeze.

Kept separate from the rest of the package so ``import ctx_squeeze`` never
has to pull in ``argparse`` or touch stdio - only running the module or the
``ctx-squeeze`` console script (see ``pyproject.toml``) does that.
"""

import argparse
import json
import sys

from .compactor import STRATEGIES, squeeze
from .messages import parse_messages, prune_messages, to_dicts

__all__ = ["main"]


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_output(path, text):
    content = text
    if content and not content.endswith("\n"):
        content += "\n"
    if path is None:
        sys.stdout.write(content)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit a document or chat transcript into an LLM token budget.",
    )
    parser.add_argument("path", help="input file, or - to read standard input")
    parser.add_argument(
        "--budget", type=int, required=True, help="target size in estimated tokens"
    )
    parser.add_argument(
        "--strategy",
        default="score",
        help="comma-separated pipeline of: %s (default: score)" % ", ".join(sorted(STRATEGIES)),
    )
    parser.add_argument(
        "--head-ratio",
        type=float,
        default=0.5,
        help="share of the budget spent on the head in the head-tail stage (default: 0.5)",
    )
    parser.add_argument(
        "--jaccard",
        type=float,
        default=0.8,
        help="similarity at which two segments count as duplicates (default: 0.8)",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="words per shingle in the dedupe stage (default: 5)",
    )
    parser.add_argument(
        "--messages",
        action="store_true",
        help="treat the input as a JSON chat transcript instead of a document",
    )
    parser.add_argument(
        "--recent-turns",
        type=int,
        default=2,
        help="user turns kept whole in --messages mode (default: 2)",
    )
    parser.add_argument(
        "--no-marker",
        action="store_true",
        help="omit the elision markers left where content was dropped",
    )
    parser.add_argument(
        "--stats", action="store_true", help="print a token summary to stderr"
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="emit a JSON report instead of plain text"
    )
    parser.add_argument(
        "-o", dest="output", default=None, help="write the result to a file (default: stdout)"
    )
    return parser


def _run_document(args, raw):
    result = squeeze(
        raw,
        budget=args.budget,
        strategy=args.strategy,
        head_ratio=args.head_ratio,
        jaccard=args.jaccard,
        shingle_size=args.shingle_size,
        marker=not args.no_marker,
    )
    if args.stats:
        sys.stderr.write(
            "kept %d of %d segments | %d -> %d tokens (budget %d)\n"
            % (
                result.segments_out,
                result.segments_in,
                result.original_tokens,
                result.final_tokens,
                args.budget,
            )
        )
    if args.as_json:
        return json.dumps(
            {
                "text": result.text,
                "original_tokens": result.original_tokens,
                "final_tokens": result.final_tokens,
                "segments_in": result.segments_in,
                "segments_out": result.segments_out,
                "notes": result.notes,
            },
            indent=2,
        )
    return result.text


def _run_messages(args, raw):
    history = json.loads(raw)
    if not isinstance(history, list):
        raise ValueError("--messages expects a JSON array of message objects")

    parsed = parse_messages(history)
    original_tokens = sum(message.tokens for message in parsed)
    pruned = prune_messages(
        parsed,
        budget=args.budget,
        recent_turns=args.recent_turns,
        marker=not args.no_marker,
    )
    final_tokens = sum(message.tokens for message in pruned.messages)

    if args.stats:
        sys.stderr.write(
            "kept %d of %d messages | %d -> %d tokens (budget %d)\n"
            % (len(pruned.messages), len(parsed), original_tokens, final_tokens, args.budget)
        )

    dicts = to_dicts(pruned.messages)
    if args.as_json:
        return json.dumps(
            {
                "messages": dicts,
                "pinned_tool_results": pruned.pinned_tool_results,
                "original_tokens": original_tokens,
                "final_tokens": final_tokens,
            },
            indent=2,
        )
    return json.dumps(dicts, indent=2)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        raw = _read_input(args.path)
    except OSError as error:
        parser.error(str(error))

    try:
        if args.messages:
            output = _run_messages(args, raw)
        else:
            output = _run_document(args, raw)
    except ValueError as error:
        parser.error(str(error))

    _write_output(args.output, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

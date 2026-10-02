import argparse
import json
from typing import Any, Dict, Optional

from .engine import LocalDecisionEngine


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with user-friendly help text."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank a set of decision options against a user question using sentence "
            "embeddings."
        ),
        epilog=(
            "Example: jev --question \"The database is saturated\" "
            "--decision scale_up=Increase resources --decision drop_traffic=Reject low-priority traffic"
        ),
    )
    parser.add_argument(
        "--question",
        type=str,
        default="The database connection pool is completely saturated.",
        help="Question or problem statement to compare against the decisions.",
    )
    parser.add_argument(
        "--decision",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Decision option in KEY=VALUE format. Repeat this argument for multiple decisions. "
            "Example: --decision scale_up=Increase database resources"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for softmax scaling. Must be greater than 0.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformer model name to use for embedding generation.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON payload with keys 'question' and 'decisions'.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    return parser


def _parse_decisions(raw_decisions: Optional[list[str]]) -> Dict[str, str]:
    """Convert the command-line decision list into a dictionary."""
    decisions: Dict[str, str] = {}
    for item in raw_decisions or []:
        if "=" not in item:
            raise ValueError("Each --decision value must be in KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("Decision keys and values must be non-empty strings.")
        decisions[key] = value
    return decisions


def main(argv: Optional[list[str]] = None) -> None:
    """Run the CLI and print the selected decision ranking.

    Args:
        argv: Optional list of CLI arguments. When omitted, sys.argv is used.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.json:
        try:
            payload = json.loads(args.json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON payload: {exc}") from exc
    else:
        decisions = _parse_decisions(args.decision)
        if not decisions:
            raise SystemExit(
                "No decisions provided. Use --decision KEY=VALUE one or more times."
            )
        payload = {"question": args.question, "decisions": decisions}

    engine = LocalDecisionEngine(model_name=args.model_name)
    result = engine.evaluate(payload, temperature=args.temperature)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

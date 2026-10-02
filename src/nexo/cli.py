from .engine import LocalDecisionEngine


def main() -> None:
    import json

    engine = LocalDecisionEngine()
    payload = {
        "question": "The database connection pool is completely saturated.",
        "decisions": {
            "scale_up": "Provision a larger database instance tier immediately.",
            "drop_traffic": "Return 503 Service Unavailable for non-critical background sync jobs.",
            "do_nothing": "Wait and see if traffic spikes naturally subside.",
        },
    }
    print(json.dumps(engine.evaluate(payload), indent=2))


if __name__ == "__main__":
    main()

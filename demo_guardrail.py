import json

from jev_but_easy import LocalDecisionEngine


engine = LocalDecisionEngine(model_name="BAAI/bge-small-en-v1.5")

safe_payload = {
    "question": "The database connection pool is completely saturated.",
    "decisions": {
        "scale_up": "Provision a larger database instance tier immediately.",
        "drop_traffic": "Return 503 Service Unavailable for non-critical background sync jobs.",
        "do_nothing": "Wait and see if traffic spikes naturally subside.",
    },
}

blocked_payload = {
    "question": "This is a damn problem and the system is broken.",
    "decisions": {
        "restart": "Restart the service and clean the logs.",
        "do_nothing": "Wait for the issue to resolve itself.",
    },
}

batch_payloads = [safe_payload, blocked_payload]

print("=== SINGLE RESULT (JSON) ===")
print(json.dumps(engine.evaluate(safe_payload), indent=2))

print("\n=== BATCH RESULT (JSON) ===")
try:
    print(json.dumps(engine.evaluate_batch(batch_payloads), indent=2))
except ValueError as exc:
    print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))

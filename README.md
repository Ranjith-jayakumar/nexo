# nexo

`nexo` is a lightweight semantic decision-ranking engine for evaluating a question against multiple decision options using sentence embeddings.

## What it does

- accepts a `question` and a set of `decisions`
- validates the input payload
- blocks unsafe or profane content with a clear guardrail error
- ranks each decision by semantic similarity and confidence
- returns structured JSON for single and batch use cases

## Installation

```bash
pip install nexo
```

## Quick example

```python
from nexo import LocalDecisionEngine

engine = LocalDecisionEngine()

payload = {
    "question": "The database connection pool is completely saturated.",
    "decisions": {
        "scale_up": "Provision a larger database instance tier immediately.",
        "drop_traffic": "Return 503 Service Unavailable for non-critical background sync jobs.",
        "do_nothing": "Wait and see if traffic spikes naturally subside."
    }
}

result = engine.evaluate(payload)
print(result)
```

## Output format

```json
{
  "job_id": "9f3a2c7d9e1b4f6a8d2c1b0e7f4a9c11",
  "status": "completed",
  "guardrail_status": "passed",
  "input": {
    "question": "The database connection pool is completely saturated.",
    "decisions": {
      "scale_up": "Provision a larger database instance tier immediately.",
      "drop_traffic": "Return 503 Service Unavailable for non-critical background sync jobs.",
      "do_nothing": "Wait and see if traffic spikes naturally subside."
    }
  },
  "selected": {
    "decision": "scale_up",
    "confidence": 0.7348
  },
  "rankings": [
    {
      "decision": "scale_up",
      "description": "Provision a larger database instance tier immediately.",
      "confidence": 0.7348
    }
  ],
  "processing_time_ms": 245,
  "estimated_cost_usd": 0.00012
}
```

## Batch use

```python
results = engine.evaluate_batch([
    payload,
    payload_2,
])
```

## Guardrails

The engine blocks obviously unsafe or profane input before ranking decisions.

Example:

```python
ValueError: Guardrail error: profanity detected ('damn'). Please replace the flagged wording and try again.
```

## Requirements

- Python 3.10+
- `numpy`
- `sentence-transformers`

## License

MIT

import pytest

from jev_but_easy import LocalDecisionEngine


@pytest.fixture
def engine():
    return LocalDecisionEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_evaluate_returns_input_metadata_and_confidence(engine):
    payload = {
        "question": "The database connection pool is completely saturated.",
        "decisions": {
            "scale_up": "Provision a larger database instance tier immediately.",
            "drop_traffic": "Return 503 Service Unavailable for non-critical background sync jobs.",
            "do_nothing": "Wait and see if traffic spikes naturally subside.",
        },
    }

    result = engine.evaluate(payload)

    assert "job_id" in result
    assert len(result["job_id"]) >= 20
    assert result["job_id"].isalnum()
    assert "input" in result
    assert result["input"]["question"] == payload["question"]
    assert result["input"]["decisions"] == payload["decisions"]
    assert "selected" in result
    assert "confidence" in result["selected"]
    assert "rankings" in result
    assert result["selected"]["decision"] in payload["decisions"]
    assert result["processing_time_ms"] >= 0


def test_evaluate_blocks_profanity_prompt(engine):
    payload = {
        "question": "This is a damn problem and the system is broken.",
        "decisions": {
            "restart": "Restart the service and clean the logs.",
            "do_nothing": "Wait for the issue to resolve itself.",
        },
    }

    with pytest.raises(ValueError, match="Guardrail error"):
        engine.evaluate(payload)

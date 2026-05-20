from typing import Optional

import pytest

from src.app.schemas.action_event import ActionEvent, ContextData
from src.inference.alert_state_machine import AlertState, AlertStateMachine
from src.inference.context_policy import ContextPolicy, ContextRule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    label: str,
    scene_tag: Optional[str] = None,
    confidence: float = 0.9,
) -> ActionEvent:
    context = (
        ContextData(scene_tag=scene_tag, confidence=confidence)
        if scene_tag is not None
        else None
    )
    return ActionEvent(
        start_frame_index=0,
        end_frame_index=1,
        label=label,
        confidence=confidence,
        context=context,
    )


# ---------------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------------


def test_no_context_returns_defaults():
    policy = ContextPolicy(default_persistence_threshold=3, default_resolve_threshold=1)
    event = _evt("fight", scene_tag=None)
    assert policy.evaluate(event) == (3, 1)


def test_unknown_scene_returns_defaults():
    policy = ContextPolicy(default_persistence_threshold=3, default_resolve_threshold=1)
    event = _evt("fight", scene_tag="unknown")
    assert policy.evaluate(event) == (3, 1)


def test_no_matching_rule_returns_defaults():
    policy = ContextPolicy(
        default_persistence_threshold=3,
        default_resolve_threshold=1,
        rules={
            ("vandalism", "outdoor"): ContextRule(persistence_threshold=2, resolve_threshold=1),
        },
    )
    event = _evt("fight", scene_tag="outdoor")
    assert policy.evaluate(event) == (3, 1)


# ---------------------------------------------------------------------------
# Context-aware rules
# ---------------------------------------------------------------------------


def test_matching_rule_returns_custom_thresholds():
    policy = ContextPolicy(
        rules={
            ("fight", "outdoor"): ContextRule(persistence_threshold=2, resolve_threshold=1),
        }
    )
    event = _evt("fight", scene_tag="outdoor")
    assert policy.evaluate(event) == (2, 1)


def test_disabled_rule_blocks_event():
    policy = ContextPolicy(
        rules={
            ("vandalism", "indoor"): ContextRule(
                persistence_threshold=3, resolve_threshold=1, enabled=False
            ),
        }
    )
    event = _evt("vandalism", scene_tag="indoor")
    assert policy.evaluate(event) is None


def test_different_scene_same_label_different_thresholds():
    policy = ContextPolicy(
        rules={
            ("fight", "outdoor"): ContextRule(persistence_threshold=2, resolve_threshold=1),
            ("fight", "indoor"): ContextRule(persistence_threshold=5, resolve_threshold=2),
        }
    )
    outdoor_result = policy.evaluate(_evt("fight", scene_tag="outdoor"))
    indoor_result = policy.evaluate(_evt("fight", scene_tag="indoor"))
    assert outdoor_result == (2, 1)
    assert indoor_result == (5, 2)


def test_outdoor_lower_threshold_than_indoor():
    policy = ContextPolicy(
        rules={
            ("fight", "outdoor"): ContextRule(persistence_threshold=2, resolve_threshold=1),
            ("fight", "indoor"): ContextRule(persistence_threshold=5, resolve_threshold=2),
        }
    )
    outdoor_threshold, _ = policy.evaluate(_evt("fight", scene_tag="outdoor"))
    indoor_threshold, _ = policy.evaluate(_evt("fight", scene_tag="indoor"))
    assert outdoor_threshold < indoor_threshold


# ---------------------------------------------------------------------------
# Integration with AlertStateMachine
# ---------------------------------------------------------------------------


def test_context_policy_with_alert_state_machine_outdoor():
    policy = ContextPolicy(
        default_persistence_threshold=3,
        default_resolve_threshold=1,
        rules={
            ("fight", "outdoor"): ContextRule(persistence_threshold=2, resolve_threshold=1),
        },
    )
    event = _evt("fight", scene_tag="outdoor")
    result = policy.evaluate(event)
    assert result is not None
    persistence_threshold, resolve_threshold = result

    sm = AlertStateMachine(
        persistence_threshold=persistence_threshold,
        resolve_threshold=resolve_threshold,
        danger_labels=["fight"],
    )
    sm.process_event(event)       # hit 1 → CANDIDATE
    alert = sm.process_event(event)  # hit 2 → ACTIVE (threshold=2)
    assert alert is not None
    assert sm.get_state() == AlertState.ACTIVE


def test_context_policy_with_alert_state_machine_indoor():
    policy = ContextPolicy(
        default_persistence_threshold=3,
        default_resolve_threshold=1,
        rules={
            ("fight", "indoor"): ContextRule(persistence_threshold=5, resolve_threshold=2),
        },
    )
    event = _evt("fight", scene_tag="indoor")
    result = policy.evaluate(event)
    assert result is not None
    persistence_threshold, resolve_threshold = result

    sm = AlertStateMachine(
        persistence_threshold=persistence_threshold,
        resolve_threshold=resolve_threshold,
        danger_labels=["fight"],
    )
    sm.process_event(event)       # hit 1
    alert = sm.process_event(event)  # hit 2 — threshold=5, still CANDIDATE
    assert alert is None
    assert sm.get_state() == AlertState.CANDIDATE


def test_blocked_event_never_reaches_active():
    policy = ContextPolicy(
        rules={
            ("vandalism", "indoor"): ContextRule(
                persistence_threshold=1, resolve_threshold=1, enabled=False
            ),
        }
    )
    sm = AlertStateMachine(persistence_threshold=1, danger_labels=["vandalism"])
    event = _evt("vandalism", scene_tag="indoor")

    result = policy.evaluate(event)
    assert result is None
    # Event is not forwarded to the state machine — it remains INACTIVE.
    assert sm.get_state() == AlertState.INACTIVE


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_default_persistence_threshold_raises():
    with pytest.raises(ValueError):
        ContextPolicy(default_persistence_threshold=0)
    with pytest.raises(TypeError):
        ContextPolicy(default_persistence_threshold="3")  # type: ignore[arg-type]


def test_invalid_rule_key_raises():
    with pytest.raises(TypeError):
        ContextPolicy(
            rules={"fight": ContextRule(persistence_threshold=2, resolve_threshold=1)}  # type: ignore[dict-item]
        )


def test_invalid_rule_value_raises():
    with pytest.raises(TypeError):
        ContextPolicy(
            rules={
                ("fight", "outdoor"): {"persistence_threshold": 2, "resolve_threshold": 1}  # type: ignore[dict-item]
            }
        )

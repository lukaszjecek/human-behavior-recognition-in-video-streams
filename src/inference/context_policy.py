"""Context-aware policy for selecting alert thresholds based on scene context."""

import logging
from dataclasses import dataclass
from typing import Optional

from src.app.schemas.action_event import ActionEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextRule:
    """Rule for a single (label, scene_tag) combination.

    Attributes:
        persistence_threshold: Number of consecutive danger windows required to
            transition to ACTIVE. Overrides the policy default.
        resolve_threshold: Number of consecutive miss windows required to
            transition to RESOLVED. Overrides the policy default.
        enabled: When False, events matching this rule are blocked entirely
            and evaluate() returns None.
    """

    persistence_threshold: int
    resolve_threshold: int
    enabled: bool = True


class ContextPolicy:
    """Evaluates per-scene alert thresholds for action events.

    Selects persistence and resolve thresholds by matching an ActionEvent's
    label and scene_tag against a rule table.  Returns None to signal that
    the event should be discarded entirely (rule with enabled=False).

    Example:
        >>> policy = ContextPolicy(
        ...     default_persistence_threshold=3,
        ...     default_resolve_threshold=1,
        ...     rules={
        ...         ("fight", "outdoor"): ContextRule(persistence_threshold=2, resolve_threshold=1),
        ...         ("vandalism", "indoor"): ContextRule(
        ...             persistence_threshold=3, resolve_threshold=1, enabled=False
        ...         ),
        ...     },
        ... )
        >>> result = policy.evaluate(event)
        >>> if result is not None:
        ...     persistence_threshold, resolve_threshold = result
    """

    def __init__(
        self,
        default_persistence_threshold: int = 3,
        default_resolve_threshold: int = 1,
        rules: Optional[dict[tuple[str, str], ContextRule]] = None,
    ) -> None:
        """Initialize the context policy.

        Args:
            default_persistence_threshold: Fallback persistence threshold when
                no rule matches or context is absent. Must be >= 1.
            default_resolve_threshold: Fallback resolve threshold when no rule
                matches or context is absent. Must be >= 1.
            rules: Mapping from (label, scene_tag) to ContextRule.  Each key
                must be a tuple of two strings; each value must be a ContextRule
                instance.  None is treated as an empty rule table.

        Raises:
            TypeError: If threshold arguments are not integers, rules is not a
                dict, a key is not a tuple of (str, str), or a value is not a
                ContextRule instance.
            ValueError: If threshold arguments are less than 1.
        """
        if not isinstance(default_persistence_threshold, int) or isinstance(
            default_persistence_threshold, bool
        ):
            raise TypeError("default_persistence_threshold must be an integer")
        if default_persistence_threshold < 1:
            raise ValueError("default_persistence_threshold must be >= 1")

        if not isinstance(default_resolve_threshold, int) or isinstance(
            default_resolve_threshold, bool
        ):
            raise TypeError("default_resolve_threshold must be an integer")
        if default_resolve_threshold < 1:
            raise ValueError("default_resolve_threshold must be >= 1")

        if rules is not None:
            if not isinstance(rules, dict):
                raise TypeError("rules must be a dict or None")
            for key, value in rules.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or not isinstance(key[0], str)
                    or not isinstance(key[1], str)
                ):
                    raise TypeError(
                        "each rules key must be a tuple of (str, str), "
                        f"got {type(key).__name__!r}: {key!r}"
                    )
                if not isinstance(value, ContextRule):
                    raise TypeError(
                        "each rules value must be a ContextRule instance, "
                        f"got {type(value).__name__!r}"
                    )

        self._default_persistence_threshold = default_persistence_threshold
        self._default_resolve_threshold = default_resolve_threshold
        self._rules: dict[tuple[str, str], ContextRule] = dict(rules) if rules else {}

    def evaluate(
        self,
        event: ActionEvent,
    ) -> Optional[tuple[int, int]]:
        """Evaluate context policy for an event.

        Returns the persistence and resolve thresholds to use when calling
        AlertStateMachine, or None if the event should be blocked entirely.

        Args:
            event: ActionEvent to evaluate.  The label and context.scene_tag
                fields are used for rule lookup.

        Returns:
            Tuple (persistence_threshold, resolve_threshold) to pass to
            AlertStateMachine, or None if the event is blocked by policy.
        """
        defaults = (self._default_persistence_threshold, self._default_resolve_threshold)

        if event.context is None:
            logger.debug("No context on event label=%r — using defaults", event.label)
            return defaults

        scene_tag = event.context.scene_tag
        if scene_tag == "unknown":
            logger.debug("scene_tag='unknown' for label=%r — using defaults", event.label)
            return defaults

        key = (event.label, scene_tag)
        rule = self._rules.get(key)

        if rule is None:
            logger.debug("No rule for key=%r — using defaults", key)
            return defaults

        if not rule.enabled:
            logger.debug("Rule for key=%r has enabled=False — blocking event", key)
            return None

        logger.debug(
            "Rule matched for key=%r — persistence=%d, resolve=%d",
            key,
            rule.persistence_threshold,
            rule.resolve_threshold,
        )
        return (rule.persistence_threshold, rule.resolve_threshold)


__all__ = ["ContextPolicy", "ContextRule"]

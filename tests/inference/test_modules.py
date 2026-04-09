#!/usr/bin/env python
"""
Test script to verify action_event and json_writer modules work correctly.
Generates a sample output JSON file demonstrating the schema.
"""

import json
import sys
from pathlib import Path

# Add root directory to path (go up 3 levels: tests/inference -> tests -> . -> .)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.action_event import ActionEvent, ActionEventLog
from src.inference.engine import InferenceResult
from src.inference.json_writer import ActionEventWriter


def main():
    """Run tests and generate sample output."""
    print("=" * 70)
    print("ACTION EVENT & JSON WRITER MODULE VALIDATION")
    print("=" * 70)
    print()

    # Test 1: ActionEvent Schema
    print("TEST 1: ActionEvent Schema Validation")
    print("-" * 70)
    try:
        event = ActionEvent(
            start_frame_index=0,
            end_frame_index=15,
            label="walking",
            confidence=0.95,
            start_timestamp=0.0,
            end_timestamp=0.5,
            track_id=1,
        )
        print("✓ Created valid ActionEvent")
        print(f"  - Label: {event.label}, Confidence: {event.confidence}")
        print(f"  - Frames: {event.start_frame_index}-{event.end_frame_index}")
    except Exception as e:
        print(f"✗ Failed to create ActionEvent: {e}")
        return False

    # Test 2: ActionEvent Serialization
    print("\nTEST 2: ActionEvent JSON Serialization")
    print("-" * 70)
    try:
        data = event.to_dict()
        json_str = event.to_json()
        print("✓ Serialized ActionEvent to JSON")
        print(f"  - JSON: {json_str}")
    except Exception as e:
        print(f"✗ Failed to serialize ActionEvent: {e}")
        return False

    # Test 3: ActionEvent Validation
    print("\nTEST 3: ActionEvent Validation")
    print("-" * 70)
    validation_tests = [
        ("Invalid frames (end < start)", {
            "start_frame_index": 10,
            "end_frame_index": 5,
            "label": "test",
            "confidence": 0.5,
        }),
        ("Invalid confidence (> 1.0)", {
            "start_frame_index": 0,
            "end_frame_index": 15,
            "label": "test",
            "confidence": 1.5,
        }),
        ("Empty label", {
            "start_frame_index": 0,
            "end_frame_index": 15,
            "label": "",
            "confidence": 0.5,
        }),
    ]

    for test_name, invalid_data in validation_tests:
        try:
            ActionEvent(**invalid_data)
            print(f"✗ {test_name}: Should have raised ValueError")
            return False
        except ValueError:
            print(f"✓ {test_name}: Correctly rejected")

    # Test 4: ActionEventLog
    print("\nTEST 4: ActionEventLog Management")
    print("-" * 70)
    try:
        log = ActionEventLog()
        log.add_event(ActionEvent(0, 15, "walking", 0.95))
        log.add_event(ActionEvent(16, 31, "running", 0.87))
        log.add_event(ActionEvent(32, 47, "jumping", 0.92))
        print("✓ Added 3 events to ActionEventLog")
        print(f"  - Event count: {len(log.events)}")
    except Exception as e:
        print(f"✗ Failed to manage ActionEventLog: {e}")
        return False

    # Test 5: ActionEventLog Serialization
    print("\nTEST 5: ActionEventLog JSON Serialization")
    print("-" * 70)
    try:
        log_data = log.to_dict()
        log_json = log.to_json()
        print("✓ Serialized ActionEventLog to JSON")
        print(f"  - Event count: {log_data['event_count']}")
        parsed = json.loads(log_json)
        print(f"  - Valid JSON with {len(parsed['events'])} events")
    except Exception as e:
        print(f"✗ Failed to serialize ActionEventLog: {e}")
        return False

    # Test 6: ActionEventWriter with Various Predictions
    print("\nTEST 6: ActionEventWriter Prediction Handling")
    print("-" * 70)
    writer = ActionEventWriter(class_labels=["walking", "running", "jumping"])

    test_predictions = [
        ("Dict with class/confidence", {
            "prediction": {"class": 0, "confidence": 0.95},
            "track_id": 1,
        }),
        ("Dict with label/confidence", {
            "prediction": {"label": "running", "confidence": 0.88},
            "track_id": 2,
        }),
        ("List (logits)", {
            "prediction": [0.1, 0.7, 0.2],
            "track_id": 1,
        }),
        ("Single confidence value", {
            "prediction": 0.82,
            "track_id": 3,
        }),
    ]

    for test_name, test_data in test_predictions:
        try:
            pred = test_data["prediction"]
            result = InferenceResult(
                window=tuple(),
                start_frame_index=writer.log.events.__len__() * 16,
                end_frame_index=writer.log.events.__len__() * 16 + 15,
                start_timestamp=writer.log.events.__len__() * 0.5,
                end_timestamp=writer.log.events.__len__() * 0.5 + 0.5,
                prediction=pred,
            )
            event = writer.process_inference_result(result, test_data.get("track_id"))
            if event:
                print(f"✓ {test_name}")
                print(f"  - Label: {event.label}, Confidence: {event.confidence}")
            else:
                print(f"✗ {test_name}: Failed to process prediction")
                return False
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            return False

    # Test 7: Handle invalid predictions gracefully
    print("\nTEST 7: Invalid Prediction Handling")
    print("-" * 70)
    try:
        result_none = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=None,
        )
        event_none = writer.process_inference_result(result_none)
        if event_none is None:
            print("✓ Correctly skipped None prediction")
        else:
            print("✗ Should have returned None for None prediction")
            return False
    except Exception as e:
        print(f"✗ Failed to handle None prediction: {e}")
        return False

    # Test 8: Generate Sample Output File
    print("\nTEST 8: Generate Sample Output File")
    print("-" * 70)
    try:
        # Create sample log with diverse actions
        sample_log = ActionEventLog()
        sample_events = [
            ActionEvent(0, 15, "walking", 0.95, 0.0, 0.5, 1),
            ActionEvent(16, 31, "running", 0.87, 0.533, 1.033, 1),
            ActionEvent(32, 47, "jumping", 0.92, 1.067, 1.567, 1),
            ActionEvent(48, 63, "standing", 0.98, 1.6, 2.1, 1),
            ActionEvent(64, 79, "walking", 0.91, 2.133, 2.633, 2),
        ]
        for event in sample_events:
            sample_log.add_event(event)

        # Save to file
        output_path = Path(__file__).parent / "data" / "logs" / "sample_actions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log.save_to_file(str(output_path))

        print("✓ Generated sample output file")
        print(f"  - Path: {output_path}")
        print(f"  - Events: {len(sample_log.events)}")

        # Verify file contents
        with open(output_path) as f:
            sample_data = json.load(f)
        print(f"  - File verified with {sample_data['event_count']} events")
        print("\nSample Output Content:")
        print("-" * 70)
        print(json.dumps(sample_data, indent=2))

    except Exception as e:
        print(f"✗ Failed to generate sample output: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nModules ready for use:")
    print("  - src.inference.action_event.ActionEvent")
    print("  - src.inference.action_event.ActionEventLog")
    print("  - src.inference.json_writer.ActionEventWriter")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""Tests for runtime reliability: interruption, shutdown, reconnect, failure surfacing."""
from __future__ import annotations

import threading
from queue import Queue
from threading import Event
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference.engine import InferenceEngine
from src.inference.offline_runtime import (
    EOF_SENTINEL,
    RuntimeFailureState,
    SourceInterruptedError,
    _interruptible_sleep,
    consume_frame_queue,
    produce_frames_from_source,
    produce_frames_safe,
    produce_frames_with_reconnect,
    run_source,
    run_source_with_reconnect,
)
from src.inference.source_adapters import RtspSourceAdapter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _DummyModel:
    def __call__(self, window):
        return {"label": "action", "confidence": 0.9}


def _make_frames(n: int = 10) -> List[np.ndarray]:
    return [np.full((64, 64, 3), fill_value=i, dtype=np.uint8) for i in range(n)]


class _FakeCapture:
    """Fake cv2.VideoCapture that returns a fixed sequence of frames."""

    def __init__(self, frames: List[np.ndarray], opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


def _patch_capture(monkeypatch, frames: List[np.ndarray]) -> dict:
    """Monkeypatch cv2.VideoCapture to return _FakeCapture(frames)."""
    state: dict = {}

    def _fake(source_ref):
        cap = _FakeCapture(list(frames))
        state["capture"] = cap
        state["source_ref"] = source_ref
        return cap

    monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)
    return state


# ---------------------------------------------------------------------------
# SourceInterruptedError
# ---------------------------------------------------------------------------


class TestSourceInterruptedError:
    def test_message_contains_source_ref_and_frame_count(self):
        err = SourceInterruptedError(source_ref="rtsp://host/live", frames_read=42)
        assert "rtsp://host/live" in str(err)
        assert "42" in str(err)

    def test_attributes_accessible(self):
        err = SourceInterruptedError(source_ref="rtsp://host/live", frames_read=7)
        assert err.source_ref == "rtsp://host/live"
        assert err.frames_read == 7

    def test_is_runtime_error(self):
        err = SourceInterruptedError("rtsp://x", 0)
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# RuntimeFailureState
# ---------------------------------------------------------------------------


class TestRuntimeFailureState:
    def test_stores_error_and_phase(self):
        exc = ValueError("boom")
        state = RuntimeFailureState(error=exc, phase="producer", frames_before_failure=5)
        assert state.error is exc
        assert state.phase == "producer"
        assert state.frames_before_failure == 5

    def test_repr_includes_phase(self):
        exc = RuntimeError("x")
        state = RuntimeFailureState(error=exc, phase="consumer")
        assert "consumer" in repr(state)

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            RuntimeFailureState(error=RuntimeError(), phase="invalid_phase")

    def test_valid_phases_accepted(self):
        for phase in ("producer", "consumer", "unknown"):
            RuntimeFailureState(error=Exception(), phase=phase)


# ---------------------------------------------------------------------------
# produce_frames_from_source – stop_event support
# ---------------------------------------------------------------------------


class TestProduceFramesFromSourceStopEvent:
    def test_stop_event_halts_producer_before_all_frames(self, monkeypatch):
        frames = _make_frames(20)
        stop = Event()
        queue: Queue = Queue()

        call_count = [0]

        class _SlowCapture(_FakeCapture):
            def read(self):
                call_count[0] += 1
                if call_count[0] == 5:
                    stop.set()
                return super().read()

        def _fake(source_ref):
            return _SlowCapture(list(frames))

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        produce_frames_from_source(adapter, queue, stop_event=stop)

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        # EOF sentinel must always be present.
        assert items[-1] is EOF_SENTINEL
        # Producer was cut short.
        assert len(items) < len(frames) + 1  # +1 for sentinel

    def test_no_stop_event_reads_all_frames(self, monkeypatch):
        frames = _make_frames(8)
        state = _patch_capture(monkeypatch, frames)
        queue: Queue = Queue()

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        produce_frames_from_source(adapter, queue, stop_event=None)

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        frame_items = [i for i in items if i is not EOF_SENTINEL]
        assert len(frame_items) == 8
        assert items[-1] is EOF_SENTINEL
        assert state["capture"].released is True

    def test_capture_always_released_even_on_stop(self, monkeypatch):
        frames = _make_frames(10)
        stop = Event()
        stop.set()  # stopped before even starting
        queue: Queue = Queue()

        captured_cap: dict = {}

        def _fake(source_ref):
            cap = _FakeCapture(list(frames))
            captured_cap["cap"] = cap
            return cap

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)
        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        produce_frames_from_source(adapter, queue, stop_event=stop)

        assert captured_cap["cap"].released is True

    def test_eof_sentinel_always_enqueued_on_open_failure(self, monkeypatch):
        captured_cap = {}

        def _fake(_):
            cap = _FakeCapture([], opened=False)
            captured_cap["cap"] = cap
            return cap

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)
        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        queue: Queue = Queue()

        with pytest.raises(RuntimeError, match="Could not open"):
            produce_frames_from_source(adapter, queue)

        # Sentinel must still be present.
        assert queue.get_nowait() is EOF_SENTINEL
        assert captured_cap["cap"].released is True


# ---------------------------------------------------------------------------
# consume_frame_queue – stop_event support
# ---------------------------------------------------------------------------


class TestConsumeFrameQueueStopEvent:
    def test_stop_event_drains_queue_and_exits(self):
        queue: Queue = Queue()
        frames = _make_frames(20)
        for f in frames:
            queue.put(f)
        queue.put(EOF_SENTINEL)

        stop = Event()
        stop.set()

        engine = InferenceEngine(model=_DummyModel())
        stats: dict = {}
        consume_frame_queue(queue, engine, stats, stop_event=stop)

        # Consumer must have exited and written stats.
        assert "frame_count" in stats
        # Because stop was already set, no frames should have been processed.
        assert stats["frame_count"] == 0

    def test_without_stop_event_processes_all_frames(self):
        queue: Queue = Queue()
        frames = _make_frames(10)
        for f in frames:
            queue.put(f)
        queue.put(EOF_SENTINEL)

        engine = InferenceEngine(window_size=4, stride=2, model=_DummyModel())
        stats: dict = {}
        consume_frame_queue(queue, engine, stats)

        assert stats["frame_count"] == 10

    def test_stop_event_set_mid_stream_terminates(self):
        """Consumer terminates when stop_event is set while consuming."""
        queue: Queue = Queue()
        stop = Event()

        def _producer():
            for i in range(100):
                if stop.is_set():
                    break
                queue.put(np.zeros((64, 64, 3), dtype=np.uint8))
            queue.put(EOF_SENTINEL)

        t = threading.Thread(target=_producer)
        t.start()

        engine = InferenceEngine(window_size=4, stride=2, model=_DummyModel())
        stats: dict = {}

        def _trigger_stop():
            # Let a few frames flow through before stopping.
            threading.Event().wait(timeout=0.02)
            stop.set()

        stopper = threading.Thread(target=_trigger_stop)
        stopper.start()

        consume_frame_queue(queue, engine, stats, stop_event=stop)

        t.join(timeout=2)
        stopper.join(timeout=1)

        # Consumer exited – frame_count must be set.
        assert "frame_count" in stats


# ---------------------------------------------------------------------------
# produce_frames_with_reconnect
# ---------------------------------------------------------------------------


class TestProduceFramesWithReconnect:
    def _make_rtsp_adapter(self) -> RtspSourceAdapter:
        return RtspSourceAdapter(rtsp_uri="rtsp://host/live")

    def test_non_rtsp_falls_back_to_simple_producer(self, monkeypatch, tmp_path):
        """File sources bypass reconnect logic."""
        video_path = tmp_path / "sample.mp4"
        frames = _make_frames(6)

        called_open: list = []

        class _CapWithTracking(_FakeCapture):
            pass

        def _fake(source_ref):
            called_open.append(source_ref)
            return _CapWithTracking(list(frames))

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        from src.inference.source_adapters import FileSourceAdapter

        video_path.write_bytes(b"\x00" * 100)
        # FileSourceAdapter validates file existence; write bytes first.
        adapter = FileSourceAdapter(video_path=video_path)
        queue: Queue = Queue()
        stats: dict = {"producer_error": None}

        produce_frames_with_reconnect(adapter, queue, stats)

        # EOF sentinel must be present.
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items[-1] is EOF_SENTINEL

    def test_reconnects_after_read_failure(self, monkeypatch):
        """Producer reconnects once when the stream drops mid-read."""
        adapter = self._make_rtsp_adapter()

        open_calls: list = []
        frames_batch1 = _make_frames(5)
        frames_batch2 = _make_frames(5)
        stop = Event()

        def _fake(_source_ref):
            call_no = len(open_calls)
            open_calls.append(call_no)
            if call_no == 0:
                # First connection: returns 5 frames then drops.
                return _FakeCapture(list(frames_batch1))
            if call_no == 1:
                # Second connection: returns 5 frames then drops.
                return _FakeCapture(list(frames_batch2))
            
            # Stop the loop after second connection drops
            stop.set()
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        queue: Queue = Queue()
        stats: dict = {"producer_error": None}

        produce_frames_with_reconnect(
            adapter,
            queue,
            stats,
            stop_event=stop,
            retry_delay=0.001,
            backoff_factor=1.0,
            max_retries=3,
        )

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        frame_items = [i for i in items if i is not EOF_SENTINEL]
        assert len(frame_items) == 10  # 5 from batch1 + 5 from batch2
        assert items[-1] is EOF_SENTINEL
        assert len(open_calls) == 3
        assert stats["producer_error"] is None

    def test_raises_source_interrupted_after_max_retries(self, monkeypatch):
        """Producer gives up after max_retries and sets producer_error."""
        adapter = self._make_rtsp_adapter()

        def _fake(_source_ref):
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        queue: Queue = Queue()
        stats: dict = {"producer_error": None}

        produce_frames_with_reconnect(
            adapter,
            queue,
            stats,
            retry_delay=0.001,
            backoff_factor=1.0,
            max_retries=2,
        )

        # EOF sentinel always pushed.
        sentinel = queue.get_nowait()
        assert sentinel is EOF_SENTINEL
        assert isinstance(stats["producer_error"], SourceInterruptedError)

    def test_stop_event_aborts_reconnect_sleep(self, monkeypatch):
        """Setting stop_event during the retry sleep causes early exit."""
        adapter = self._make_rtsp_adapter()
        stop = Event()

        def _fake(_source_ref):
            stop.set()  # trigger stop on every open attempt
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        queue: Queue = Queue()
        stats: dict = {"producer_error": None}

        produce_frames_with_reconnect(
            adapter,
            queue,
            stats,
            stop_event=stop,
            retry_delay=10.0,  # would hang if sleep isn't interrupted
            max_retries=5,
        )

        # Should finish quickly without sleeping the full 10 s.
        assert queue.get_nowait() is EOF_SENTINEL

    def test_stop_event_set_before_start_exits_immediately(self, monkeypatch):
        adapter = self._make_rtsp_adapter()
        stop = Event()
        stop.set()

        open_calls: list = []

        def _fake(_):
            open_calls.append(1)
            return _FakeCapture(_make_frames(10))

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        queue: Queue = Queue()
        stats: dict = {"producer_error": None}

        produce_frames_with_reconnect(adapter, queue, stats, stop_event=stop)

        # No open should have been attempted (stop was already set).
        assert open_calls == []
        assert queue.get_nowait() is EOF_SENTINEL


# ---------------------------------------------------------------------------
# run_source – stop_event integration
# ---------------------------------------------------------------------------


class TestRunSourceStopEvent:
    def test_stop_event_terminates_session_gracefully(self, monkeypatch):
        frames = _make_frames(50)
        stop = Event()
        call_count = [0]

        def _fake(_source_ref):
            class _SlowCap(_FakeCapture):
                def read(self_inner):
                    call_count[0] += 1
                    if call_count[0] == 10:
                        stop.set()
                    return super().read()

            return _SlowCap(list(frames))

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        engine = InferenceEngine(window_size=4, stride=2, model=_DummyModel())

        frame_count, *_ = run_source(
            source_adapter=adapter,
            engine=engine,
            emit_runtime_summary=False,
            stop_event=stop,
        )

        # Less than all 50 frames were processed.
        assert frame_count < 50

    def test_run_source_producer_error_surfaces_as_exception(self, monkeypatch):
        def _fake(_source_ref):
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        engine = InferenceEngine(model=_DummyModel())

        with pytest.raises(RuntimeFailureState, match="Could not open") as exc_info:
            run_source(source_adapter=adapter, engine=engine, emit_runtime_summary=False)
            
        assert exc_info.value.phase == "producer"
        assert isinstance(exc_info.value.error, RuntimeError)


# ---------------------------------------------------------------------------
# run_source_with_reconnect
# ---------------------------------------------------------------------------


class TestRunSourceWithReconnect:
    def test_reconnect_runner_returns_results_after_reconnect(self, monkeypatch):
        open_calls: list = []
        frames_a = _make_frames(4)
        frames_b = _make_frames(4)
        stop = Event()

        def _fake(_source_ref):
            n = len(open_calls)
            open_calls.append(n)
            if n == 0:
                return _FakeCapture(list(frames_a))
            if n == 1:
                return _FakeCapture(list(frames_b))
            
            stop.set()
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        engine = InferenceEngine(window_size=4, stride=2, model=_DummyModel())

        frame_count, inference_count, results, events = run_source_with_reconnect(
            source_adapter=adapter,
            engine=engine,
            emit_runtime_summary=False,
            stop_event=stop,
            retry_delay=0.001,
            backoff_factor=1.0,
            max_retries=3,
        )

        assert frame_count == 8
        assert inference_count > 0

    def test_reconnect_runner_raises_on_exhausted_retries(self, monkeypatch):
        def _fake(_):
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        engine = InferenceEngine(model=_DummyModel())

        with pytest.raises(RuntimeFailureState) as exc_info:
            run_source_with_reconnect(
                source_adapter=adapter,
                engine=engine,
                emit_runtime_summary=False,
                retry_delay=0.001,
                max_retries=1,
            )
            
        assert exc_info.value.phase == "producer"
        assert isinstance(exc_info.value.error, SourceInterruptedError)

    def test_stop_event_terminates_reconnect_runner(self, monkeypatch):
        stop = Event()
        stop.set()

        def _fake(_):
            return _FakeCapture(_make_frames(100))

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        engine = InferenceEngine(model=_DummyModel())

        frame_count, *_ = run_source_with_reconnect(
            source_adapter=adapter,
            engine=engine,
            emit_runtime_summary=False,
            stop_event=stop,
        )
        assert frame_count == 0


# ---------------------------------------------------------------------------
# _interruptible_sleep
# ---------------------------------------------------------------------------


class TestInterruptibleSleep:
    def test_returns_early_when_event_set(self):
        stop = Event()
        stop.set()
        import time

        start = time.monotonic()
        _interruptible_sleep(5.0, stop)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # should not wait 5 s

    def test_sleeps_full_duration_when_event_not_set(self):
        stop = Event()
        import time

        start = time.monotonic()
        _interruptible_sleep(0.05, stop)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04

    def test_works_without_event(self):
        import time

        start = time.monotonic()
        _interruptible_sleep(0.05, None)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04


# ---------------------------------------------------------------------------
# produce_frames_safe backward-compat
# ---------------------------------------------------------------------------


class TestProduceFramesSafe:
    def test_stores_exception_in_stats(self, monkeypatch):
        def _fake(_):
            return _FakeCapture([], opened=False)

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        queue: Queue = Queue()
        stats: dict = {"producer_error": None}
        produce_frames_safe(adapter, queue, stats)

        assert isinstance(stats["producer_error"], RuntimeError)
        # Sentinel still queued.
        assert queue.get_nowait() is EOF_SENTINEL

    def test_respects_stop_event(self, monkeypatch):
        frames = _make_frames(20)
        stop = Event()

        call_count = [0]

        def _fake(_):
            class _Cap(_FakeCapture):
                def read(self_inner):
                    call_count[0] += 1
                    if call_count[0] == 3:
                        stop.set()
                    return super().read()

            return _Cap(list(frames))

        monkeypatch.setattr("src.inference.source_adapters.cv2.VideoCapture", _fake)

        adapter = RtspSourceAdapter(rtsp_uri="rtsp://host/stream")
        queue: Queue = Queue()
        stats: dict = {"producer_error": None}
        produce_frames_safe(adapter, queue, stats, stop_event=stop)

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        assert items[-1] is EOF_SENTINEL
        assert len(items) < len(frames) + 1

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the viewer-agnostic live MP4 recorder.

These tests exercise :class:`newton._src.viewer.recording.LiveMp4Recorder`
directly without instantiating any viewer. Tests that require an actual
``ffmpeg`` binary are skipped when neither ``imageio-ffmpeg`` nor a system
``ffmpeg`` is available.
"""

from __future__ import annotations

import logging
import queue as _queue
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import newton._src.viewer.recording as recording_mod
from newton._src.viewer.recording import LiveMp4Recorder, _resolve_ffmpeg_executable


def _ffmpeg_available() -> bool:
    return _resolve_ffmpeg_executable() is not None


class TestQualityClamping(unittest.TestCase):
    """Quality setter must clamp to [0, 100]."""

    def test_clamping_and_roundtrip(self):
        rec = LiveMp4Recorder()
        self.assertEqual(rec.quality, 90.0)
        rec.set_quality(50.0)
        self.assertEqual(rec.quality, 50.0)
        rec.set_quality(-5.0)
        self.assertEqual(rec.quality, 0.0)
        rec.set_quality(120.0)
        self.assertEqual(rec.quality, 100.0)


class TestSuggestedOutputPath(unittest.TestCase):
    """Default output path uses the configured prefix and ends in ``.mp4``."""

    def test_prefix_used_in_suggested_path(self):
        rec = LiveMp4Recorder()
        rec.set_filename_prefix("unit_test_recording")
        path = rec.suggested_output_path()
        self.assertEqual(path.suffix, ".mp4")
        self.assertTrue(path.name.startswith("unit_test_recording_"))


class TestWriteFrameWithoutStart(unittest.TestCase):
    """Calling ``write_frame`` before ``start`` must be a silent no-op."""

    def test_write_without_start_is_noop(self):
        rec = LiveMp4Recorder()
        rec.write_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertFalse(rec.is_recording)
        self.assertIsNone(rec.stop())


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not available (install imageio-ffmpeg or system ffmpeg)")
class TestLiveMp4Recording(unittest.TestCase):
    """End-to-end recording into a temporary directory.

    Pinned to ``libx264`` so the tests do not depend on the host having a
    working NVENC/QSV/AMF driver (e.g. CPU-only CI runners).
    """

    def setUp(self):
        self._encoder_patch = patch.object(LiveMp4Recorder, "_pick_encoder", return_value="libx264")
        self._encoder_patch.start()

    def tearDown(self):
        self._encoder_patch.stop()

    def test_record_synthetic_frames(self):
        width, height, n_frames = 256, 256, 24
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.mp4"
            rec = LiveMp4Recorder()
            rec.set_quality(50.0)
            self.assertTrue(
                rec.start(width=width, height=height, fps=30.0, output_path=out),
                "Recorder failed to start with ffmpeg available",
            )
            try:
                self.assertTrue(rec.is_recording)
                for i in range(n_frames):
                    frame = np.full((height, width, 3), i * 20 % 255, dtype=np.uint8)
                    rec.write_frame(frame)
            finally:
                stopped = rec.stop()

            self.assertEqual(stopped, out)
            self.assertFalse(rec.is_recording)
            self.assertTrue(out.exists(), f"Expected MP4 output at {out}")
            self.assertGreater(out.stat().st_size, 0, "MP4 output should be non-empty")

    def test_rejects_wrong_shape_and_dtype(self):
        size = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "shape.mp4"
            rec = LiveMp4Recorder()
            self.assertTrue(rec.start(size, size, fps=15.0, output_path=out))
            try:
                # Wrong dtype: silently dropped.
                rec.write_frame(np.zeros((size, size, 3), dtype=np.float32))
                # Wrong channel count.
                rec.write_frame(np.zeros((size, size, 4), dtype=np.uint8))
                # Wrong size.
                rec.write_frame(np.zeros((size // 2, size // 2, 3), dtype=np.uint8))
                # A handful of valid frames so the encoder has something to flush.
                for _ in range(8):
                    rec.write_frame(np.zeros((size, size, 3), dtype=np.uint8))
            finally:
                rec.stop()
            self.assertTrue(out.exists())


class TestQueueDropping(unittest.TestCase):
    """Pushing more than ``maxsize`` frames without a worker must not raise."""

    def test_drops_excess_frames_silently(self):
        rec = LiveMp4Recorder()

        # Simulate an active recording with a real queue but no live ffmpeg
        # subprocess: we manipulate internals directly to avoid spawning ffmpeg.
        # ``write_frame`` only cares about ``is_recording`` and ``self._queue``.

        class _DummyProc:
            pass

        rec._proc = _DummyProc()
        rec._queue = _queue.Queue(maxsize=4)
        rec._width = 16
        rec._height = 16
        try:
            for _ in range(64):
                rec.write_frame(np.zeros((16, 16, 3), dtype=np.uint8))
            self.assertLessEqual(rec._queue.qsize(), 4)
        finally:
            # Manual cleanup to avoid touching the (non-existent) worker thread.
            rec._proc = None
            rec._queue = None


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not available")
class TestDoubleStartIsIdempotent(unittest.TestCase):
    """Calling ``start`` twice is a no-op while already recording."""

    def setUp(self):
        self._encoder_patch = patch.object(LiveMp4Recorder, "_pick_encoder", return_value="libx264")
        self._encoder_patch.start()

    def tearDown(self):
        self._encoder_patch.stop()

    def test_double_start(self):
        size = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "double.mp4"
            rec = LiveMp4Recorder()
            self.assertTrue(rec.start(size, size, fps=15.0, output_path=out))
            try:
                # Second start should report success but not respawn ffmpeg.
                self.assertTrue(rec.start(size * 2, size * 2, fps=30.0, output_path=Path(tmpdir) / "other.mp4"))
                self.assertEqual(rec.output_path, out)
                for _ in range(4):
                    rec.write_frame(np.zeros((size, size, 3), dtype=np.uint8))
            finally:
                rec.stop()
            self.assertTrue(out.exists())
            # Confirm the second path was not created.
            self.assertFalse((Path(tmpdir) / "other.mp4").exists())


class TestFfmpegFallback(unittest.TestCase):
    """When ffmpeg is truly unavailable, start() must return False (not raise)."""

    def test_start_returns_false_when_ffmpeg_missing(self):
        rec = LiveMp4Recorder()

        # Force both resolution paths to fail.
        original_which = shutil.which
        original_resolve = recording_mod._resolve_ffmpeg_executable

        def _no_ffmpeg(_cmd: str) -> str | None:
            return None

        def _no_resolve() -> str | None:
            return None

        shutil.which = _no_ffmpeg  # type: ignore[assignment]
        recording_mod._resolve_ffmpeg_executable = _no_resolve  # type: ignore[assignment]
        try:
            self.assertFalse(rec.start(width=16, height=16, fps=30.0))
            self.assertFalse(rec.is_recording)
        finally:
            shutil.which = original_which  # type: ignore[assignment]
            recording_mod._resolve_ffmpeg_executable = original_resolve  # type: ignore[assignment]
        # Ensure no lingering subprocess handle.
        self.assertIsNone(rec.stop())


if __name__ == "__main__":
    # Surface log output during local debugging runs.
    logging.basicConfig(level=logging.INFO)
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional live MP4 recording helper for Newton viewers.

The recorder is viewer-agnostic: it takes ``HxWx3`` ``uint8`` RGB frames and
streams them to an ``ffmpeg`` subprocess over stdin. Frames are queued on a
bounded queue and drained by a daemon writer thread; on sustained pressure
frames are dropped rather than blocking the rendering thread, which keeps the
viewer real-time.

The ``ffmpeg`` binary is resolved via ``imageio_ffmpeg.get_ffmpeg_exe()`` when
that optional dependency is installed; otherwise a system ``ffmpeg`` on
``PATH`` is used. If neither is available the recorder silently disables
itself (``start()`` returns ``False``).
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_ffmpeg_executable() -> str | None:
    """Return a path to an ffmpeg executable or None if unavailable.

    Prefers the binary bundled by ``imageio-ffmpeg`` (no system install
    required); falls back to whatever ``ffmpeg`` is on ``PATH``.
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


class LiveMp4Recorder:
    """Record RGB frames to MP4 via an ``ffmpeg`` subprocess.

    Thread-safety: ``start``/``stop`` must be called from the same thread
    (typically the viewer's render thread). ``write_frame`` is also expected
    to be called from that thread; the writer thread is internal.
    """

    #: Maximum number of pending frames buffered between the renderer and the
    #: ffmpeg writer thread. Large enough to absorb the multi-second hiccups
    #: that occur when the encoder primes its rate-control look-ahead or when
    #: the OS briefly preempts the writer thread, while still bounded so a
    #: stalled ffmpeg cannot exhaust memory.
    _QUEUE_MAXSIZE = 256

    #: Minimum interval (seconds) between successive "dropped frame" warnings
    #: so a sustained slow-encoder stall doesn't flood the log.
    _DROP_WARN_INTERVAL_S = 5.0

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[bytes | None] | None = None
        self._worker: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._output_path: Path | None = None
        self._width = 0
        self._height = 0
        self._quality = 90.0
        self._filename_prefix = "newton_recording"
        self._dropped_frames = 0
        self._last_drop_warn_t = 0.0

    @property
    def is_recording(self) -> bool:
        return self._proc is not None

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def quality(self) -> float:
        """Recording quality in [0, 100], where 100 is highest quality."""
        return float(self._quality)

    def set_quality(self, quality: float):
        """Set recording quality in [0, 100], where 100 is highest quality."""
        self._quality = float(np.clip(float(quality), 0.0, 100.0))

    def set_filename_prefix(self, prefix: str):
        """Set the basename prefix used by :meth:`suggested_output_path`."""
        if prefix:
            self._filename_prefix = str(prefix)

    def default_output_directory(self) -> Path:
        """Return a cross-platform default directory for recordings."""
        videos_dir = Path.home() / "Videos"
        if videos_dir.is_dir():
            return videos_dir / "NewtonRecordings"
        return Path.home() / "NewtonRecordings"

    def suggested_output_path(self) -> Path:
        """Return a timestamped default output path."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.default_output_directory() / f"{self._filename_prefix}_{stamp}.mp4"

    def _pick_encoder(self, ffmpeg_exe: str) -> str:
        preferred = ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")
        try:
            result = subprocess.run(
                [ffmpeg_exe, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            out = result.stdout or ""
            for codec in preferred:
                if codec in out:
                    return codec
        except Exception:
            pass
        return "libx264"

    def _stderr_reader_loop(self):
        """Drain ffmpeg's stderr and keep the most recent lines around.

        ffmpeg writes its banner, per-frame progress, and any encoder errors
        to stderr. We need to drain the pipe so the subprocess doesn't block
        when its stderr buffer fills, and we keep the tail so that on an
        unexpected exit we can surface the actual encoder error.
        """
        assert self._proc is not None
        assert self._proc.stderr is not None
        max_lines = 200
        try:
            for raw in self._proc.stderr:
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                if not line:
                    continue
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > max_lines:
                    del self._stderr_lines[: len(self._stderr_lines) - max_lines]
        except Exception:
            pass

    def _writer_loop(self):
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._queue is not None
        unexpected_exit = False
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    self._proc.stdin.write(item)
                except BrokenPipeError:
                    unexpected_exit = True
                    break
        finally:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            if unexpected_exit:
                # ffmpeg closed its stdin before we asked it to stop. This
                # usually means the encoder couldn't initialize (e.g. a
                # hardware encoder on a machine without the driver).
                # Surface it instead of silently producing an empty file.
                tail = self._format_stderr_tail()
                if tail:
                    logger.warning(
                        "ffmpeg exited unexpectedly during recording to %s; "
                        "the output file may be empty or truncated. ffmpeg stderr tail:\n%s",
                        self._output_path,
                        tail,
                    )
                else:
                    logger.warning(
                        "ffmpeg exited unexpectedly during recording to %s; the output file may be empty or truncated.",
                        self._output_path,
                    )

    def _format_stderr_tail(self, max_lines: int = 20) -> str:
        """Return the tail of captured ffmpeg stderr, indented for logging."""
        if not self._stderr_lines:
            return ""
        tail = self._stderr_lines[-max_lines:]
        return "\n".join("  " + line for line in tail)

    def _note_dropped_frame(self):
        """Record one dropped frame and log a rate-limited warning."""
        self._dropped_frames += 1
        now = time.monotonic()
        if now - self._last_drop_warn_t < self._DROP_WARN_INTERVAL_S:
            return
        self._last_drop_warn_t = now
        logger.warning(
            "MP4 recording dropped %d frame(s) so far due to encoder backpressure; "
            "the recorded video may stutter. Consider lowering the recording FPS or quality.",
            self._dropped_frames,
        )

    def start(
        self,
        width: int,
        height: int,
        fps: float = 60.0,
        output_path: str | Path | None = None,
        flip_vertical: bool = False,
    ) -> bool:
        """Start recording. Returns False when ``ffmpeg`` is unavailable.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frames-per-second metadata written into the MP4.
            output_path: Output file path. ``None`` selects a timestamped
                default under :meth:`default_output_directory`.
            flip_vertical: If ``True``, ffmpeg flips frames vertically. Set
                this when feeding raw OpenGL bottom-left-origin pixels;
                leave ``False`` when frames are already top-left-origin
                (e.g. from the Newton PBO readback kernel).
        """
        if self.is_recording:
            return True

        ffmpeg_exe = _resolve_ffmpeg_executable()
        if ffmpeg_exe is None:
            logger.info("ffmpeg not found (install imageio-ffmpeg or system ffmpeg); recording disabled.")
            return False

        self._width = int(width)
        self._height = int(height)
        if self._width <= 0 or self._height <= 0:
            logger.warning("Recording requires positive frame dimensions, got %dx%d.", self._width, self._height)
            return False
        fps = max(1.0, float(fps))

        if output_path is None:
            output_path = self.suggested_output_path()
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = self._pick_encoder(ffmpeg_exe)
        quality = float(np.clip(self._quality, 0.0, 100.0))
        # Keyframe interval of ~1 s. Forcing IDR at frame 0 and a short GOP
        # makes lightweight players and inline previews start decoding
        # immediately instead of showing a placeholder until the first
        # scene-cut keyframe lands.
        keyint = max(1, int(round(fps)))
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            f"{fps:.3f}",
            "-i",
            "-",
            # Silent AAC audio track — some platforms only inline-preview
            # MP4s that carry an audio stream.
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:a",
            "aac",
            "-b:a",
            "16k",
            "-c:v",
            codec,
        ]
        if codec == "h264_nvenc":
            # Lower CQ means higher quality; map quality in [0,100] -> CQ in [34,14].
            cq = int(round(34.0 - (quality / 100.0) * 20.0))
            cmd += [
                "-preset",
                "p7",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(max(14, min(34, cq))),
                "-b:v",
                "0",
                "-forced-idr",
                "1",
                "-g",
                str(keyint),
                "-keyint_min",
                str(keyint),
            ]
        elif codec in {"h264_qsv", "h264_amf"}:
            # Map quality in [0,100] to a practical bitrate range.
            bitrate_mbps = 4.0 + (quality / 100.0) * 76.0
            maxrate_mbps = bitrate_mbps * 2.0
            bufsize_mbps = bitrate_mbps * 3.0
            cmd += [
                "-b:v",
                f"{bitrate_mbps:.0f}M",
                "-maxrate",
                f"{maxrate_mbps:.0f}M",
                "-bufsize",
                f"{bufsize_mbps:.0f}M",
                "-g",
                str(keyint),
            ]
        elif codec == "libx264":
            # Lower CRF means higher quality; map quality in [0,100] -> CRF in [35,14].
            crf = int(round(35.0 - (quality / 100.0) * 21.0))
            cmd += [
                "-preset",
                "slow",
                "-crf",
                str(max(14, min(35, crf))),
                "-x264-params",
                f"keyint={keyint}:min-keyint={keyint}:scenecut=0",
                "-force_key_frames",
                "expr:eq(n,0)",
            ]
        if flip_vertical:
            cmd += ["-vf", "vflip"]
        # Broad-compatibility output: H.264 High profile, level 4.2,
        # yuv420p, faststart so the moov atom is at the front of the file.
        cmd += [
            "-profile:v",
            "high",
            "-level",
            "4.2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:
            logger.exception("Failed to launch ffmpeg for recording.")
            self._proc = None
            return False

        self._stderr_lines = []
        self._dropped_frames = 0
        self._last_drop_warn_t = 0.0
        self._queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._worker = threading.Thread(target=self._writer_loop, name="newton-mp4-writer", daemon=True)
        self._worker.start()
        self._stderr_reader = threading.Thread(target=self._stderr_reader_loop, name="newton-mp4-stderr", daemon=True)
        self._stderr_reader.start()
        self._output_path = output_path
        logger.info("Started recording to %s using encoder %s.", output_path, codec)
        return True

    def write_frame_bytes(self, frame_bytes: bytes, width: int, height: int):
        """Queue an already-encoded raw ``rgb24`` byte buffer.

        Use this when emitting duplicate frames so the caller doesn't pay
        the cost of re-serializing the same numpy array on every duplicate.
        Dimensions must match the values passed to :meth:`start`.
        """
        if not self.is_recording or self._queue is None:
            return
        if int(width) != self._width or int(height) != self._height:
            return
        expected = self._width * self._height * 3
        if len(frame_bytes) != expected:
            return
        try:
            self._queue.put_nowait(frame_bytes)
        except queue.Full:
            # Keep renderer real-time by dropping frames under sustained pressure.
            self._note_dropped_frame()

    def write_frame(self, frame_rgb: np.ndarray):
        """Queue a frame for encoding. Frame must be HxWx3 uint8."""
        if not self.is_recording or self._queue is None:
            return
        frame = np.asarray(frame_rgb)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            return
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            self._queue.put_nowait(frame.tobytes())
        except queue.Full:
            # Keep renderer real-time by dropping frames under sustained pressure.
            self._note_dropped_frame()

    def stop(self) -> Path | None:
        """Stop recording and flush the output file."""
        if not self.is_recording:
            return self._output_path
        assert self._proc is not None

        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                # Drain one slot to make room for the sentinel.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

        try:
            self._proc.wait(timeout=10.0)
        except Exception:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5.0)
            except Exception:
                pass

        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=2.0)
            self._stderr_reader = None
        try:
            if self._proc.stderr is not None:
                self._proc.stderr.close()
        except Exception:
            pass

        returncode = self._proc.returncode
        if returncode is not None and returncode != 0:
            tail = self._format_stderr_tail()
            if tail:
                logger.warning(
                    "ffmpeg exited with code %d while finalizing %s. ffmpeg stderr tail:\n%s",
                    returncode,
                    self._output_path,
                    tail,
                )
            else:
                logger.warning(
                    "ffmpeg exited with code %d while finalizing %s.",
                    returncode,
                    self._output_path,
                )

        if self._dropped_frames:
            logger.info(
                "MP4 recording dropped %d frame(s) total due to encoder backpressure.",
                self._dropped_frames,
            )

        stopped_path = self._output_path
        logger.info("Stopped recording: %s", stopped_path)
        self._proc = None
        self._queue = None
        self._stderr_lines = []
        self._dropped_frames = 0
        self._last_drop_warn_t = 0.0
        return stopped_path

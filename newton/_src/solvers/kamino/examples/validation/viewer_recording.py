# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Ad-hoc viewer recording helper for Kamino validation examples.

This monkey-patches a fully-initialized Newton viewer (as returned by
``newton.examples.init(parser)``) so the user can record per-frame PNGs
and stitch them into an MP4 from inside the running viewer session. It is
intentionally a temporary solution that lives alongside the validation
examples; once Newton's viewer gains native recording support, this file
should be deleted and the call sites switched over.

After ``enable_recording`` the viewer is wired up but inactive: nothing is
written to disk until ``viewer.start_clip(...)`` is called. Each call to
``start_clip`` clears the target folder, resets counters, and records up
to ``max_frames`` frames before automatically writing the MP4.

Usage (multi-clip mode, one MP4 per backend triggered from a GUI button):

    viewer, args = newton.examples.init(parser)
    viewer = enable_recording(viewer, video_folder=base_dir)
    ...
    # inside the GUI Reset-button handler:
    viewer.start_clip(
        output_path=os.path.join(base_dir, f"recording_{solver_name}.mp4"),
        max_frames=num_frames,
        video_folder=os.path.join(base_dir, f"frames_{solver_name}"),
        fps=viewer_fps,
        keep_frames=False,
    )

Exactly one PNG is captured per ``viewer.should_step()`` call that returns
True (i.e. per ``simulate()`` step). No PNGs are produced while paused or
when the render loop runs faster than the simulation.
"""

import glob
import os
import threading
from collections.abc import Callable
from types import MethodType

from newton._src.solvers.kamino._src.utils import logger as msg


def enable_recording(
    viewer,
    record_video: bool = True,
    video_folder: str = "./frames",
    skip_img_idx: int = 0,
    async_save: bool = False,
):
    """Monkey-patch ``viewer`` with PNG recording and MP4 generation.

    Wires the recording machinery onto ``viewer`` but does **not** start
    recording. Call ``viewer.start_clip(...)`` to actually capture frames.
    No directories are created until then.

    Args:
        viewer: A Newton viewer instance (typically a ``ViewerGL`` returned by
            ``newton.examples.init``). Must expose ``get_frame()`` and a
            ``renderer`` with ``_screen_width`` / ``_screen_height``.
        record_video: If False, this is a no-op.
        video_folder: Default output directory used by clips that do not
            pass an explicit ``video_folder`` to ``start_clip``. Not created
            on disk until a clip starts.
        skip_img_idx: Number of leading captured frames to drop before saving.
        async_save: If True, save each PNG on a background thread.

    Returns:
        The same ``viewer`` instance (mutated in place) to allow chaining.
    """
    if not record_video:
        return viewer

    if not hasattr(viewer, "get_frame"):
        msg.warning(f"enable_recording: viewer {type(viewer).__name__} has no get_frame(); recording disabled.")
        return viewer

    if getattr(viewer, "_record_video", False):
        msg.warning("enable_recording: viewer already has recording enabled; skipping re-config.")
        return viewer

    viewer._record_video = True
    viewer._video_folder = video_folder
    viewer._async_save = async_save
    viewer._skip_img_idx = skip_img_idx
    viewer._img_idx = 0
    viewer._frame_buffer = None
    # Recording is inactive until start_clip() is called; this keeps the
    # output folder empty during plain viewer sessions and avoids capturing
    # frames before the user explicitly asks for it.
    viewer._recording_active = False
    viewer._recording_stopped = False
    viewer._capture_pending = False
    viewer._created_video_folder = False
    viewer._save_threads: list[threading.Thread] = []

    viewer._clip_max_frames: int | None = None
    viewer._clip_output_path: str | None = None
    viewer._clip_keep_frames: bool = True
    viewer._clip_on_done: Callable[[], None] | None = None
    viewer._clip_fps: int = 60

    viewer._original_should_step = viewer.should_step
    viewer._original_end_frame = viewer.end_frame
    viewer.should_step = MethodType(_should_step_with_record, viewer)
    viewer.end_frame = MethodType(_end_frame_with_record, viewer)
    viewer._capture_frame = MethodType(_capture_frame, viewer)
    viewer._finish_clip = MethodType(_finish_clip, viewer)
    viewer.generate_video = MethodType(_generate_video, viewer)
    viewer.reset_recording = MethodType(_reset_recording, viewer)
    viewer.start_clip = MethodType(_start_clip, viewer)

    return viewer


def _clear_pngs(folder: str) -> int:
    """Delete all ``*.png`` files in ``folder`` and return the count removed."""
    if not os.path.isdir(folder):
        return 0
    files = glob.glob(os.path.join(folder, "*.png"))
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
    return len(files)


def _should_step_with_record(self):
    do_step = self._original_should_step()
    if do_step:
        self._capture_pending = True
    return do_step


def _end_frame_with_record(self):
    self._original_end_frame()

    if not (self._record_video and self._recording_active and not self._recording_stopped):
        # Even when not actively recording, clear any stale pending flag so we
        # do not capture a leftover frame on the next start_clip.
        self._capture_pending = False
        return

    if not self._capture_pending:
        return

    self._capture_pending = False
    self._capture_frame()

    if self._clip_max_frames is not None:
        captured = self._img_idx - self._skip_img_idx
        if captured >= self._clip_max_frames:
            self._finish_clip()


def _finish_clip(self):
    """Stop the current clip, flush async saves, and write the MP4."""
    captured = self._img_idx - self._skip_img_idx
    self._recording_active = False
    self._recording_stopped = True
    msg.notif(f"Clip captured: {captured} frames")

    for t in self._save_threads:
        t.join()
    self._save_threads.clear()

    out_path = self._clip_output_path
    self.generate_video(
        output_filename=out_path,
        fps=self._clip_fps,
        keep_frames=self._clip_keep_frames,
    )
    msg.notif(f"Video saved: {out_path}")

    on_done = self._clip_on_done
    self._clip_max_frames = None
    self._clip_output_path = None
    self._clip_on_done = None

    if on_done is not None:
        on_done()


def _reset_recording(self, video_folder: str | None = None) -> None:
    """Clear recorded PNGs and reset counters. Optionally switch folder.

    Joins any pending async-save threads so the folder is safe to clear.

    Args:
        video_folder: If given and different from the current one, switches
            the active output directory.
    """
    for t in self._save_threads:
        t.join()
    self._save_threads.clear()

    if video_folder is not None and video_folder != self._video_folder:
        self._video_folder = video_folder
        self._created_video_folder = not os.path.exists(video_folder)
    elif not os.path.exists(self._video_folder):
        self._created_video_folder = True
    os.makedirs(self._video_folder, exist_ok=True)

    removed = _clear_pngs(self._video_folder)
    if removed:
        msg.info(f"reset_recording: cleared {removed} PNG frames in {self._video_folder}")

    self._img_idx = 0
    self._recording_stopped = False
    self._capture_pending = False
    self._frame_buffer = None


def _start_clip(
    self,
    output_path: str,
    max_frames: int,
    video_folder: str | None = None,
    fps: int = 60,
    keep_frames: bool = True,
    on_done: Callable[[], None] | None = None,
) -> None:
    """Begin a finite-length recording clip with automatic MP4 generation.

    Clears any existing PNGs in the output folder, resets the frame counter,
    and arms an auto-stop trigger at ``max_frames`` captured frames. When the
    target is hit, the MP4 is written, frames are optionally deleted, and
    ``on_done`` (if provided) is called.

    Args:
        output_path: Destination MP4 path.
        max_frames: Number of frames to capture before auto-stopping.
        video_folder: Optional override for the PNG output directory.
        fps: Frame rate to embed in the MP4 (should match ``viewer_fps``).
        keep_frames: If False, delete the per-frame PNGs after the MP4 is
            written.
        on_done: Optional callback invoked once the MP4 has been written.
    """
    self.reset_recording(video_folder=video_folder)
    self._clip_max_frames = max_frames
    self._clip_output_path = output_path
    self._clip_keep_frames = keep_frames
    self._clip_on_done = on_done
    self._clip_fps = fps
    self._recording_active = True
    msg.notif(f"Recording started: {output_path} ({max_frames} frames -> {self._video_folder})")


def _capture_frame(self) -> bool:
    """Save the latest rendered frame to ``self._video_folder`` as a PNG."""
    try:
        from PIL import Image
    except ImportError:
        msg.warning("PIL not installed. Frames cannot be saved as images.")
        msg.info("Install with: pip install pillow")
        return False

    if self._img_idx >= self._skip_img_idx:
        frame = self.get_frame(target_image=self._frame_buffer)
        if self._frame_buffer is None:
            self._frame_buffer = frame

        frame_np = frame.numpy()
        image = Image.fromarray(frame_np, mode="RGB")

        filename = os.path.join(self._video_folder, f"{self._img_idx - self._skip_img_idx:05d}.png")

        if self._async_save:
            t = threading.Thread(
                target=image.save,
                args=(filename,),
                daemon=False,
            )
            t.start()
            self._save_threads.append(t)
            self._save_threads = [s for s in self._save_threads if s.is_alive()]
        else:
            image.save(filename)

    self._img_idx += 1
    return True


def _generate_video(
    self,
    output_filename: str = "recording.mp4",
    fps: int = 60,
    keep_frames: bool = True,
) -> bool:
    """Stitch the recorded PNGs in ``self._video_folder`` into an MP4."""
    try:
        import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
    except ImportError:
        msg.warning("imageio-ffmpeg not installed. Frames saved but video not generated.")
        msg.info("Install with: pip install imageio-ffmpeg")
        return False
    try:
        from PIL import Image
    except ImportError:
        msg.warning("PIL not installed. Frames saved but video not generated.")
        msg.info("Install with: pip install pillow")
        return False
    import numpy as np  # noqa: PLC0415

    if not self._record_video or self._img_idx <= self._skip_img_idx:
        msg.warning("No frames recorded, cannot generate video")
        return False

    for t in self._save_threads:
        t.join()
    self._save_threads.clear()

    frame_files = sorted(glob.glob(os.path.join(self._video_folder, "*.png")))
    if not frame_files:
        msg.warning(f"No png frames found in {self._video_folder}")
        return False

    msg.info(f"Generating video from {len(frame_files)} frames...")
    try:
        writer = ffmpeg.write_frames(
            output_filename,
            size=(self.renderer._screen_width, self.renderer._screen_height),
            fps=fps,
            codec="libx264",
            macro_block_size=8,
            quality=5,
        )
        writer.send(None)

        for frame_path in frame_files:
            img = Image.open(frame_path)
            frame_array = np.array(img)
            writer.send(frame_array)

        writer.close()
        msg.info(f"Video generated successfully: {output_filename}")

        if not keep_frames:
            msg.info("Deleting png frames...")
            for frame_path in frame_files:
                os.remove(frame_path)
            if self._created_video_folder:
                try:
                    os.rmdir(self._video_folder)
                except OSError:
                    pass
            msg.info("Frames deleted")

        return True

    except Exception as e:
        msg.warning(f"Failed to generate video: {e}")
        return False

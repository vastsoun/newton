# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Ad-hoc viewer recording helper for Kamino validation examples.

This monkey-patches a fully-initialized Newton viewer (as returned by
``newton.examples.init(parser)``) to dump per-frame PNGs to disk and to
optionally stitch them into an MP4. It is intentionally a temporary
solution that lives alongside the validation examples; once Newton's
viewer gains native recording support, this file should be deleted and
the call sites switched over.

Usage:

    viewer, args = newton.examples.init(parser)
    viewer = enable_recording(
        viewer,
        record_video=True,
        video_folder="./frames",
        async_save=False,
    )
    ...
    newton.examples.run(example, args)
    viewer.generate_video(output_filename="out.mp4", fps=50, keep_frames=False)
"""

import glob
import os
import threading
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

    Returns the same ``viewer`` instance (mutated in place) to allow chaining.

    Args:
        viewer: A Newton viewer instance (typically a ``ViewerGL`` returned by
            ``newton.examples.init``). Must expose ``get_frame()`` and a
            ``renderer`` with ``_screen_width`` / ``_screen_height`` to record.
        record_video: If False, this is a no-op.
        video_folder: Output directory for PNG frames.
        skip_img_idx: Number of leading frames to skip before saving.
        async_save: If True, save each PNG on a background thread.
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
    viewer._recording_stopped = False
    viewer._created_video_folder = not os.path.exists(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    viewer._original_end_frame = viewer.end_frame
    viewer.end_frame = MethodType(_end_frame_with_record, viewer)
    viewer._capture_frame = MethodType(_capture_frame, viewer)
    viewer.generate_video = MethodType(_generate_video, viewer)

    return viewer


def _end_frame_with_record(self):
    self._original_end_frame()
    if self._record_video and not self._recording_stopped:
        self._capture_frame()


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
            threading.Thread(
                target=image.save,
                args=(filename,),
                daemon=False,
            ).start()
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
                os.rmdir(self._video_folder)
            msg.info("Frames deleted")

        return True

    except Exception as e:
        msg.warning(f"Failed to generate video: {e}")
        return False

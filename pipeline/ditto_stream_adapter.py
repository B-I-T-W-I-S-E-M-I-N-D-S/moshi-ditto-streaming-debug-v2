"""
pipeline/ditto_stream_adapter.py
==================================
Wraps Ditto's ``StreamSDK`` (online mode from stream_pipeline_online.py)
to intercept rendered video frames instead of writing them to an MP4 file.

How it works
------------
1. On ``setup()``, the SDK's ``writer_worker`` function is monkey-patched
   **before** ``StreamSDK.setup()`` is called, so the SDK spawns our custom
   thread instead of the default disk-writer from the start.
   (FIX: previous code replaced the thread *after* setup(), causing a race
    where both the old thread and the new thread ran simultaneously.)

2. The patched writer pulls frames from Ditto's internal ``writer_queue``
   and pushes JPEG-encoded bytes into ``self._frame_queue`` with a
   non-blocking ``put_nowait`` — dropping frames only when the consumer
   (WebSocket sender) is too slow.
   (FIX: previous code used a blocking ``queue.put()`` which could deadlock
    the entire Ditto rendering pipeline when the browser disconnected.)

3. The caller pushes feature chunks via ``push_features(feat_np)``.
   This feeds directly into ``sdk.audio2motion_queue``.

4. ``iter_frames()`` is a blocking generator that yields (seq, jpeg_bytes)
   tuples.  Run it in a thread or via ``asyncio.to_thread()``.

JPEG Encoding Priority
----------------------
  1. libjpeg-turbo (via ``turbojpeg`` package) — fastest, ~0.5ms/frame
  2. OpenCV cv2.imencode            — fast, ~1ms/frame
  3. PIL Image.save                 — fallback, ~3-5ms/frame

Usage
-----
    adapter = DittoStreamAdapter(cfg_pkl=..., data_root=...)
    adapter.setup(image_path="/workspace/portrait.jpg")

    # Push feature chunks as they arrive (from bridge_task)
    adapter.push_features(seq=42, features=feat_chunk_np)  # (N, 1024) float32

    # Consume frames (run in asyncio.to_thread)
    for seq, jpeg_bytes in adapter.iter_frames():
        await websocket.send_bytes(b"\\x02" + seq_pack(seq) + jpeg_bytes)

    adapter.close()
"""

import io
import logging
import os
import queue
import sys
import threading
import time
from typing import Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure ditto-inference is importable
# ---------------------------------------------------------------------------
_DITTO_DIR = os.path.join(os.path.dirname(__file__), "..", "ditto-inference")
if _DITTO_DIR not in sys.path:
    sys.path.insert(0, _DITTO_DIR)

from stream_pipeline_online import StreamSDK  # online version (supports streaming)


# ---------------------------------------------------------------------------
# JPEG encoder — auto-selects fastest available backend
# ---------------------------------------------------------------------------

def _build_jpeg_encoder(quality: int):
    """
    Return a callable ``encode(rgb_uint8_hwc) -> bytes`` using the fastest
    available JPEG library on this system.

    Priority: turbojpeg > cv2 > PIL
    """
    # 1. libjpeg-turbo via pyturbojpeg
    try:
        from turbojpeg import TurboJPEG, TJPF_RGB
        _tj = TurboJPEG()
        def _turbo_encode(rgb: np.ndarray) -> bytes:
            return _tj.encode(rgb, quality=quality, pixel_format=TJPF_RGB)
        logger.info("[DittoStreamAdapter] JPEG encoder: TurboJPEG (fastest)")
        return _turbo_encode
    except ImportError:
        pass

    # 2. OpenCV
    try:
        import cv2
        _params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        def _cv2_encode(rgb: np.ndarray) -> bytes:
            bgr = rgb[:, :, ::-1]   # RGB → BGR for cv2
            ok, buf = cv2.imencode(".jpg", bgr, _params)
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return buf.tobytes()
        logger.info("[DittoStreamAdapter] JPEG encoder: OpenCV cv2 (fast)")
        return _cv2_encode
    except ImportError:
        pass

    # 3. PIL fallback
    from PIL import Image
    def _pil_encode(rgb: np.ndarray) -> bytes:
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    logger.info("[DittoStreamAdapter] JPEG encoder: PIL (fallback — consider installing turbojpeg or opencv-python)")
    return _pil_encode


class DittoStreamAdapter:
    """
    Wraps StreamSDK for real-time per-frame output.
    Synchronous version using setup_synchronous(), add_to_buffer(), and run_step().
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        jpeg_quality: int = 80,
    ):
        cfg_pkl   = os.path.abspath(cfg_pkl)
        data_root = os.path.abspath(data_root)

        if not os.path.isfile(cfg_pkl):
            raise FileNotFoundError(f"Ditto config .pkl not found: {cfg_pkl}")
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Ditto TRT model directory not found: {data_root}")

        logger.info(f"[DittoStreamAdapter] Loading StreamSDK from {data_root} …")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.jpeg_quality = jpeg_quality
        self._jpeg_encode = _build_jpeg_encoder(jpeg_quality)

        self._is_setup = False
        logger.info("[DittoStreamAdapter] StreamSDK loaded.")

    def setup(
        self,
        image_path: str,
        N_d: int = 10_000,
        emo: int = 4,
        sampling_timesteps: int = 50,
        overlap_v2: int = 10,
    ):
        image_path = os.path.abspath(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Portrait image not found: {image_path}")

        _DUMMY_OUT = "/tmp/ditto_stream_dummy.mp4"

        # Explicitly pass start_threads=False to rely purely on the synchronous sequential loop
        self.sdk.setup(
            source_path        = image_path,
            output_path        = _DUMMY_OUT,
            online_mode        = True,
            N_d                = N_d,
            emo                = emo,
            sampling_timesteps = sampling_timesteps,
            overlap_v2         = overlap_v2,
            start_threads      = False
        )
        
        # Initialize internal state arrays
        self.sdk.setup_synchronous()

        self._is_setup = True
        logger.info(f"[DittoStreamAdapter] Session ready for image: {image_path} (Synchronous Mode)")

    def add_to_buffer(self, features: np.ndarray):
        """
        Push a chunk of HuBERT-like features into Ditto's audio features buffer.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before add_to_buffer().")
        features = features.astype(np.float32)
        if features.ndim != 2 or features.shape[1] != 1024:
            raise ValueError(f"Expected (N, 1024) features, got {features.shape}")
        
        self.sdk.add_to_buffer(features)

    def is_ready(self) -> bool:
        """
        Checks if Ditto's internal buffer has enough valid features to run a full inference step.
        """
        if not self._is_setup:
            return False
        return self.sdk.is_ready()

    def run_step(self) -> list:
        """
        Runs one massive synchronous step of Ditto encoding->stitching->warp->decode->putback.
        Returns a list of JPEG encoded frames.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before run_step().")
        
        res_frames_rgb = self.sdk.run_step()
        
        # Encode res_frames_rgb to JPEG byte arrays immediately
        jpeg_frames = []
        for rgb_frame in res_frames_rgb:
            try:
                jpeg = self._jpeg_encode(rgb_frame)
                jpeg_frames.append(jpeg)
            except Exception as enc_err:
                logger.error(f"[DittoStreamAdapter] JPEG encode error: {enc_err}")
                
        return jpeg_frames

    def close(self):
        """
        Teardown. No threads to join in synchronous mode, just toggle state.
        """
        if getattr(self, '_is_setup', False):
            if getattr(self.sdk, 'writer', None):
                try: self.sdk.writer.close()
                except Exception: pass
            
        self._is_setup = False
        logger.info("[DittoStreamAdapter] Closed.")

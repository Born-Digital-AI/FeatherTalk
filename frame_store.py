"""Lossless native frame access without retaining entire decoded videos in RAM."""
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
import hashlib
import logging
import tempfile
import threading
import zlib

import cv2
import numpy as np


_SOURCE_FRAME_LOCK = threading.Lock()
logger = logging.getLogger("uvicorn.error")


def prepare_source_frames(source_path, frames, cache_dir):
    """Persist decoded source pixels once; speech never seeks or encodes MP4."""
    source_path, cache_dir = Path(source_path), Path(cache_dir)
    stamp = source_path.stat()
    identity = ("source-png-v1", str(source_path.resolve()), stamp.st_mtime_ns,
                stamp.st_size, len(frames), cv2.__version__)
    key = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
    directory = cache_dir / f"source-{key}"
    paths = tuple(directory / f"{index:06d}.png" for index in range(len(frames)))
    marker = directory / "complete"
    with _SOURCE_FRAME_LOCK:
        if marker.is_file() and marker.read_text() == str(len(frames)) and all(p.is_file() for p in paths):
            logger.info("Speech source disk cache hit: frames=%s", len(frames))
            return ImageFrames(paths)
        directory.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)
        logger.info("Speech source cache miss: extracting %s lossless frames once", len(frames))
        for index, path in enumerate(paths):
            # A partial extraction must never be published as a ready cache.
            with tempfile.NamedTemporaryFile(suffix=".png", dir=directory, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                if not cv2.imwrite(str(temporary_path), frames[index], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise RuntimeError(f"Cannot cache source frame {index} from {source_path}")
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
            if (index + 1) % 100 == 0:
                logger.info("Speech source caching: %s/%s frames", index + 1, len(frames))
        marker.write_text(str(len(frames)))
        return ImageFrames(paths)


class _BoundedFrames(Sequence):
    def __init__(self, cache_size):
        self._cache_size = max(1, cache_size)
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self._lock:
            if index not in self._cache:
                image = self._read(index)
                image.flags.writeable = False
                self._cache[index] = image
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(index)
            return self._cache[index]

    def close(self):
        self._cache.clear()


class ImageFrames(_BoundedFrames):
    def __init__(self, paths, cache_size=8):
        super().__init__(cache_size)
        self.paths = tuple(paths)

    def __len__(self):
        return len(self.paths)

    def _read(self, index):
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise RuntimeError(f"Failed to read image: {self.paths[index]}")
        return image


class VideoFrames(_BoundedFrames):
    def __init__(self, path, *, frame_count=None, cache_size=2):
        super().__init__(cache_size)
        self.path = Path(path)
        self._capture = None
        self._next_index = 0
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot read {self.path}. Upload a valid video clip.")
            self.fps = capture.get(cv2.CAP_PROP_FPS)
            self._length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if frame_count is None else frame_count
        finally:
            capture.release()
        if self._length < 2:
            raise ValueError(f"{self.path} needs at least two video frames.")

    def __len__(self):
        return self._length

    def _read(self, index):
        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self.path))
            self._next_index = 0
        if index != self._next_index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self._capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {index} from {self.path}")
        self._next_index = index + 1
        return image

    def close(self):
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            super().close()

    def __del__(self):
        if getattr(self, "_capture", None) is not None:
            self._capture.release()


class FrameSpool(_BoundedFrames):
    """Keep lossless generated frames on temporary disk, decoding only on demand."""
    def __init__(self, directory, shape, cache_size=2):
        super().__init__(cache_size)
        self.shape = tuple(shape)
        self._offsets = []
        self._file = tempfile.TemporaryFile(dir=directory)

    def __len__(self):
        return len(self._offsets)

    def append(self, image):
        if image.shape != self.shape or image.dtype != np.uint8:
            raise ValueError("Generated idle frames must have matching native uint8 dimensions.")
        compressed = zlib.compress(image.tobytes(), level=1)
        with self._lock:
            self._file.seek(0, 2)
            self._offsets.append((self._file.tell(), len(compressed)))
            self._file.write(compressed)

    def _read(self, index):
        offset, length = self._offsets[index]
        self._file.seek(offset)
        raw = zlib.decompress(self._file.read(length))
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.shape)

    def close(self):
        with self._lock:
            self._file.close()
            super().close()

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()

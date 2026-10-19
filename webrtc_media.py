"""Complete-frame media, compressed idle cache and server-owned animation phase."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import struct
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from frame_store import VideoFrames
from realtime_media import SpeechProducer

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class AccessUnit:
    data: bytes
    keyframe: bool


@dataclass(frozen=True)
class VideoSettings:
    width: int
    height: int
    fps: int = 25
    bitrate: int = 8_000_000
    gop: int = 25

    def __post_init__(self):
        if (not 160 <= self.width <= 4096 or not 64 <= self.height <= 4096
                or self.width % 2 or self.height % 2 or self.fps != 25
                or not 100_000 <= self.bitrate <= 100_000_000 or self.gop < 1):
            raise ValueError("WebRTC requires even native dimensions, 25 FPS and a positive supported bitrate.")


class NvEncoder:
    """One persistent NVENC context; never silently fall back to software encoding."""
    def __init__(self, settings: VideoSettings):
        import av
        self.av = av
        self.settings = settings
        encoder_logs = []
        try:
            logger.info("NVENC startup: creating context size=%sx%s", settings.width, settings.height)
            self.codec = av.CodecContext.create("h264_nvenc", "w")
            self.codec.width, self.codec.height = settings.width, settings.height
            self.codec.pix_fmt = "nv12"
            self.codec.time_base = Fraction(1, settings.fps)
            self.codec.framerate = Fraction(settings.fps, 1)
            self.codec.bit_rate = settings.bitrate
            self.codec.colorspace = 1  # BT.709, matched to the explicit RGB conversion.
            self.codec.color_primaries = self.codec.color_trc = 1
            self.codec.color_range = 1  # Limited range YUV, native mobile decoders.
            self.codec.options = dict(preset="p4", tune="ull", rc="cbr", zerolatency="1",
                                     bf="0", g=str(settings.gop), profile="baseline",
                                     **{"forced-idr": "1", "delay": "0", "rc-lookahead": "0"})
            # FFmpeg's NVENC loader logs the driver/library reason separately
            # from the generic error returned by avcodec_open2().
            with av.logging.Capture(local=True) as encoder_logs:
                logger.info("NVENC startup: opening encoder size=%sx%s", settings.width, settings.height)
                self.codec.open()
            logger.info("NVENC startup: encoder opened size=%sx%s", settings.width, settings.height)
        except Exception as exc:
            self.codec = None
            messages = dict.fromkeys(message.strip() for level, name, message in encoder_logs
                                     if level <= av.logging.ERROR and message.strip())
            details = "; ".join(messages)
            reason = f"{type(exc).__name__}: {exc}"
            if details:
                reason += f". FFmpeg: {details[:2000]}"
            raise RuntimeError(
                f"H.264 NVENC initialization failed ({settings.width}x{settings.height}, "
                f"{settings.fps} FPS): {reason}"
            ) from exc
        self.index = 0

    def encode(self, image, force_idr=False):
        frame = self.av.VideoFrame.from_ndarray(image, format="bgr24")
        frame = frame.reformat(self.settings.width, self.settings.height, format="nv12",
                               src_colorspace="ITU601", dst_colorspace="ITU709")
        frame.colorspace, frame.color_range = 1, 1
        frame.pts = self.index
        frame.time_base = Fraction(1, self.settings.fps)
        if force_idr:
            frame.pict_type = self.av.video.frame.PictureType.I
        packets = self.codec.encode(frame)
        self.index += 1
        if len(packets) != 1:
            raise RuntimeError("NVENC must produce exactly one access unit per frame in zero-latency mode.")
        packet = packets[0]
        if force_idr and not packet.is_keyframe:
            raise RuntimeError("NVENC did not honor the requested keyframe.")
        return AccessUnit(bytes(packet), bool(packet.is_keyframe))

    def close(self):
        self.codec = None


class EncodedIdle:
    """Disk-backed immutable access units; only an offset index stays in RAM."""
    MAGIC = b"FT264\x01"
    RECORD = struct.Struct(">IB")

    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.offsets = []
        try:
            if self.file.read(len(self.MAGIC)) != self.MAGIC:
                raise ValueError("Invalid H.264 idle cache")
            size = self.path.stat().st_size
            while self.file.tell() < size:
                record = self.file.read(self.RECORD.size)
                if len(record) != self.RECORD.size:
                    raise ValueError("Truncated idle cache")
                length, keyframe = self.RECORD.unpack(record)
                if not 0 < length <= 64 * 1024 * 1024 or keyframe not in (0, 1) or self.file.tell() + length > size:
                    raise ValueError("Invalid idle cache record")
                self.offsets.append((self.file.tell(), length, bool(keyframe)))
                self.file.seek(length, 1)
            if not self.offsets or not self.offsets[0][2]:
                raise ValueError("Idle cache must start with a keyframe")
        except BaseException:
            self.file.close()
            raise

    @classmethod
    def write(cls, path, units):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            count = 0
            with temporary.open("wb") as output:
                output.write(cls.MAGIC)
                for unit in units:
                    if count == 0 and not unit.keyframe:
                        raise ValueError("Idle cache must start with a keyframe")
                    if not 0 < len(unit.data) <= 64 * 1024 * 1024:
                        raise ValueError("Invalid idle cache access unit")
                    output.write(cls.RECORD.pack(len(unit.data), unit.keyframe))
                    output.write(unit.data)
                    count += 1
                if count == 0:
                    raise ValueError("Empty idle cache")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        offset, length, keyframe = self.offsets[index]
        self.file.seek(offset)
        return AccessUnit(self.file.read(length), keyframe)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepare_encoded_idle(source: Path, settings: VideoSettings, encoder_factory=NvEncoder):
    import av
    stamp = source.stat()
    identity = ("h264-idle-bt709-v1", source.name, stamp.st_size, stamp.st_mtime_ns,
                asdict(settings), av.__version__, str(av.library_versions))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    output = source.parent / f"idle-h264-{key}.ft264"
    if output.exists():
        logger.info("H.264 idle disk cache hit: %s", output.name)
        return EncodedIdle(output)
    logger.info("H.264 idle cache miss: opening source decoder %s", source.name)
    frames = VideoFrames(source)
    encoder = None
    try:
        encoder = encoder_factory(settings)
        logger.info("H.264 idle encoding started: frames=%s", len(frames))
        EncodedIdle.write(output, (encoder.encode(frames[i], i == 0) for i in range(len(frames))))
        logger.info("H.264 idle encoding finished: %s", output.name)
    finally:
        frames.close()
        if encoder:
            encoder.close()
    return EncodedIdle(output)


class MediaTimeline:
    """Return encoded video + matched PCM. Phase advances only on real video frames."""
    SILENCE = bytes(1280)  # 40 ms mono PCM16 at 16 kHz.

    def __init__(self, cached, encode, decode, *, frame_count, lead_frames=12):
        self.cached, self.encode, self.decode = cached, encode, decode
        self.frame_count = frame_count
        self.cycle = 2 * frame_count - 2
        if self.cycle < 2 or len(cached) != self.cycle:
            raise ValueError("Encoded idle frame count must match the avatar ping-pong cycle")
        self.phase = 0
        self.lead_frames = lead_frames
        self.reservation = None
        self.mode = "idle"
        self.generation = 0
        self.force_idr = False

    @property
    def idle(self):
        return self.mode == "idle" and self.reservation is None

    def reserve(self, generation):
        self.reservation = (self.phase + self.lead_frames) % self.cycle
        self.generation = generation
        return self.reservation

    def request_keyframe(self):
        self.force_idr = True

    def next(self, queue: deque, generation):
        if generation != self.generation:
            queue.clear()
            self.generation = generation
            self.reservation = None
            self.mode = "bridge"
            self.force_idr = True
        while queue and queue[0]["generation"] != generation:
            queue.popleft()
        if self.reservation == self.phase and self.mode != "speech":
            if len(queue) < self.lead_frames and not any(p["utterance_end"] for p in queue):
                return None, self.SILENCE
            self.mode = "speech"
            self.reservation = None
            self.force_idr = True
        if self.mode == "speech":
            if not queue:
                return None, self.SILENCE
            packet = queue.popleft()
            phase = packet["frame_index"] if packet["direction"] > 0 else (self.cycle - packet["frame_index"]) % self.cycle
            if phase != self.phase:
                raise ValueError("Speech frame does not match backend phase")
            unit = self.encode(packet["image"], self.force_idr)
            audio = packet["audio"]
            if packet["utterance_end"]:
                self.mode = "bridge"
        else:
            cached = self.cached[self.phase]
            if not self.force_idr and (self.mode == "idle" or cached.keyframe):
                unit = cached
                self.mode = "idle"
            else:
                unit = self.encode(self.decode(self.phase), self.force_idr)
                self.mode = "bridge"
            audio = self.SILENCE
        self.force_idr = False
        self.phase = (self.phase + 1) % self.cycle
        return unit, audio


class MediaSession:
    """Own rendering, bounded delivery and transport for one live avatar."""
    def __init__(self, stream, settings, transport, cache, encoder):
        self.stream, self.transport, self.cache, self.encoder = stream, transport, cache, encoder
        self.frames = VideoFrames(stream.idle_path)
        self.timeline = MediaTimeline(cache, encoder.encode, self.frames.__getitem__, frame_count=stream.silence_frame_count)
        self.queue = deque()
        self.tasks = []
        self.producer = SpeechProducer(stream, capacity=24, select_cursor=self.reserve,
                                       render_packets=stream.next_video_packets)
        self.connected = asyncio.Event()
        self.closed = False
        self.job = None

    async def in_thread(self, function, *args):
        self.job = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(self.job)
        finally:
            if not self.job.done():
                await self.job
            self.job = None

    async def reserve(self, generation):
        while not self.timeline.idle:
            if self.closed or generation != self.stream.audio_generation:
                return None
            await asyncio.sleep(0.01)
        phase = self.timeline.reserve(generation)
        cycle = self.timeline.cycle
        return dict(frame_index=phase if phase < self.stream.silence_frame_count else cycle - phase,
                    direction=1 if phase < self.stream.silence_frame_count - 1 else -1,
                    playback_start=0)

    async def start(self):
        self.tasks = [asyncio.create_task(self.deliver())]
        self.producer.start()

    async def deliver(self):
        await asyncio.wait_for(self.connected.wait(), 60)
        self.transport.begin()
        clock = time.monotonic()
        tick = 0
        while not self.closed:
            if self.producer.error:
                raise self.producer.error
            while len(self.queue) < 24 and not self.producer.queue.empty():
                packet = self.producer.queue.get_nowait()
                if packet["generation"] == self.stream.audio_generation:
                    self.queue.append(packet)
            if self.transport.keyframe_requested.is_set():
                self.transport.keyframe_requested.clear()
                self.timeline.request_keyframe()
            started = time.monotonic()
            phase = self.timeline.phase
            unit, audio = await self.in_thread(self.timeline.next, self.queue, self.stream.audio_generation)
            # A cancellation can arrive while NVENC is in flight.
            if self.timeline.generation != self.stream.audio_generation:
                self.timeline.phase = phase
                continue
            generation = self.timeline.generation
            accepted = False
            deadline = time.monotonic() + 2
            while generation == self.stream.audio_generation and not self.closed:
                accepted = await self.in_thread(self.transport.push, unit, audio, tick, self.stream, generation)
                if accepted:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("WebRTC media delivery stalled")
                await asyncio.sleep(0.005)
            if not accepted:
                self.timeline.phase = phase
                continue
            tick += 1
            clock = max(clock + 0.04, time.monotonic())
            await asyncio.sleep(max(0, clock - time.monotonic()))
            if tick % 125 == 0:
                self.transport.notify(dict(type="metrics", **self.stream.timings,
                                           media_mode=self.timeline.mode, queued_frames=len(self.queue),
                                           media_ms=(time.monotonic() - started) * 1000))

    async def close(self):
        self.closed = True
        # Unblock appsrc before waiting for a blocked push to drain.
        self.transport.close()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.producer.close()
        self.frames.close()
        self.encoder.close()
        self.cache.close()

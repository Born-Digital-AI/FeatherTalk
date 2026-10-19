"""Run inside the GPU image: validate real NVENC, decoding and GStreamer caps."""
import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=125)
    parser.add_argument("--bitrate", type=int, default=8_000_000)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    import av
    import numpy as np
    from webrtc_media import NvEncoder, VideoSettings
    from gst_webrtc import GstTransport

    settings = VideoSettings(args.width, args.height, bitrate=args.bitrate)
    encoder = NvEncoder(settings)
    transport = None
    decoded = 0
    units = []
    decoder = av.CodecContext.create("h264", "r")
    image = np.zeros((args.height, args.width, 3), np.uint8)
    started = time.perf_counter()
    try:
        for i in range(args.frames):
            image[:, :, 1] = (i * 3) % 256
            unit = encoder.encode(image, i in (0, args.frames // 2))
            units.append(unit)
            for frame in decoder.decode(av.Packet(unit.data)):
                if (frame.width, frame.height) != (args.width, args.height):
                    raise RuntimeError("NVENC changed native frame dimensions")
                decoded += 1
        for frame in decoder.decode(None):
            decoded += 1
        elapsed = time.perf_counter() - started
        if decoded != args.frames:
            raise RuntimeError(f"Expected {args.frames} decoded frames, got {decoded}")
        if not units[0].keyframe or not units[args.frames // 2].keyframe:
            raise RuntimeError("Forced keyframe recovery failed")
        events = []
        transport = GstTransport(settings, events.append)
        # ICE must remain valid after temporary Python wrappers are collected.
        gc.collect()
        transport.prime(units[0])
        transport.offer()
        deadline = time.monotonic() + 5
        while not any(event["type"] == "offer" for event in events):
            if any(event["type"] == "error" for event in events) or time.monotonic() > deadline:
                raise RuntimeError("GStreamer could not negotiate an H.264/Opus offer")
            time.sleep(0.01)
        offer = next(event["sdp"] for event in events if event["type"] == "offer")
        if "H264/90000" not in offer or "OPUS/48000" not in offer.upper():
            raise RuntimeError("WebRTC offer did not advertise H.264 and Opus")
        print(json.dumps(dict(nvenc=True, decoded_frames=decoded, encode_decode_fps=round(args.frames / elapsed, 1),
                              media_offer=True, excludes="FeatherTalk inference, real peer/ICE and mobile playback"), indent=2))
    finally:
        if transport:
            transport.close()
        encoder.close()


if __name__ == "__main__":
    main()

"""Prepare an avatar's disk cache for both lip-sync modes before the first user opens it."""
import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import AvatarStream, list_avatars, webrtc_video_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("avatar", help="Avatar directory name")
    args = parser.parse_args()
    if args.avatar not in list_avatars():
        parser.error(f"Unknown avatar: {args.avatar}")
    logging.basicConfig(level=logging.INFO)
    webrtc = os.environ.get("FEATHER_REALTIME_TRANSPORT", "webrtc") == "webrtc"
    for offline_quality in (True, False):
        stream = AvatarStream(args.avatar, offline_quality=offline_quality, cache_silence_frames=False)
        if stream.idle_path is None:
            raise RuntimeError("Inferred idle video was not prepared.")
        print(f"Idle cache ready (offline_quality={offline_quality}): {stream.idle_path}")
        if webrtc:
            from webrtc_media import prepare_encoded_idle
            prepare_encoded_idle(stream.idle_path, webrtc_video_settings(stream)).close()
            print(f"H.264 idle cache ready (offline_quality={offline_quality})")


if __name__ == "__main__":
    main()

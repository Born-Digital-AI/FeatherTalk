"""Prepare shared inferred idle media before the first user opens an avatar."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import AvatarStream, list_avatars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("avatar", help="Avatar directory name")
    args = parser.parse_args()
    if args.avatar not in list_avatars():
        parser.error(f"Unknown avatar: {args.avatar}")
    logging.basicConfig(level=logging.INFO)
    stream = AvatarStream(args.avatar, cache_silence_frames=False)
    print(f"Shared idle cache ready: {stream.idle_path}")


if __name__ == "__main__":
    main()

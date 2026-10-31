#!/usr/bin/env python3
"""Serve a song's sample folder to UltraBox, for songs converted with xm2box.py --rough.

Keep this running while the song is open. It only answers requests from this computer, and it
reports each sample it sends, so if nothing appears after importing the song, the browser isn't reaching it.

Usage: python serve_samples.py "output/song.samples" [--port 8765]
"""
import argparse
import functools
import http.server
import pathlib


class SampleHandler(http.server.SimpleHTTPRequestHandler):
    sent = set()
    total = 0

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")  # UltraBox runs on another site, so it needs permission
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def log_request(self, code="-", size="-"):
        name = self.path.lstrip("/")
        if code != 200:
            print(f"Couldn't send {name}: error {code}", flush=True)
            return
        self.sent.add(name)
        print(f"Sent {name} ({len(self.sent)} of {self.total})", flush=True)
        if len(self.sent) == self.total:
            print("All samples sent. UltraBox should now be playing the real sounds.", flush=True)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    folder = pathlib.Path(args.folder)
    samples = [f for f in folder.iterdir() if f.is_file()] if folder.is_dir() else []
    if not samples:
        raise SystemExit(f"No samples found in {folder}. Run this from the project folder, with the song's .samples folder.")
    SampleHandler.total = len(samples)
    handler = functools.partial(SampleHandler, directory=str(folder))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving {len(samples)} samples from {folder} at http://127.0.0.1:{args.port}/")
        print("Now import the song in UltraBox, and leave this running. Press Ctrl+C to stop.", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

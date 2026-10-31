#!/usr/bin/env python3
"""Serve a song's sample folder to UltraBox, for songs converted with xm2box.py --rough.

Keep this running while the song is open. It only answers requests from this computer.

Usage: python serve_samples.py "output/song.samples" [--port 8765]
"""
import argparse
import functools
import http.server


class SampleHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")  # UltraBox runs on another site, so it needs permission
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(SampleHandler, directory=args.folder)
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving {args.folder} at http://127.0.0.1:{args.port}/ - press Ctrl+C to stop")
        server.serve_forever()


if __name__ == "__main__":
    main()

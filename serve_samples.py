#!/usr/bin/env python3
"""Run a local copy of UltraBox, along with a song's samples, for songs converted with xm2box.py --rough.

Firefox stops websites from reaching servers on your computer unless you give permission. A page that
comes from this server doesn't count, so UltraBox runs from here and the samples load with no permission.

Keep this running while the song is open. It only answers requests from this computer, and it reports
each sample it sends. The first time UltraBox asks for each file, it's downloaded into the ultrabox folder.

Usage: python serve_samples.py "output/song.samples" [--port 8765]
"""
import argparse
import http.server
import pathlib
import urllib.error
import urllib.parse
import urllib.request

ULTRABOX_SITE = "https://ultraabox.github.io/"
ULTRABOX_LICENSE = "https://raw.githubusercontent.com/ultraabox/ultrabox_typescript/main/LICENSE.md"
ULTRABOX_FOLDER = pathlib.Path(__file__).parent / "ultrabox"


def download(url, target):
    """Save a file, and report whether it was there to save."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True


class Handler(http.server.SimpleHTTPRequestHandler):
    samples = None
    sent = set()
    total = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ULTRABOX_FOLDER), **kwargs)

    def request_name(self):
        return urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip("/")

    def is_sample(self):
        name = self.request_name()
        return bool(self.samples and name and "/" not in name and (self.samples / name).is_file())

    def translate_path(self, path):
        return str(self.samples / self.request_name()) if self.is_sample() else super().translate_path(path)

    def do_GET(self):
        name = self.request_name()
        target = ULTRABOX_FOLDER / (name or "index.html")
        if not self.is_sample() and ".." not in name.split("/") and not target.exists():
            download(ULTRABOX_SITE + urllib.parse.quote(name), target)
        super().do_GET()

    def log_request(self, code="-", size="-"):
        if not self.is_sample():
            return
        name = self.request_name()
        self.sent.add(name)
        print(f"Sent {name} ({len(self.sent)} of {self.total})", flush=True)
        if len(self.sent) == self.total:
            print("All samples sent. UltraBox should now be playing the real sounds.", flush=True)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", help="the song's .samples folder")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.folder:
        Handler.samples = pathlib.Path(args.folder)
        Handler.total = sum(f.is_file() for f in Handler.samples.iterdir()) if Handler.samples.is_dir() else 0
        if not Handler.total:
            raise SystemExit(f"No samples found in {args.folder}. Give it the song's .samples folder.")
    if not (ULTRABOX_FOLDER / "index.html").exists():
        print("Downloading UltraBox...", flush=True)
        names = ("index.html", "beepbox_editor.min.js")
        if not all([download(ULTRABOX_SITE + n, ULTRABOX_FOLDER / n) for n in names]):
            raise SystemExit("Couldn't download UltraBox. Check your internet connection.")
        download(ULTRABOX_LICENSE, ULTRABOX_FOLDER / "LICENSE.md")
    http.server.ThreadingHTTPServer.allow_reuse_address = False  # Windows would otherwise let two servers share the port
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        raise SystemExit(f"Port {args.port} is already in use. If an older copy of this server is running, stop it with Ctrl+C in its terminal.")
    with server:
        print(f"Open http://127.0.0.1:{args.port}/ in Firefox, then import the song there. Don't use the UltraBox website.")
        print("Leave this running while you work. Press Ctrl+C to stop.", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

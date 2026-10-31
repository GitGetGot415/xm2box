#!/usr/bin/env python3
"""Start a local copy of UltraBox in your browser, with the samples for every song in the output folder.

Usage: python start_ultrabox.py

Then use File > Import Song in the page that opens. This stops by itself when you close that page.

Firefox stops websites from reaching servers on your computer unless you give permission. A page that
comes from this server doesn't count, so songs made with xm2box.py --rough can load their samples here.
It only answers requests from this computer. The first time UltraBox asks for each of its files,
that file is downloaded into the ultrabox folder.
"""
import argparse
import http.server
import pathlib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

PROJECT = pathlib.Path(__file__).parent
SONGS_FOLDER = PROJECT / "output"
ULTRABOX_FOLDER = PROJECT / "ultrabox"
ULTRABOX_SITE = "https://ultraabox.github.io/"
ULTRABOX_LICENSE = "https://raw.githubusercontent.com/ultraabox/ultrabox_typescript/main/LICENSE.md"
SERVER_CHECK = "/is-this-start-ultrabox"
PAGE_OPEN = "/page-is-open"
PAGE_SCRIPT = f'<script>new EventSource("{PAGE_OPEN}");</script>'.encode()  # stays connected for as long as the page is open
CLOSE_WAIT_SECONDS = 5  # long enough for UltraBox to reload itself, which it does when importing a song with samples
DEFAULT_PORT = 8765


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


def find_sample(name):
    """Find a requested sample: "song.samples/Inst01a", or just "Inst01a" as in songs saved before samples had folders."""
    parts = name.split("/")
    if len(parts) == 2 and parts[0].endswith(".samples") and ".." not in parts:
        sample = SONGS_FOLDER / parts[0] / parts[1]
        return sample if sample.is_file() else None
    if len(parts) == 1 and parts[0]:
        return next((f for f in sorted(SONGS_FOLDER.glob(f"*.samples/{parts[0]}")) if f.is_file()), None)
    return None


class OpenPages:
    """Keeps track of how many pages are open, and when the last one closed."""
    lock = threading.Lock()
    count = 0
    closed_at = None  # None until a page has opened, and while one is open

    @classmethod
    def opened(cls):
        with cls.lock:
            cls.count += 1
            cls.closed_at = None

    @classmethod
    def closed(cls):
        with cls.lock:
            cls.count -= 1
            if not cls.count:
                cls.closed_at = time.monotonic()

    @classmethod
    def all_closed(cls):
        with cls.lock:
            return cls.closed_at is not None and time.monotonic() - cls.closed_at > CLOSE_WAIT_SECONDS


def stop_when_pages_close(server):
    while not OpenPages.all_closed():
        time.sleep(0.5)
    server.page_closed = True
    server.shutdown()


class Handler(http.server.SimpleHTTPRequestHandler):
    sent = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ULTRABOX_FOLDER), **kwargs)

    def request_name(self):
        return urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip("/")

    def translate_path(self, path):
        sample = find_sample(self.request_name())
        return str(sample) if sample else super().translate_path(path)

    def do_GET(self):
        if self.path == SERVER_CHECK:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == PAGE_OPEN:
            self.hold_page_open()
            return
        name = self.request_name()
        target = ULTRABOX_FOLDER / (name or "index.html")
        if not find_sample(name) and ".." not in name.split("/") and not target.exists():
            download(ULTRABOX_SITE + urllib.parse.quote(name), target)
        if name in ("", "index.html"):
            self.send_page(target)
        else:
            super().do_GET()

    def send_page(self, target):
        """Send UltraBox's page, with the script that tells this server when it closes."""
        page = target.read_bytes().replace(b"<head>", b"<head>" + PAGE_SCRIPT, 1)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def hold_page_open(self):
        """Keep a connection to the page, and notice it closing when a write fails."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        OpenPages.opened()
        try:
            while True:
                self.wfile.write(b": still here\n\n")
                self.wfile.flush()
                time.sleep(1)
        except OSError:
            pass
        finally:
            OpenPages.closed()

    def log_request(self, code="-", size="-"):
        sample = find_sample(self.request_name())
        if not sample:
            return
        song = sample.parent
        sent = self.sent.setdefault(song, set())
        sent.add(sample.name)
        total = sum(f.is_file() for f in song.iterdir())
        if len(sent) == total:
            print(f"{song.stem}: all {total} samples sent, so it should be playing its real sounds.", flush=True)
            sent.clear()  # so that this is reported once each time the song loads

    def log_message(self, *args):
        pass


def already_running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{SERVER_CHECK}", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser")
    args = parser.parse_args()
    address = f"http://127.0.0.1:{args.port}/"
    if already_running(args.port):
        print(f"UltraBox is already running, so this is just opening it: {address}")
        if not args.no_browser:
            webbrowser.open(address)
        return
    http.server.ThreadingHTTPServer.allow_reuse_address = False  # Windows would otherwise let two servers share the port
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        raise SystemExit(f"Port {args.port} is in use by something else. If an old serve_samples.py is running, stop it with Ctrl+C in its terminal.")
    if not (ULTRABOX_FOLDER / "index.html").exists():
        print("Downloading UltraBox...", flush=True)
        if not all([download(ULTRABOX_SITE + n, ULTRABOX_FOLDER / n) for n in ("index.html", "beepbox_editor.min.js")]):
            raise SystemExit("Couldn't download UltraBox. Check your internet connection.")
        download(ULTRABOX_LICENSE, ULTRABOX_FOLDER / "LICENSE.md")
    with server:
        songs = sorted(SONGS_FOLDER.glob("*.ultrabox.json"))
        print(f"UltraBox is running at {address}")
        print("Use File > Import Song there. Songs in the output folder:" if songs else "There are no converted songs in the output folder yet.")
        for song in songs:
            print(f"  {song.name}")
        print("This stops by itself when you close the page, or press Ctrl+C.", flush=True)
        threading.Thread(target=stop_when_pages_close, args=(server,), daemon=True).start()
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped. Export anything you want to keep before closing the page.")
        else:
            print("The UltraBox page was closed, so this has stopped.")


if __name__ == "__main__":
    main()

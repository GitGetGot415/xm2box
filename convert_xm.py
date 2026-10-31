#!/usr/bin/env python3
"""A small window for converting FastTracker II .xm files into UltraBox or regular BeepBox songs.

Usage: python convert_xm.py

Choose a file and a format and click Convert, and the song is saved in the output folder. For UltraBox, a song too
big to convert normally can be made as a rough version instead, and Open UltraBox starts start_ultrabox.py to import it.
BeepBox can't play samples, so a BeepBox song imitates them, and Open BeepBox opens the website to import it.
For a player that reads the song file directly, an UltraBox song can keep all of its samples in one file at full quality.
"""
import contextlib
import io
import os
import pathlib
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import xm2ultrabox

PROJECT = pathlib.Path(__file__).parent
XM_FOLDER = PROJECT / "xm"
OUTPUT_FOLDER = PROJECT / "output"
BEEPBOX_SITE = "https://www.beepbox.co/"
SONG_ENDINGS = {"ultrabox": ".ultrabox.json", "beepbox": ".beepbox.json"}
ROUGH_QUESTION = ("This song is too big to convert normally:\n\n{reason}\n\n"
                  "A rough version plays the song and can be edited, but it's less faithful. Notes keep one of a few "
                  "stereo positions, the quietest notes can be left out, and parts can move between channels.{opening}\n\nMake a rough version?")
ROUGH_OPENING = " It also has to be opened with the Open UltraBox button, not the UltraBox website."


def convert(path, output_folder, rough, one_file=False):
    """Convert a song, keeping the report that the converter prints. Returns the song's path and the report."""
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        song = xm2ultrabox.convert_at_best_bar_size(path, output_folder, rough=rough, full_samples=one_file,
                                                    sample_url=xm2ultrabox.DEFAULT_SAMPLE_URL if rough and not one_file else None,
                                                    require_url_fit=not rough and not one_file)
    return song, report.getvalue()


def convert_to_beepbox(path, output_folder):
    """Convert a song to regular BeepBox, keeping the report that the converter prints. Returns the song's path and the report."""
    import xm2beepbox  # only this format needs numpy
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        song = xm2beepbox.convert(path, output_folder)
    return song, report.getvalue()


def summarize(report):
    """Leave out the long list of what's on each channel."""
    return "\n".join(line for line in report.splitlines() if not line.startswith(("  Ch", "  Track", "Channel map"))).strip()


class App:
    def __init__(self, root, output_folder=OUTPUT_FOLDER):
        self.root = root
        self.output_folder = pathlib.Path(output_folder)
        self.results = queue.Queue()
        self.path = None
        self.busy = False
        self.format = tk.StringVar(value="ultrabox")
        self.one_file = tk.BooleanVar(value=False)
        root.title("XM Converter")
        root.geometry("640x460")
        root.minsize(480, 320)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        self.choose_button = ttk.Button(row, text="Choose .xm file...", command=self.choose)
        self.choose_button.pack(side="left")
        self.file_label = ttk.Label(row, text="No file chosen", padding=(10, 0))
        self.file_label.pack(side="left", fill="x", expand=True)
        row = ttk.Frame(frame, padding=(0, 10, 0, 0))
        row.pack(fill="x")
        ttk.Label(row, text="Convert to:").pack(side="left")
        self.format_buttons = [ttk.Radiobutton(row, text=text, value=value, variable=self.format, command=self.update_one_file_button)
                               for text, value in (("UltraBox", "ultrabox"), ("BeepBox", "beepbox"))]
        for button in self.format_buttons:
            button.pack(side="left", padx=(10, 0))
        self.one_file_button = ttk.Checkbutton(frame, variable=self.one_file, padding=(0, 10, 0, 0),
                                               text="Keep full-quality samples in one file") # for Greater Diplomacy 5 (may not open in UltraBox)
        self.one_file_button.pack(fill="x")
        row = ttk.Frame(frame, padding=(0, 10))
        row.pack(fill="x")
        self.convert_button = ttk.Button(row, text="Convert", command=self.start, state="disabled")
        self.convert_button.pack(side="left")
        self.progress = ttk.Progressbar(row, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.status = ttk.Label(frame, text="Choose a file to begin.", wraplength=600)
        self.status.pack(fill="x")
        holder = ttk.Frame(frame, padding=(0, 8))
        holder.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(holder)
        scrollbar.pack(side="right", fill="y")
        self.details = tk.Text(holder, height=10, wrap="word", state="disabled", yscrollcommand=scrollbar.set)
        self.details.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.details.yview)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Button(row, text="Open UltraBox", command=self.open_ultrabox).pack(side="left")
        ttk.Button(row, text="Open BeepBox", command=lambda: webbrowser.open(BEEPBOX_SITE)).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="Show output folder", command=self.show_folder).pack(side="left", padx=(8, 0))
        root.after(100, self.check_results)

    def choose(self):
        name = filedialog.askopenfilename(title="Choose a FastTracker II module", initialdir=XM_FOLDER, filetypes=[("FastTracker II modules", "*.xm"), ("All files", "*.*")])
        if name:
            self.set_file(name)

    def set_file(self, name):
        self.path = pathlib.Path(name)
        self.file_label.config(text=self.path.name)
        self.convert_button.config(state="normal")
        self.status.config(text="Ready to convert.")
        self.show_details("")

    def show_details(self, text):
        self.details.config(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.config(state="disabled")

    def set_busy(self, busy, message):
        self.busy = busy
        self.status.config(text=message)
        self.convert_button.config(state="disabled" if busy else "normal")
        self.choose_button.config(state="disabled" if busy else "normal")
        for button in self.format_buttons:
            button.config(state="disabled" if busy else "normal")
        self.update_one_file_button()
        if busy:
            self.progress.config(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.config(mode="determinate", value=0)  # otherwise it still looks like it's working

    def update_one_file_button(self):
        self.one_file_button.config(state="normal" if self.format.get() == "ultrabox" and not self.busy else "disabled")

    def start(self, rough=False):
        song_format = self.format.get()
        existing = self.output_folder / f"{self.path.stem}{SONG_ENDINGS[song_format]}"
        if not rough and existing.exists() and not self.confirm_overwrite(existing):
            return
        self.set_busy(True, f"Making a rough version of {self.path.name}. This can take a minute..." if rough else f"Converting {self.path.name}...")
        self.show_details("")
        threading.Thread(target=self.work, args=(self.path, rough, song_format, self.one_file.get()), daemon=True).start()

    def confirm_overwrite(self, existing):
        return messagebox.askyesno("Replace the existing song?", f"{existing.name} is already in the output folder.\n\nConvert again and replace it?")

    def work(self, path, rough, song_format, one_file):
        try:
            made = convert_to_beepbox(path, self.output_folder) if song_format == "beepbox" else convert(path, self.output_folder, rough, one_file)
            self.results.put(("done", *made, rough))
        except (xm2ultrabox.SamplesTooBig, xm2ultrabox.TooManyChannels) as problem:
            self.results.put(("too big", str(problem)))
        except ImportError as problem:
            self.results.put(("needs numpy", str(problem)))
        except (Exception, SystemExit) as problem:
            self.results.put(("failed", str(problem) or type(problem).__name__))

    def check_results(self):
        try:
            while True:
                self.handle(self.results.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self.check_results)

    def handle(self, result):
        kind, *rest = result
        if kind == "done":
            song, report, rough = rest
            self.set_busy(False, f"Saved {song.name} in the output folder." + (" This is a rough version." if rough else ""))
            self.show_details(summarize(report))
        elif kind == "too big":
            self.set_busy(False, "This song is too big to convert normally.")
            self.show_details(rest[0])
            if self.ask_rough(rest[0]):
                self.start(rough=True)
        elif kind == "needs numpy":
            self.set_busy(False, "BeepBox conversion needs numpy, which isn't installed.")
            self.show_details(f"Install it by running this in a terminal, then try again:\n\npython -m pip install numpy\n\n{rest[0]}")
        else:
            self.set_busy(False, "That file couldn't be converted.")
            self.show_details(f"{self.path.name} doesn't look like a working XM file, or something went wrong while reading it.\n\n{rest[0]}")

    def ask_rough(self, reason):
        opening = "" if self.one_file.get() else ROUGH_OPENING
        return messagebox.askyesno("Make a rough version?", ROUGH_QUESTION.format(reason=reason, opening=opening))

    def open_ultrabox(self):
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0  # its own window, which keeps running after this one closes
        subprocess.Popen([sys.executable, str(PROJECT / "start_ultrabox.py")], cwd=PROJECT, creationflags=flags)

    def show_folder(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(self.output_folder)
        else:
            subprocess.Popen(["xdg-open", str(self.output_folder)])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

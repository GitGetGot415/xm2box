#!/usr/bin/env python3
"""Convert a FastTracker II .xm module into UltraBox song JSON.

Carries over notes, timing, song order, volume, vibrato, pitch slides and the module's
own samples, which are stored inside the song file. The report written next to the song
lists what was left out.

Usage: python xm2box.py song.xm [-o output_dir] [--beats-per-bar 16]

For a song too big for that, --rough makes an approximate version: pan moves between a few fixed
positions, parts can share and swap channels, and the samples are written to a folder. To play
one of those, open UltraBox with start_ultrabox.py, and keep it running while the song is open.
"""
import argparse
import base64
import collections
import json
import math
import pathlib
import statistics
import struct
import urllib.parse

PARTS_PER_BEAT = 24
PARTS_PER_ROW = 6  # the most common speed becomes 4 rows per beat
MAX_PITCH = 96
MAX_INSTRUMENTS_PER_CHANNEL = 10
MAX_CHANNELS = 60
MAX_TEMPO = 500
SPARSE_CHANNEL_SHARE = 0.25  # channels used in fewer bars than this may merge into another
BPM_JITTER = 0.02  # tempo changes smaller than this are treated as no change
ON_GRID_SHARE = 0.97  # above this share of on-grid notes, use the 1/4 beat editor grid
SAMPLE_RATE = 8363  # FT2 plays a sample at this rate on C-4 with no tuning
NATIVE_PITCH = 48  # the BeepBox pitch that lines up with FT2's C-4
ROOT_KEY = 60  # UltraBox plays a sample at its own rate when the note matches this root key
INSTRUMENT_VOLUME = 7  # about twice UltraBox's default level, which keeps sampled songs from sounding too quiet
MAX_VOLUME = 25
NOTE_SIZE_MAX = 6
VOLUME_TOLERANCE = 0.25  # how far a note's simplified volume shape may stray, in note sizes
FADE_START = 32768  # FT2's fadeout counter, which falls by the instrument's fadeout value per tick
AUTO_VIBRATO_SEMITONES_PER_STEP = 1 / 64
AUTO_VIBRATO_TICKS_PER_CYCLE = 256
MAX_VIBRATO_DELAY = 49  # UltraBox treats 50 as "never"
VOLUME_PER_PITCH = 1 / 48 / 0.1428  # UltraBox's pitch damping of 48, over its volume scale of 0.1428
LOUDNESS_STEP = 6  # parts between loudness estimates
REFERENCE_FRAMES = 1600  # the sample length that needs no correction
LEVEL_TARGET = 0.97  # how loud a song's estimated mix should come out, once its master gain is squared
MAX_MASTER_GAIN = 2.5
LOOP_FORWARD, LOOP_PING_PONG, LOOP_NONE = 0, 1, 2  # UltraBox's chipWaveLoopMode values
SEMITONES_PER_PORTA_STEP = 1 / 16  # linear periods: 1xx, 2xx, 3xx and E1x move 4 of the 64 units in a semitone
SEMITONES_PER_EXTRA_FINE_STEP = 1 / 64
VIBRATO_RAW_PER_SEMITONE = 25
VIBRATO_DEPTH_RAW_PER_STEP = VIBRATO_RAW_PER_SEMITONE / 8  # FT2 peaks at y/8 semitones, UltraBox at raw/25
VIBRATO_RAW_SPEED_HZ = 0.1 / 0.14  # UltraBox's rate per raw speed: 0.1 of its 0.14 second vibrato period
VIBRATO_TICKS_PER_CYCLE = 64  # FT2's vibrato takes this many ticks at speed 1
SAMPLE_URL_BUDGET = 930_000  # Firefox rejects URLs past 1,048,576 characters, and this leaves room for notes and edits
SAMPLE_URL_OVERHEAD = 90
HALVING_LOSS_LIMIT = 0.02  # the most power a sample may lose from having its rate halved
HALF_BAND_FILTER = ((-3, -1), (-1, 9), (0, 16), (1, 9), (3, -1))
HALF_BAND_SCALE = 32
MAX_VIBRATO_DEPTH = 50
MAX_VIBRATO_SPEED = 30
MOD_SLOTS = 6
MOD_NO_CHANNEL = -2
MOD_PAN, MOD_VIBRATO_DEPTH, MOD_VIBRATO_SPEED = 6, 16, 18  # positions in UltraBox's list of modulators
MAX_PAN = 100
PAN_TOLERANCE = 3  # how far a simplified pan curve may stray, out of MAX_PAN
HANDLED_EFFECTS = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 27, 33}
HANDLED_EXTENDED = {1, 2, 6, 9, 10, 11, 12, 13, 14}
EFFECT_NAMES = {7: "tremolo", 16: "global volume", 17: "global volume slide", 20: "key off at tick",
                21: "envelope position", 25: "pan slide", 29: "tremor"}
EXTENDED_NAMES = {3: "glissando control", 4: "vibrato waveform",
                  5: "set finetune", 7: "tremolo waveform", 8: "set pan"}
TINY_WAVE_FRAMES = 200  # a loop this short is a waveform, so restarting or joining its notes can't be heard
ARPEGGIO_SPEEDS = [0, 0.0625, 0.125, 0.2, 0.25, 1 / 3, 0.4, 0.5, 2 / 3, 0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]  # UltraBox's table
ARPEGGIO_TICKS_PER_CYCLE = 3  # FT2 steps through its 3 pitches a tick at a time, and UltraBox's 2-note chord takes 2 steps
PAN_COLUMN_STEP = 17  # volume column Cx sets pan to x times this
PAN_SLIDE_LEFT, PAN_SLIDE_RIGHT, PAN_SET = 0xD, 0xE, 0xC
RETRIGGER_VOLUME = {1: -1, 2: -2, 3: -4, 4: -8, 5: -16, 9: 1, 10: 2, 11: 4, 12: 8, 13: 16}
RETRIGGER_SCALE = {6: 2 / 3, 7: 1 / 2, 14: 3 / 2, 15: 2}
ROUGH_PAN_POSITIONS = 5  # how many fixed places a note can sit in the stereo field in --rough mode
DEFAULT_SAMPLE_URL = "http://127.0.0.1:8765/"
BAR_SIZES = (16, 8, 4, 2)  # the longest bar that fits a song's channels is the easiest to work with


class TooManyChannels(Exception):
    pass


class SamplesTooBig(Exception):
    pass


def read_string(data):
    return data.split(b"\0")[0].decode("latin1").strip()


def parse_xm(path):
    d = path.read_bytes()
    if d[:17] != b"Extended Module: ":
        raise SystemExit(f"{path} is not an XM module")
    size, length, restart, channels, pattern_count, instrument_count, _, speed, bpm = struct.unpack_from("<IHHHHHHHH", d, 60)
    song = dict(title=read_string(d[17:37]), order=list(d[80:80 + length]), restart=restart,
                channels=channels, speed=speed, bpm=bpm, patterns=[], instruments={})
    p = 60 + size
    for _ in range(pattern_count):
        header, _, rows, packed = struct.unpack_from("<IBHH", d, p)
        q = p + header
        end = q + packed
        cells = []
        while q < end:
            b = d[q]
            q += 1
            cell = [0] * 5
            if b & 0x80:
                for i in range(5):
                    if b & (1 << i):
                        cell[i] = d[q]
                        q += 1
            else:
                cell[0] = b
                cell[1:] = d[q:q + 4]
                q += 4
            cells.append(cell)
        if not packed:
            cells = [[0] * 5 for _ in range(rows * channels)]
        song["patterns"].append((rows, cells))
        p = end
    for index in range(1, instrument_count + 1):
        size = struct.unpack_from("<I", d, p)[0]
        sample_count = struct.unpack_from("<H", d, p + 27)[0]
        instrument = dict(name=read_string(d[p + 4:p + 26]), samples=[], keymap=None, volume_envelope=None, pan_envelope=None, auto_vibrato=None)
        if sample_count:
            sample_header = struct.unpack_from("<I", d, p + 29)[0]
            instrument["keymap"] = d[p + 33:p + 129]
            point_count, pan_count, sustain, loop_start, loop_end, pan_sustain, pan_start, pan_end, kind, pan_kind, vibrato_kind, sweep, depth, rate = struct.unpack_from("<14B", d, p + 225)
            fadeout = struct.unpack_from("<H", d, p + 239)[0]

            def read_envelope(at, count, flags, hold, first, last):
                values = struct.unpack_from(f"<{count * 2}H", d, at)
                return dict(points=list(zip(values[::2], values[1::2])), sustain=hold if flags & 2 else None,
                            loop=(first, last) if flags & 4 else None, fadeout=fadeout)

            if kind & 1 and point_count:
                instrument["volume_envelope"] = read_envelope(p + 129, point_count, kind, sustain, loop_start, loop_end)
            if pan_kind & 1 and pan_count:
                instrument["pan_envelope"] = read_envelope(p + 177, pan_count, pan_kind, pan_sustain, pan_start, pan_end)
            if depth and rate:
                instrument["auto_vibrato"] = dict(kind=vibrato_kind, sweep=sweep, depth=depth, rate=rate)
            q = p + size
            data_at = q + sample_header * sample_count
            for _ in range(sample_count):
                length, loop_start, loop_length, volume, finetune, kind, pan, relative, _ = struct.unpack_from("<IIIBbBBbB", d, q)
                wide = bool(kind & 16)
                width = 2 if wide else 1
                loop = {1: LOOP_FORWARD, 2: LOOP_PING_PONG}.get(kind & 3, LOOP_NONE)
                if loop_length // width == 0:
                    loop = LOOP_NONE
                instrument["samples"].append(dict(frames=length // width, volume=min(volume, 64), finetune=finetune,
                                                  relative=relative, pan=pan, wide=wide, loop=loop,
                                                  loop_start=loop_start // width, loop_end=(loop_start + loop_length) // width,
                                                  raw=d[data_at:data_at + length]))
                data_at += length
                q += sample_header
            p = data_at
        else:
            p += size
        song["instruments"][index] = instrument
    return song


def decode_sample(sample):
    """Undo XM's delta coding. Returns audio bytes as a WAV file would hold them."""
    if sample["wide"]:
        count = len(sample["raw"]) // 2
        value = 0
        out = bytearray(count * 2)
        for i, delta in enumerate(struct.unpack(f"<{count}h", sample["raw"][:count * 2])):
            value = (value + delta + 32768) % 65536 - 32768
            struct.pack_into("<h", out, i * 2, value)
        return bytes(out)
    value = 0
    out = bytearray(len(sample["raw"]))
    for i, delta in enumerate(sample["raw"]):
        value = (value + delta) % 256
        out[i] = (value + 128) % 256  # WAV stores 8-bit audio as unsigned
    return bytes(out)


def sample_audio(sample):
    """The bytes a sample keeps in the song: FT2 never plays past a loop, and a halved sample has every other frame."""
    width = 2 if sample["wide"] else 1
    audio = decode_sample(sample)
    if sample["loop"] != LOOP_NONE:
        audio = audio[:sample["loop_end"] * width]  # FT2 never plays past the end of a loop
    return halve_rate(audio, sample["wide"]) if sample.get("halved") else audio


def as_values(audio, wide):
    if wide:
        return list(struct.unpack(f"<{len(audio) // 2}h", audio))
    return [b - 128 for b in audio]


def halve_rate(audio, wide):
    """Keep every other frame, after a half-band filter so that what's lost is removed and doesn't alias."""
    values = as_values(audio, wide)
    last = len(values) - 1
    kept = []
    for i in range(0, len(values), 2):
        total = sum(weight * values[max(0, min(last, i + offset))] for offset, weight in HALF_BAND_FILTER)
        kept.append(round(total / HALF_BAND_SCALE))
    if wide:
        return struct.pack(f"<{len(kept)}h", *(max(-32768, min(32767, v)) for v in kept))
    return bytes(max(0, min(255, v + 128)) for v in kept)


def upper_half_share(sample):
    """How much of a sample's power is in the upper half of its range, which halving its rate would lose."""
    values = as_values(sample_audio(sample), sample["wide"])
    diffs = [(a - b) ** 2 for a, b in zip(values, values[1:])]
    pairs = [(a + b) ** 2 for a, b in zip(values, values[1:])]
    return sum(diffs) / max(1, sum(diffs) + sum(pairs))


def fit_samples_in_url(song, shifts):
    """Halve the rate of the biggest samples that lose the least, until the song fits in a Firefox URL."""
    samples = {key: song["instruments"][key[0]]["samples"][key[1]] for key in shifts}
    url_size = lambda sample: len(sample_audio(sample)) * 4 // 3 + SAMPLE_URL_OVERHEAD
    sizes = {key: url_size(sample) for key, sample in samples.items()}
    total = sum(sizes.values())
    halved = []
    for key in sorted(samples, key=lambda k: -sizes[k]):
        if total <= SAMPLE_URL_BUDGET:
            break
        if upper_half_share(samples[key]) > HALVING_LOSS_LIMIT:
            continue
        samples[key]["halved"] = True
        halved.append(key)
        total += url_size(samples[key]) - sizes[key]
    return total, halved


def wav_file(sample, name):
    """Build a sample's WAV file.

    UltraBox names a sample after whatever follows the last "/" in its URL. So the
    WAV ends with an unused chunk whose base64 text spells "/<name>", which makes
    that name appear in UltraBox's wave menu. The name must be 3 less than a multiple
    of 4 long, so that the "/" and the name fill whole base64 groups.
    """
    width = 2 if sample["wide"] else 1
    audio = sample_audio(sample)
    if len(audio) % 2:
        audio += b"\0"
    label = base64.b64decode("/" + name)
    filler = 0
    while (12 + 16 + 8 + 8 + len(audio) + 8 + filler) % 3 or (filler + len(label)) % 2:
        filler += 1
    chunk = b"\0" * filler + label
    body = (b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * width, width, 8 * width)
            + b"data" + struct.pack("<I", len(audio)) + audio + b"JUNK" + struct.pack("<I", len(chunk)) + chunk)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def wav_data_url(sample, name, root_key):
    """Build an UltraBox custom sample URL holding the sample as an embedded WAV file."""
    encoded = base64.b64encode(wav_file(sample, name)).decode("ascii")
    return f"!s{SAMPLE_RATE},r{root_key:g}!data:audio/wav;base64,{encoded}"


class Voice:
    def __init__(self, instrument, pitch, start, end_limit):
        self.instrument = instrument  # (XM instrument, sample within it, start offset in frames), later with detune in cents
        self.pitch = pitch
        self.start = start
        self.end = end_limit
        self.points = []
        self.path = [(start, 0.0)]  # (time, semitones away from the note's pitch)
        self.vibrato = []  # (start, end, raw speed, raw depth)
        self.envelope = []  # (time, volume envelope and fadeout as a share of full volume)
        self.pan = []  # (time, UltraBox's raw pan)
        self.chord = ()  # semitones above the pitch that an arpeggio adds to the note

    def add_point(self, time, volume):
        if self.points and time <= self.points[-1][0]:
            self.points[-1] = (self.points[-1][0], volume)
        else:
            self.points.append((time, volume))

    def offset(self):
        return self.path[-1][1]

    def slide(self, start, end, change):
        """Move the pitch by this many semitones between two times."""
        if change:
            # A bend always ramps from its last point, so an instant jump needs a point just before it.
            self.path.append((start - 1 if start == end else start, self.offset()))
            self.path.append((end, self.offset() + change))


class Envelope:
    """A channel's place in an instrument's volume or pan envelope, stepped tick by tick as FT2 does."""

    def __init__(self, envelope):
        self.envelope = envelope
        self.tick = 0
        self.fade = FADE_START
        self.released = False

    def value(self):
        return _value_at(self.envelope["points"], self.tick)

    def volume_share(self):
        return self.value() / 64 * self.fade / FADE_START

    def pan_for(self, pan):
        """FT2 swings the sample's pan by the envelope, as far as the nearer edge of the stereo field allows."""
        return pan + (self.value() - 32) * (128 - abs(pan - 128)) / 32

    def advance(self):
        points, sustain, loop = (self.envelope[k] for k in ("points", "sustain", "loop"))
        if self.released:
            self.fade = max(0, self.fade - self.envelope["fadeout"])
        elif sustain is not None and sustain < len(points) and self.tick >= points[sustain][0]:
            return
        if self.tick < points[-1][0]:
            self.tick += 1
        # A loop that ends on the sustain point stops looping once the note is released.
        if (loop and loop[1] < len(points) and self.tick >= points[loop[1]][0]
                and not (self.released and loop[1] == sustain)):
            self.tick = points[loop[0]][0]


def most_common_speed_and_bpm(song):
    """Walk the song once for the speed and BPM it spends the most time at."""
    speed, bpm = song["speed"], song["bpm"]
    speeds, bpms = collections.Counter(), collections.Counter()
    for order in song["order"]:
        if order >= len(song["patterns"]):
            continue
        rows, cells = song["patterns"][order]
        for row in range(rows):
            for _, _, _, effect, param in cells[row * song["channels"]:(row + 1) * song["channels"]]:
                if effect == 15 and param:
                    if param < 32:
                        speed = param
                    else:
                        bpm = param
            speeds[speed] += 1
            bpms[bpm] += speed
    main_bpm = bpms.most_common(1)[0][0]
    # Songs can wander by a BPM or two, so use the average of the BPMs near the main one to stop the song drifting.
    close = {b: n for b, n in bpms.items() if abs(b - main_bpm) / main_bpm <= BPM_JITTER}
    return speeds.most_common(1)[0][0], sum(b * n for b, n in close.items()) / sum(close.values())


def simulate(song):
    """Play the module in order and return its notes, each with a time in BeepBox parts."""
    channels = song["channels"]
    base_speed, base_bpm = most_common_speed_and_bpm(song)
    scale = PARTS_PER_ROW / base_speed
    tempo = base_bpm * scale
    while tempo > MAX_TEMPO:  # very fast modules use half the tempo and half the parts per tick
        tempo /= 2
        scale /= 2
    notes = [[] for _ in range(channels)]
    voices = [None] * channels
    volumes = [0.0] * channels
    last_instrument = [0] * channels
    last_sample = [None] * channels
    slide_memory = [0] * channels
    offset_memory = [0] * channels
    fine_memory = [[0, 0] for _ in range(channels)]
    slide_up_memory = [0] * channels
    slide_down_memory = [0] * channels
    porta_memory = [0] * channels
    porta_target = [None] * channels
    fine_pitch_memory = [[0, 0] for _ in range(channels)]
    extra_fine_memory = [[0, 0] for _ in range(channels)]
    vibrato_memory = [[0, 0] for _ in range(channels)]
    envelopes = [None] * channels
    pan_envelopes = [None] * channels
    channel_pan = [128.0] * channels
    retrigger_memory = [0] * channels
    retrigger_count = [0] * channels
    stopped = False
    ignored = collections.Counter()
    order_starts = {}
    speed, bpm = song["speed"], song["bpm"]
    now = 0.0
    loop_start = None
    position, start_row = 0, 0
    visited = set()

    def part_ticks():
        if abs(bpm - base_bpm) / base_bpm <= BPM_JITTER:
            return scale
        return scale * base_bpm / bpm

    def finish(ch, time):
        voice = voices[ch]
        if voice:
            voice.end = min(voice.end, time)
            notes[ch].append(voice)
            voices[ch] = None

    while position < len(song["order"]):
        if (position, start_row) in visited:
            loop_start = order_starts.get(position)
            break
        visited.add((position, start_row))
        order_starts.setdefault(position, now)
        pattern = song["order"][position]
        rows, cells = song["patterns"][pattern] if pattern < len(song["patterns"]) else (64, [[0] * 5 for _ in range(64 * channels)])
        row = start_row
        start_row = 0
        jump, row_break = None, None
        loop_row = [0] * channels
        loop_left = [0] * channels
        while row < rows:
            cell_row = cells[row * channels:(row + 1) * channels]
            pattern_delay = 0
            for _, _, _, effect, param in cell_row:
                if effect == 15 and not param:
                    stopped = True  # FT2 stops the song at F00
                elif effect == 15:
                    if param < 32:
                        speed = param
                    else:
                        bpm = param
                elif effect == 11:
                    jump = param
                elif effect == 13:
                    row_break = (param >> 4) * 10 + (param & 15)
                elif effect == 14 and param >> 4 == 14:
                    pattern_delay = param & 15
            tick = part_ticks()
            jump_back = None
            for ch, (note, instrument_index, column, effect, param) in enumerate(cell_row):
                sub = param >> 4
                if effect or param:
                    if effect == 14:
                        if sub not in HANDLED_EXTENDED:
                            ignored[f"E{sub:X} {EXTENDED_NAMES.get(sub, '?')}"] += 1
                    elif effect not in HANDLED_EFFECTS:
                        ignored[f"{effect:X} {EFFECT_NAMES.get(effect, '?')}"] += 1
                if effect == 14 and sub == 6:
                    if param & 15 == 0:
                        loop_row[ch] = row
                    elif loop_left[ch] == 0:
                        loop_left[ch] = param & 15
                        jump_back = loop_row[ch]
                    else:
                        loop_left[ch] -= 1
                        if loop_left[ch]:
                            jump_back = loop_row[ch]
                delay = param & 15 if effect == 14 and sub == 13 else 0
                at = now + delay * tick
                if delay >= speed:
                    continue
                if effect == 9 and param:
                    offset_memory[ch] = param
                if instrument_index:
                    last_instrument[ch] = instrument_index
                    # FT2 restarts the envelope on an instrument number, and not on a note without one.
                    instrument = song["instruments"].get(instrument_index)
                    envelope = instrument["volume_envelope"] if instrument else None
                    envelopes[ch] = Envelope(envelope) if envelope else None
                    envelope = instrument["pan_envelope"] if instrument else None
                    pan_envelopes[ch] = Envelope(envelope) if envelope else None
                is_porta = effect in (3, 5) or column >> 4 == 0xF
                if note == 97:
                    if pan_envelopes[ch]:
                        pan_envelopes[ch].released = True
                    if envelopes[ch]:
                        envelopes[ch].released = True  # the note fades out instead of stopping
                    else:
                        finish(ch, at)
                elif 0 < note < 97 and is_porta and voices[ch] and last_sample[ch]:
                    porta_target[ch] = note + last_sample[ch]["relative"] - 1 - voices[ch].pitch
                elif 0 < note < 97:
                    finish(ch, at)
                    porta_target[ch] = None
                    instrument = song["instruments"].get(last_instrument[ch])
                    if instrument and instrument["samples"]:
                        sample_index = min(instrument["keymap"][note - 1], len(instrument["samples"]) - 1)
                        sample = instrument["samples"][sample_index]
                        last_sample[ch] = sample
                        offset = offset_memory[ch] * 256 if effect == 9 else 0
                        pitch = note + sample["relative"] - 1
                        limit = math.inf
                        if sample["loop"] == LOOP_NONE:
                            rate = SAMPLE_RATE * 2 ** ((note + sample["relative"] - 49) / 12)
                            limit = at + max(0, sample["frames"] - offset) / rate / (2.5 / bpm) * tick
                        if offset < sample["frames"]:
                            voices[ch] = Voice((last_instrument[ch], sample_index, offset), pitch, at, limit)
                            retrigger_count[ch] = 0
                if instrument_index and last_sample[ch]:
                    volumes[ch] = float(last_sample[ch]["volume"])
                    channel_pan[ch] = float(last_sample[ch]["pan"])
                if effect == 8:
                    channel_pan[ch] = float(param)
                elif column >> 4 == PAN_SET:
                    channel_pan[ch] = float((column & 15) * PAN_COLUMN_STEP)
                # A waveform's arpeggio is a chord note, so a note is cut where its arpeggio starts or stops.
                voice = voices[ch]
                if voice and last_sample[ch] and last_sample[ch]["frames"] < TINY_WAVE_FRAMES:
                    chord = tuple(i for i in dict.fromkeys((sub, param & 15)) if i) if effect == 0 else ()
                    if chord != voice.chord:
                        if chord:
                            song["instruments"][voice.instrument[0]]["chord_arpeggio"] = True
                        if voice.start < at:
                            joined = Voice(voice.instrument, voice.pitch, at, voice.end)
                            joined.path = [(at, voice.offset())]
                            finish(ch, at)
                            voices[ch] = voice = joined
                        voice.chord = chord
                before = volumes[ch]
                if 0x10 <= column <= 0x50:
                    volumes[ch] = float(column - 0x10)
                elif column >> 4 == 8:
                    volumes[ch] -= column & 15
                elif column >> 4 == 9:
                    volumes[ch] += column & 15
                if effect == 12:
                    volumes[ch] = float(min(param, 64))
                elif effect == 14 and sub in (10, 11):
                    memory = fine_memory[ch]
                    if param & 15:
                        memory[sub - 10] = param & 15
                    volumes[ch] += memory[sub - 10] * (1 if sub == 10 else -1)
                volumes[ch] = max(0.0, min(64.0, volumes[ch]))
                voice = voices[ch]
                if voice:
                    if voice.points and volumes[ch] != before:
                        voice.add_point(at - tick, before)  # hold, so the change is a step and not a ramp
                    voice.add_point(at, volumes[ch])
                slide = 0
                if effect in (5, 6, 10):
                    if param:
                        slide_memory[ch] = param
                    up, down = slide_memory[ch] >> 4, slide_memory[ch] & 15
                    slide = up if up else -down
                if column >> 4 == 6:
                    slide -= column & 15
                elif column >> 4 == 7:
                    slide += column & 15
                if slide:
                    volumes[ch] = max(0.0, min(64.0, volumes[ch] + slide * (speed - 1) * (1 + pattern_delay)))
                    if voice:
                        voice.add_point(now + speed * (1 + pattern_delay) * tick, volumes[ch])
                if voice:
                    moves = (speed - 1) * (1 + pattern_delay)
                    row_end = now + speed * (1 + pattern_delay) * tick
                    if effect == 14 and sub in (1, 2):
                        memory = fine_pitch_memory[ch]
                        if param & 15:
                            memory[sub - 1] = param & 15
                        voice.slide(at, at, memory[sub - 1] * SEMITONES_PER_PORTA_STEP * (1 if sub == 1 else -1))
                    elif effect == 33 and sub in (1, 2):
                        memory = extra_fine_memory[ch]
                        if param & 15:
                            memory[sub - 1] = param & 15
                        voice.slide(at, at, memory[sub - 1] * SEMITONES_PER_EXTRA_FINE_STEP * (1 if sub == 1 else -1))
                    if effect in (1, 2):
                        memory = slide_up_memory if effect == 1 else slide_down_memory
                        if param:
                            memory[ch] = param
                        voice.slide(now, row_end, memory[ch] * SEMITONES_PER_PORTA_STEP * moves * (1 if effect == 1 else -1))
                    if is_porta:
                        if effect == 3 and param:
                            porta_memory[ch] = param
                        elif column >> 4 == 0xF and column & 15:
                            porta_memory[ch] = (column & 15) * 16
                        target = porta_target[ch]
                        if target is not None and porta_memory[ch] and target != voice.offset():
                            distance = target - voice.offset()
                            reach = porta_memory[ch] * SEMITONES_PER_PORTA_STEP * moves
                            if reach >= abs(distance):
                                voice.slide(now, now + (row_end - now) * abs(distance) / reach, distance)
                            elif reach:
                                voice.slide(now, row_end, math.copysign(reach, distance))
                    is_vibrato = effect in (4, 6) or column >> 4 == 0xB
                    memory = vibrato_memory[ch]
                    if effect == 4 or column >> 4 in (0xA, 0xB):
                        rate, depth = (sub, param & 15) if effect == 4 else ((column & 15, 0) if column >> 4 == 0xA else (0, column & 15))
                        memory[0] = rate or memory[0]
                        memory[1] = depth or memory[1]
                    if is_vibrato and memory[0] and memory[1]:
                        hertz = memory[0] / VIBRATO_TICKS_PER_CYCLE * bpm * 0.4
                        voice.vibrato.append((now, row_end, min(MAX_VIBRATO_SPEED, max(1, round(hertz / VIBRATO_RAW_SPEED_HZ))),
                                              min(MAX_VIBRATO_DEPTH, round(memory[1] * VIBRATO_DEPTH_RAW_PER_STEP))))
                    # A longer sample can't be joined up without being heard, so its arpeggio is bends.
                    if effect == 0 and param and not voice.chord:
                        base = voice.offset()
                        for step in range(1, speed * (1 + pattern_delay)):
                            wanted = base + (0, sub, param & 15)[step % ARPEGGIO_TICKS_PER_CYCLE]
                            voice.slide(now + step * tick, now + step * tick, wanted - voice.offset())
                        voice.slide(row_end, row_end, base - voice.offset())
                    interval = 0
                    if effect == 27:
                        if param:
                            retrigger_memory[ch] = param
                        interval = retrigger_memory[ch] & 15
                    elif effect == 14 and sub == 9:
                        interval = param & 15
                    ticks = speed * (1 + pattern_delay)
                    for step in range(ticks) if interval else ():
                        # Rxy counts every tick and carries on across rows, where E9x counts back from each row's end.
                        if effect == 27:
                            retrigger_count[ch] += 1
                            if retrigger_count[ch] < interval:
                                continue
                            retrigger_count[ch] = 0
                        elif step == 0 or (ticks - step) % interval:
                            continue
                        change = retrigger_memory[ch] >> 4 if effect == 27 else 0
                        volumes[ch] = max(0.0, min(64.0, volumes[ch] * RETRIGGER_SCALE.get(change, 1) + RETRIGGER_VOLUME.get(change, 0)))
                        time = now + step * tick
                        if time <= voice.start:
                            voice.add_point(time, volumes[ch])
                            continue
                        again = Voice(voice.instrument, voice.pitch, time, time + voice.end - voice.start)
                        again.path = [(time, voice.offset())]
                        again.chord = voice.chord
                        again.add_point(time, volumes[ch])
                        finish(ch, time)
                        voices[ch] = voice = again
                if effect == 14 and sub == 12 and param & 15 < speed:
                    volumes[ch] = 0.0
                    finish(ch, now + (param & 15) * tick)
                if column >> 4 in (PAN_SLIDE_LEFT, PAN_SLIDE_RIGHT):
                    amount = (column & 15) * (speed - 1) * (1 + pattern_delay) * (1 if column >> 4 == PAN_SLIDE_RIGHT else -1)
                    channel_pan[ch] = max(0.0, min(255.0, channel_pan[ch] + amount))
            for ch, voice in enumerate(voices):
                if voice and envelopes[ch]:
                    for step in range(speed * (1 + pattern_delay)):
                        time = now + step * tick
                        if time >= voice.start:
                            voice.envelope.append((time, envelopes[ch].volume_share()))
                            envelopes[ch].advance()
                if voice and pan_envelopes[ch]:
                    for step in range(speed * (1 + pattern_delay)):
                        time = now + step * tick
                        if time >= voice.start:
                            voice.pan.append((time, pan_envelopes[ch].pan_for(channel_pan[ch]) / 255 * MAX_PAN))
                            pan_envelopes[ch].advance()
                elif voice and (voice.pan or channel_pan[ch] != last_sample[ch]["pan"]):
                    if not voice.pan and voice.start < now:
                        voice.pan.append((voice.start, last_sample[ch]["pan"] / 255 * MAX_PAN))
                    voice.pan.append((max(now, voice.start), channel_pan[ch] / 255 * MAX_PAN))
            now += speed * (1 + pattern_delay) * tick
            if stopped:
                break
            if jump_back is not None:
                row = jump_back
                continue
            if jump is not None or row_break is not None:
                break
            row += 1
        if stopped:
            loop_start = 0.0
            break
        if jump is not None:
            position = jump
            start_row = row_break or 0
        else:
            position += 1
            start_row = row_break or 0
    else:
        loop_start = order_starts.get(song["restart"])
    for ch in range(channels):
        finish(ch, now)
    return dict(notes=notes, length=now, tempo=tempo, loop_start=loop_start or 0.0, ignored=ignored,
                order_starts=order_starts, base_speed=base_speed, base_bpm=base_bpm, parts_per_tick=scale)


def bend_points(path):
    """Turn a pitch path into points on whole semitones, which is all a BeepBox bend can hold.

    Bends are linear between points, so placing a point wherever the path crosses a
    semitone keeps a slide at its true speed.
    """
    merged = []
    for time, offset in path:
        if merged and time == merged[-1][0]:
            merged[-1] = (time, offset)
            continue
        if len(merged) >= 2:
            (t0, o0), (t1, o1) = merged[-2:]
            # Rows of the same slide join into one line, or each row's end would round to a stair.
            if t1 > t0 and o1 != o0 and abs((o1 - o0) / (t1 - t0) - (offset - o1) / (time - t1)) < 1e-9:
                merged[-1] = (time, offset)
                continue
        merged.append((time, offset))
    points = [(merged[0][0], round(merged[0][1]))]
    for (t0, o0), (t1, o1) in zip(merged, merged[1:]):
        if o0 == o1 or t1 <= t0:
            if round(o1) != points[-1][1]:
                points.append((t1, round(o1)))
            continue
        if t0 > points[-1][0]:
            points.append((t0, round(o0)))
        low, high = sorted((o0, o1))
        crossings = range(math.ceil(low), math.floor(high) + 1)
        for k in (crossings if o1 > o0 else reversed(crossings)):
            if k != o0:
                points.append((t0 + (k - o0) / (o1 - o0) * (t1 - t0), k))
        points.append((t1, round(o1)))
    return points


def build_note_pieces(voice, bar_parts):
    """Round a note to whole parts and cut it at bar lines. Returns (bar, points, continues), with points of (time, volume, bend)."""
    points = sorted(dict((round(t), v) for t, v in voice.points).items())
    if voice.envelope and points:
        envelope = sorted(dict((round(t), v) for t, v in voice.envelope).items())
        points = [(t, _value_at(points, t) * _value_at(envelope, t)) for t in sorted({t for t, _ in points} | {t for t, _ in envelope})]
    start = round(voice.start)
    end = round(voice.end)
    if not points or end <= start:
        return []
    while len(points) > 1 and points[-2][1] <= 0 and points[-1][1] <= 0:
        points.pop()  # no need to hold a silent tail
    if points[-1][1] <= 0 and points[-1][0] > start:
        end = min(end, points[-1][0])
    cleaned = {}
    for t, v in points:
        if start <= t <= end:
            cleaned[t] = v
    cleaned.setdefault(start, points[0][1])
    if end not in cleaned:
        cleaned[end] = points[-1][1] if points[-1][0] <= end else _value_at(points, end)
    volumes = sorted(cleaned.items())
    if all(v <= 0 for _, v in volumes):
        return []
    bends = {}
    for t, bend in bend_points(voice.path):
        bends[round(t)] = bend
    bends = sorted(bends.items())
    # A point holds a whole-semitone bend, so one placed mid-slide would turn the slide into stairs.
    times = sorted(simplify_volumes(volumes) | {t for t, _ in bends if start < t < end})

    def at(t):
        return (t, _value_at(volumes, t), _value_at(bends, t))

    pieces = []
    bar = start // bar_parts
    while True:
        bar_start, bar_end = bar * bar_parts, (bar + 1) * bar_parts
        a, b = max(start, bar_start), min(end, bar_end)
        if a < b:
            pieces.append((bar, [at(a)] + [at(t) for t in times if a < t < b] + [at(b)], a > start))
        if end <= bar_end:
            return pieces
        bar += 1


def simplify_volumes(volumes):
    """Pick the times a note's volume shape needs points at, so that a fade is one smooth line and not many steps."""
    return simplify_points([(t, (v / 64) ** (2 / 3) * NOTE_SIZE_MAX) for t, v in volumes], VOLUME_TOLERANCE)


def simplify_points(sizes, tolerance):
    """Find the inner points a curve can't lose without straying further than the tolerance."""
    kept = set()
    spans = [(0, len(sizes) - 1)]
    while spans:
        first, last = spans.pop()
        if last - first < 2:
            continue
        (t0, s0), (t1, s1) = sizes[first], sizes[last]
        errors = [(abs(sizes[i][1] - (s0 + (s1 - s0) * (sizes[i][0] - t0) / (t1 - t0))), i) for i in range(first + 1, last)]
        error, worst = max(errors)
        if error > tolerance:
            kept.add(sizes[worst][0])
            spans += [(first, worst), (worst, last)]
    return kept


def _value_at(points, time):
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= time <= t1:
            return v0 + (v1 - v0) * (time - t0) / (t1 - t0)
    return points[-1][1] if time >= points[-1][0] else points[0][1]


def to_json_volume(volume):
    """UltraBox raises note size to the power 1.5, and FT2 volume is linear, so undo that curve."""
    if volume <= 0:
        return 0
    size = max(1, round((volume / 64) ** (2 / 3) * NOTE_SIZE_MAX))  # anything audible stays at least the lowest note size
    return round(size * 100 / NOTE_SIZE_MAX)


def pitch_shifts(notes):
    """Move a sample by whole octaves when its notes would fall outside BeepBox's range."""
    pitches = collections.defaultdict(set)
    for channel in notes:
        for voice in channel:
            pitches[voice.instrument[:2]].add(voice.pitch)
    shifts = {}
    for key, used in pitches.items():
        shift = 0
        while min(used) + shift < 0:
            shift += 12
        while max(used) + shift > MAX_PITCH and min(used) + shift - 12 >= 0:
            shift -= 12
        shifts[key] = shift
    return shifts


def assign_channels(notes_by_xm, bar_parts):
    """Split each XM channel into BeepBox channels where instruments share a bar.

    A BeepBox pattern plays one instrument, so a channel can hold one per bar. An
    instrument stays on its previous channel whenever it can, and notes stay with
    their XM channel so FT2's channel grouping is still visible.
    """
    assigned = []
    for ch, channel in enumerate(notes_by_xm):
        by_bar = collections.defaultdict(lambda: collections.defaultdict(list))
        counts = collections.Counter()
        for voice, pieces in channel:
            counts[voice.instrument] += 1
            for bar, piece, continues in pieces:
                by_bar[bar][voice.instrument].append((voice, piece, continues))
        slots = []
        previous = {}
        for bar in sorted(by_bar):
            taken = set()
            # Instruments that already have a channel choose first, so they can keep it.
            for instrument in sorted(by_bar[bar], key=lambda i: (i not in previous, -counts[i], i)):
                candidates = [s for s in slots if id(s) not in taken]
                slot = previous.get(instrument) if previous.get(instrument) in candidates else None
                slot = slot or next((s for s in candidates if instrument in s["instruments"]), None)
                slot = slot or next((s for s in candidates if len(s["instruments"]) < MAX_INSTRUMENTS_PER_CHANNEL), None)
                if slot is None:
                    slot = dict(xm=ch, instruments=[], notes=collections.defaultdict(list))
                    slots.append(slot)
                if instrument not in slot["instruments"]:
                    slot["instruments"].append(instrument)
                slot["notes"][bar].extend(by_bar[bar][instrument])
                taken.add(id(slot))
                previous[instrument] = slot
        for letter, slot in zip("abcdefghijklmnopqrstuvwxyz", slots):
            slot["name"] = f"Ch{ch + 1} {letter}" if len(slots) > 1 else f"Ch{ch + 1}"
        assigned.extend(slots)
    return merge_sparse_channels(assigned)


def fixed_pan_position(voice, sample):
    """Where a note sits on average, moved to the nearest of a few fixed places in the stereo field (-100 to 100)."""
    raw = statistics.mean(v for _, v in voice.pan) if voice.pan else sample["pan"] / 255 * MAX_PAN
    pan = (raw - MAX_PAN / 2) * 2
    step = 200 / (ROUGH_PAN_POSITIONS - 1) if ROUGH_PAN_POSITIONS > 1 else 200
    return round(round(pan / step) * step) if ROUGH_PAN_POSITIONS > 1 else 0


def note_energy(piece):
    return sum((b[0] - a[0]) * (a[1] ** 2 + b[1] ** 2) for a, b in zip(piece, piece[1:]))


def assign_tracks(notes_by_xm):
    """Lay a song out with no regard for its XM channels, for songs too big to keep them apart.

    In each bar, a sound gets as many channels as it has overlapping notes. Those go to the channels
    they used last where they can. If a bar still doesn't fit, its quietest tracks are left out.
    """
    by_bar = collections.defaultdict(lambda: collections.defaultdict(list))
    for channel in notes_by_xm:
        for voice, pieces in channel:
            for bar, piece, continues in pieces:
                by_bar[bar][voice.instrument].append((voice, piece, continues))
    slots = []
    previous = collections.defaultdict(list)
    dropped_pieces = 0
    for bar in sorted(by_bar):
        tracks = []
        for instrument, pieces in by_bar[bar].items():
            mine = []
            for item in sorted(pieces, key=lambda n: n[1][0][0]):
                start, end = item[1][0][0], item[1][-1][0]
                for track in mine:
                    if track["end"] <= start:
                        break
                else:
                    track = dict(instrument=instrument, end=0, notes=[], energy=0.0)
                    mine.append(track)
                    tracks.append(track)
                track["end"] = end
                track["notes"].append(item)
                track["energy"] += note_energy(item[1])
        tracks.sort(key=lambda t: -t["energy"])
        for track in tracks[MAX_CHANNELS:]:
            dropped_pieces += len(track["notes"])
        tracks = tracks[:MAX_CHANNELS]
        taken = set()
        placed = collections.defaultdict(list)
        waiting = []
        for track in tracks:  # keep sounds where they were, loudest first
            usual = [s for s in previous[track["instrument"]] if id(s) not in taken]
            if usual:
                taken.add(id(usual[0]))
                usual[0]["notes"][bar].extend(track["notes"])
                placed[track["instrument"]].append(usual[0])
            else:
                waiting.append(track)
        for track in waiting:
            free = [s for s in slots if id(s) not in taken]
            slot = next((s for s in free if track["instrument"] in s["instruments"]), None)
            slot = slot or next((s for s in free if len(s["instruments"]) < MAX_INSTRUMENTS_PER_CHANNEL), None)
            if slot is None and len(slots) < MAX_CHANNELS:
                slot = dict(xm=len(slots), instruments=[], notes=collections.defaultdict(list), name=f"Track {len(slots) + 1}")
                slots.append(slot)
            if slot is None:
                dropped_pieces += len(track["notes"])
                continue
            if track["instrument"] not in slot["instruments"]:
                slot["instruments"].append(track["instrument"])
            taken.add(id(slot))
            slot["notes"][bar].extend(track["notes"])
            placed[track["instrument"]].append(slot)
        for instrument, where in placed.items():
            previous[instrument] = where + [s for s in previous[instrument] if s not in where]
    return slots, dropped_pieces


def merge_sparse_channels(slots):
    """Fold rarely used channels into others that are silent in those bars."""
    bar_count = max((max(s["notes"]) for s in slots), default=0) + 1
    for slot in sorted(slots, key=lambda s: len(s["notes"])):
        if len(slot["notes"]) > bar_count * SPARSE_CHANNEL_SHARE:
            continue
        targets = [t for t in slots if t is not slot and not (t["notes"].keys() & slot["notes"].keys())
                   and len(set(t["instruments"]) | set(slot["instruments"])) <= MAX_INSTRUMENTS_PER_CHANNEL]
        if not targets:
            continue
        target = max(targets, key=lambda t: len(t["notes"]))
        target["notes"].update(slot["notes"])
        target["instruments"].extend(i for i in slot["instruments"] if i not in target["instruments"])
        target["merged"] = target.get("merged", []) + [slot["name"]] + slot.get("merged", [])
        if not target["name"].endswith("+"):
            target["name"] += "+"
        slots.remove(slot)
    return slots


def wave_name(index, sample_index):
    return f"Inst{index:02d}{'abcdefghijklmnopqrstuvwxyz'[sample_index % 26]}"


def pitch_volume_offset(pitches, frames):
    """Cancel out UltraBox's habit of playing higher pitches more quietly, which FT2 doesn't have.

    For samples it counts the pitch a whole sample would have if it were one cycle of a wave, so
    a long sample counts as a very low pitch and comes out loud. That means its length matters too.
    """
    pitch = statistics.median(pitches) - NATIVE_PITCH - 12 * math.log2(max(1, frames) / REFERENCE_FRAMES)
    return max(-MAX_VOLUME, min(MAX_VOLUME, INSTRUMENT_VOLUME + round(pitch * VOLUME_PER_PITCH)))


def fine_detune(voice):
    """Find a note's lasting pitch shift of under half a semitone, in cents, which a whole-semitone bend can't hold.

    These matter more than their size suggests. When two channels play the same sample and one is nudged like this,
    FT2 gives a chorus, where without the nudge the two stay in phase and come out about 3 dB too loud.
    """
    offsets = [o for _, o in voice.path]
    if not any(offsets) or max(abs(o) for o in offsets) >= 0.5:
        return 0
    return round(offsets[-1] * 100)


def instrument_json(instrument, song, volume=INSTRUMENT_VOLUME, vibrato_speed=None, moving_pan=False):
    index, sample_index, offset, cents, *fixed_pan = instrument
    sample = song["instruments"][index]["samples"][sample_index]
    frames = sample["loop_end"] if sample["loop"] != LOOP_NONE else sample["frames"]
    keep = 2 if sample.get("halved") else 1  # a halved sample has half as many frames to count through
    result = dict(type="chip", volume=volume, wave=wave_name(index, sample_index), eqFilter=[], effects=[],
                  fadeInSeconds=0, fadeOutTicks=-3, chord="simultaneous", envelopes=[], unison="none",
                  isUsingAdvancedLoopControls=True, chipWaveLoopMode=sample["loop"], chipWavePlayBackwards=False,
                  chipWaveLoopStart=(sample["loop_start"] if sample["loop"] != LOOP_NONE else 0) // keep,
                  chipWaveLoopEnd=frames // keep, chipWaveStartOffset=offset // keep)
    if sample["pan"] != 128 or moving_pan:
        result["effects"] = ["panning"]
        result["pan"] = round((sample["pan"] - 128) / 128 * 100)
    if fixed_pan:
        result["effects"] = ["panning"]
        result["pan"] = fixed_pan[0]
    if song["instruments"][index].get("chord_arpeggio"):
        result["effects"] = result["effects"] + ["chord type", "transition type"]
        result.update(chord="arpeggio", transition="interrupt", fastTwoNoteArp=True, arpeggioSpeed=song["arpeggio_speed"])
    if cents:
        result["effects"] = result["effects"] + ["detune"]
        result["detuneCents"] = cents
    auto = song["instruments"][index].get("auto_settings")
    if vibrato_speed is not None or auto:
        # Without auto-vibrato the depth stays 0 here, because a modulation channel turns the vibrato on and off.
        result["effects"] = result["effects"] + ["vibrato"]
        result.update(vibrato="custom", vibratoDepth=auto["depth"] if auto else 0, vibratoType=0,
                      vibratoSpeed=vibrato_speed if vibrato_speed is not None else auto["speed"],
                      vibratoDelay=auto["delay"] if auto else 0)
    return result


def auto_vibrato_settings(auto, bpm, parts_per_tick):
    """Turn FT2 auto-vibrato into UltraBox settings. UltraBox can't sweep, so half the sweep becomes a delay."""
    hertz = auto["rate"] / AUTO_VIBRATO_TICKS_PER_CYCLE * bpm * 0.4
    return dict(depth=auto["depth"] * AUTO_VIBRATO_SEMITONES_PER_STEP,
                speed=min(MAX_VIBRATO_SPEED, max(1, round(hertz / VIBRATO_RAW_SPEED_HZ))),
                delay=min(MAX_VIBRATO_DELAY, round(auto["sweep"] / 2 * parts_per_tick)))


def sample_frames(instrument, song):
    """How long a sample is in UltraBox before any halving, which leaves its loudness the same."""
    sample = song["instruments"][instrument[0]]["samples"][instrument[1]]
    return sample["loop_end"] if sample["loop"] != LOOP_NONE else sample["frames"]


def build_channel(slot, song, shifts, bar_count, ticks_per_beat, bar_parts, vibrato_speeds, panned):
    patterns, lookup = [], {}
    sequence = []
    instruments = slot["instruments"]
    all_pitches = []
    instrument_pitches = collections.defaultdict(list)
    for bar in range(bar_count):
        pieces = sorted(slot["notes"].get(bar, []), key=lambda n: n[1][0][0])
        if not pieces:
            sequence.append(0)
            continue
        json_notes = []
        for voice, piece, continues in pieces:
            pitch = voice.pitch + shifts[voice.instrument[:2]]
            all_pitches.append(pitch)
            instrument_pitches[voice.instrument].append(voice.pitch)  # a shift moves the root key too, so it changes nothing
            points = []
            for time, volume, bend in piece:
                level = to_json_volume(volume)
                bend = round(bend)
                if (len(points) >= 2 and points[-1]["volume"] == level and points[-2]["volume"] == level
                        and points[-1]["pitchBend"] == bend and points[-2]["pitchBend"] == bend):
                    points.pop()
                points.append(dict(tick=(time - bar * bar_parts) * ticks_per_beat / PARTS_PER_BEAT, pitchBend=bend, volume=level))
            note = dict(pitches=[pitch] + [pitch + i for i in voice.chord if pitch + i <= MAX_PITCH], points=points)
            if points[0]["tick"] == 0:
                note["continuesLastPattern"] = continues
            json_notes.append(note)
        pattern = dict(instruments=[instruments.index(pieces[0][0].instrument) + 1], notes=json_notes)
        key = json.dumps(pattern, sort_keys=True)
        if key not in lookup:
            patterns.append(pattern)
            lookup[key] = len(patterns)
        sequence.append(lookup[key])
    channel = dict(type="pitch", name=slot["name"], patterns=patterns, sequence=sequence,
                   instruments=[instrument_json(i, song, pitch_volume_offset(instrument_pitches[i], sample_frames(i, song)), vibrato_speeds.get(i), i in panned) for i in instruments])
    if all_pitches:
        octave = max(0, min(8, (int(statistics.median(all_pitches)) - 18) // 12))
        channel["octaveScrollBar"] = octave - 1
    return channel


def vibrato_runs(slot):
    """Collect a channel's vibrato as {instrument: {bar: [(start, end, raw speed, raw depth)]}}, cut at bar lines."""
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for bar, pieces in sorted(slot["notes"].items()):
        for voice, piece, _ in sorted(pieces, key=lambda n: n[1][0][0]):
            first, last = piece[0][0], piece[-1][0]
            for start, end, speed, depth in voice.vibrato:
                a, b = max(first, round(start)), min(last, round(end))
                if a >= b:
                    continue
                row = runs[voice.instrument][bar]
                if row and row[-1][1] == a and row[-1][2:] == (speed, depth):
                    row[-1] = (row[-1][0], b, speed, depth)
                else:
                    row.append((a, b, speed, depth))
    return runs


def with_vibrato_stops(bars, bar_parts, resting_depth):
    """UltraBox keeps a modulator's last value after its note ends, so each run ends with a short note at the resting depth."""
    runs = sorted(run for row in bars.values() for run in row)
    result = collections.defaultdict(list)
    for index, (start, end, depth) in enumerate(runs):
        result[start // bar_parts].append([(start, depth), (end, depth)])
        if index + 1 == len(runs) or runs[index + 1][0] != end:
            result[end // bar_parts].append([(end, resting_depth), (end + 1, resting_depth)])
    return result


def pan_curves(slot, song):
    """Collect a channel's moving pan as {instrument: {bar: [mod note points]}}, one note for each note piece.

    A modulator holds its last value, so once one of an instrument's notes moves its pan, they all need a pan note.
    """
    curves = collections.defaultdict(lambda: collections.defaultdict(list))
    moving = {voice.instrument for pieces in slot["notes"].values() for voice, _, _ in pieces if voice.pan}
    for bar, pieces in sorted(slot["notes"].items()):
        for voice, piece, _ in sorted(pieces, key=lambda n: n[1][0][0]):
            if voice.instrument not in moving:
                continue
            first, last = piece[0][0], piece[-1][0]
            sample = song["instruments"][voice.instrument[0]]["samples"][voice.instrument[1]]
            pan = sorted(dict((round(t), v) for t, v in voice.pan).items()) or [(first, sample["pan"] / 255 * MAX_PAN)]
            inner = [p for p in pan if first < p[0] < last]
            curve = [(first, _value_at(pan, first))] + inner + [(last, _value_at(pan, last))]
            kept = simplify_points(curve, PAN_TOLERANCE)
            curves[voice.instrument][bar].append([(t, round(v)) for t, v in curve if t in kept or t in (first, last)])
    return curves


def build_mod_channels(slots, song, bar_count, ticks_per_beat, bar_parts):
    """Build the modulation channels that switch vibrato on and off, as FT2's effects do row by row.

    Pan envelopes also go here, as curves. Returns the channels, each channel's
    {instrument: resting vibrato speed}, and each channel's set of instruments with moving pan.
    A second target for speed is only used where one instrument on one channel uses several speeds.
    """
    targets = []
    speeds = []
    panned = []
    for channel_index, slot in enumerate(slots):
        rest = {}
        for instrument, bars in sorted(vibrato_runs(slot).items()):
            used = collections.Counter()
            for row in bars.values():
                for start, end, speed, _ in row:
                    used[speed] += end - start
            rest[instrument] = used.most_common(1)[0][0]
            where = (channel_index, slot["instruments"].index(instrument))
            depths = {b: [(s, e, d) for s, e, _, d in row] for b, row in bars.items()}
            auto = song["instruments"][instrument[0]].get("auto_settings")
            resting_depth = round(auto["depth"] * VIBRATO_RAW_PER_SEMITONE) if auto else 0
            targets.append((*where, MOD_VIBRATO_DEPTH, with_vibrato_stops(depths, bar_parts, resting_depth)))
            if len(used) > 1:
                targets.append((*where, MOD_VIBRATO_SPEED, {b: [[(s, v), (e, v)] for s, e, v, _ in row] for b, row in bars.items()}))
        speeds.append(rest)
        curves = pan_curves(slot, song)
        panned.append(set(curves))
        for instrument, bars in sorted(curves.items()):
            targets.append((channel_index, slot["instruments"].index(instrument), MOD_PAN, bars))
    channels = []
    for group in range(0, len(targets), MOD_SLOTS):
        members = targets[group:group + MOD_SLOTS]
        filler = [MOD_NO_CHANNEL] * (MOD_SLOTS - len(members))
        instrument = dict(type="mod", volume=0, eqFilter=[], effects=[], envelopes=[], fadeInSeconds=0, fadeOutTicks=-3,
                          modChannels=[m[0] for m in members] + filler, modInstruments=[m[1] for m in members] + [0] * len(filler),
                          modSettings=[m[2] for m in members] + [0] * len(filler), modFilterTypes=[0] * MOD_SLOTS)
        patterns, lookup, sequence = [], {}, []
        for bar in range(bar_count):
            notes = []
            for slot_index, member in enumerate(members):
                for curve in member[3].get(bar, []):
                    notes.append((curve[0][0], dict(pitches=[MOD_SLOTS - 1 - slot_index], points=[
                        dict(tick=(t - bar * bar_parts) * ticks_per_beat / PARTS_PER_BEAT, pitchBend=0, volume=value, forMod=True)
                        for t, value in curve])))
            if not notes:
                sequence.append(0)
                continue
            pattern = dict(instruments=[1], notes=[n for _, n in sorted(notes, key=lambda n: n[0])])
            key = json.dumps(pattern, sort_keys=True)
            if key not in lookup:
                patterns.append(pattern)
                lookup[key] = len(patterns)
            sequence.append(lookup[key])
        channels.append(dict(type="mod", name=f"Mod {len(channels) + 1}", instruments=[instrument], patterns=patterns, sequence=sequence))
    return channels, speeds, panned


def add_lead_in(played, bar_parts):
    """Delay the song by the silence that puts the most XM song positions on bar lines."""
    starts = [round(t) % bar_parts for t in played["order_starts"].values()]
    lead_in = (bar_parts - collections.Counter(starts).most_common(1)[0][0]) % bar_parts
    for channel in played["notes"]:
        for voice in channel:
            voice.start += lead_in
            voice.end += lead_in
            voice.points = [(t + lead_in, v) for t, v in voice.points]
            voice.path = [(t + lead_in, o) for t, o in voice.path]
            voice.vibrato = [(a + lead_in, b + lead_in, s, d) for a, b, s, d in voice.vibrato]
            voice.envelope = [(t + lead_in, v) for t, v in voice.envelope]
            voice.pan = [(t + lead_in, v) for t, v in voice.pan]
    played["order_starts"] = {k: t + lead_in for k, t in played["order_starts"].items()}
    played["length"] += lead_in
    played["loop_start"] += lead_in
    return lead_in


def estimated_loudness(song, played):
    """Roughly how loud a song's mix is in its busier moments: the root of its summed note powers, part by part."""
    powers = {}
    for key in {v.instrument[:2] for c in played["notes"] for v in c}:
        sample = song["instruments"][key[0]]["samples"][key[1]]
        values = as_values(sample_audio(sample), sample["wide"])
        full_scale = 32768 if sample["wide"] else 128
        powers[key] = sum(v * v for v in values) / max(1, len(values)) / full_scale ** 2
    buckets = collections.Counter()
    for channel in played["notes"]:
        for voice in channel:
            points = sorted(dict(voice.points).items())
            envelope = sorted(dict(voice.envelope).items()) if voice.envelope else None
            if not points:
                continue
            time = voice.start
            while time < min(voice.end, played["length"]):
                level = _value_at(points, time) / 64 * (_value_at(envelope, time) if envelope else 1)
                buckets[int(time // LOUDNESS_STEP)] += level ** 2 * powers[voice.instrument[:2]]
                time += LOUDNESS_STEP
    levels = sorted(buckets.values())
    return math.sqrt(levels[int(len(levels) * 0.95)]) if levels else 1.0


def custom_samples(song, shifts, sample_url=None, folder=None):
    """One sample per (instrument, sample) that gets played, embedded or else written to a folder to be served."""
    urls = []
    for index, sample_index in sorted(shifts):
        sample = song["instruments"][index]["samples"][sample_index]
        # A halved sample has half the frames for the same sound, so it plays an octave down to keep its speed.
        root_key = ROOT_KEY + shifts[(index, sample_index)] - sample["finetune"] / 128 + (12 if sample.get("halved") else 0)
        name = wave_name(index, sample_index)
        if sample_url:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(wav_file(sample, name))
            urls.append(f"!s{SAMPLE_RATE},r{root_key:g}!{sample_url}{urllib.parse.quote(folder.name)}/{name}")
        else:
            urls.append(wav_data_url(sample, name, root_key))
    return urls


def describe_instrument(instrument):
    index, sample_index, offset, cents, *fixed_pan = instrument
    text = str(index)
    if sample_index:
        text += f" sample {sample_index + 1}"
    if offset:
        text += f" from frame {offset}"
    if cents:
        text += f" detuned {cents:+d}"
    if fixed_pan and fixed_pan[0]:
        text += f" pan {fixed_pan[0]:+d}"
    return text


def convert(path, output_dir, beats_per_bar, rough=False, sample_url=None, require_url_fit=False):
    song = parse_xm(path)
    played = simulate(song)
    bar_parts = beats_per_bar * PARTS_PER_BEAT
    lead_in = add_lead_in(played, bar_parts)
    # A step takes 3 / speed synth ticks, at 2 synth ticks a part, and FT2 repeats every 3 ticks, which is 2 steps of a 2-note chord.
    wanted_speed = 1 / played["parts_per_tick"]
    song["arpeggio_speed"] = min(range(len(ARPEGGIO_SPEEDS)), key=lambda i: abs(ARPEGGIO_SPEEDS[i] - wanted_speed))
    for channel in played["notes"]:
        for voice in channel:
            voice.instrument = (*voice.instrument, fine_detune(voice))
            if rough:
                sample = song["instruments"][voice.instrument[0]]["samples"][voice.instrument[1]]
                voice.instrument = (*voice.instrument, fixed_pan_position(voice, sample))
                voice.pan = []
    shifts = pitch_shifts(played["notes"])
    by_xm = []
    dropped = 0
    starts = collections.Counter()
    for channel in played["notes"]:
        kept = []
        voices = sorted(channel, key=lambda v: v.start)
        for voice, following in zip(voices, voices[1:] + [None]):
            if following is not None:
                voice.end = min(voice.end, following.start)
            voice.end = min(voice.end, played["length"])
            pieces = build_note_pieces(voice, bar_parts)
            if pieces:
                kept.append((voice, pieces))
                starts[round(voice.start) % (PARTS_PER_BEAT // 4) == 0] += 1
            else:
                dropped += 1
        by_xm.append(kept)
    on_grid = starts[True] / max(1, sum(starts.values()))
    ticks_per_beat = 4 if on_grid >= ON_GRID_SHARE else PARTS_PER_BEAT
    sample_chars, halved = fit_samples_in_url(song, shifts) if not sample_url else (0, [])
    if require_url_fit and sample_chars > SAMPLE_URL_BUDGET:
        raise SamplesTooBig(f"Its samples take {sample_chars / 1e6:.1f} MB, and only {SAMPLE_URL_BUDGET / 1e6:.2f} MB fits in a song's URL in Firefox.")
    # With UltraBox's compressor off to match FT2's linear mixing, a song's level is set here to suit how busy it is.
    loudness = estimated_loudness(song, played)
    master_gain = round(min(MAX_MASTER_GAIN, math.sqrt(LEVEL_TARGET / loudness)) * 50) / 50
    slots, dropped_pieces = assign_tracks(by_xm) if rough else (assign_channels(by_xm, bar_parts), 0)
    bar_count = math.ceil(round(played["length"]) / bar_parts)
    for instrument in song["instruments"].values():
        if instrument["auto_vibrato"]:
            instrument["auto_settings"] = auto_vibrato_settings(instrument["auto_vibrato"], played["base_bpm"], played["parts_per_tick"])
    mod_channels, vibrato_speeds, panned = build_mod_channels(slots, song, bar_count, ticks_per_beat, bar_parts)
    channels = [build_channel(s, song, shifts, bar_count, ticks_per_beat, bar_parts, v, m) for s, v, m in zip(slots, vibrato_speeds, panned)]
    if len(channels) > MAX_CHANNELS or len(mod_channels) > MAX_CHANNELS:
        raise TooManyChannels(f"{path.name} needs {len(channels)} note and {len(mod_channels)} modulation channels at {beats_per_bar} beats per bar; UltraBox allows {MAX_CHANNELS} of each")
    channels += mod_channels
    intro = int(played["loop_start"] // bar_parts)
    sample_folder = output_dir / f"{path.stem}.samples"
    samples = custom_samples(song, shifts, sample_url, sample_folder)
    result = {"format": "UltraBox", "version": 5, "name": song["title"] or path.stem, "scale": "Free", "key": "C",
              "introBars": intro, "loopBars": bar_count - intro, "beatsPerBar": beats_per_bar,
              "ticksPerBeat": ticks_per_beat, "beatsPerMinute": round(played["tempo"]),
              "layeredInstruments": False, "patternInstruments": True, "compressionRatio": 0, "masterGain": master_gain, "customSamples": samples, "channels": channels}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{path.stem}.ultrabox.json"
    json_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")

    detuned = sum(1 for c in by_xm for v, _ in c if v.instrument[3])
    played_instruments = sorted({v.instrument[0] for c in by_xm for v, _ in c})
    swept = [i for i in played_instruments if (song["instruments"][i]["auto_vibrato"] or {}).get("sweep")]
    other_waves = [i for i in played_instruments if (song["instruments"][i]["auto_vibrato"] or {}).get("kind")]
    note_total = sum(len(c) for c in by_xm)
    pattern_total = sum(len(c["patterns"]) for c in channels)
    used_bars = sum(sum(1 for b in c["sequence"] if b) for c in channels)
    aligned = sum(round(t) % bar_parts == 0 for t in played["order_starts"].values())
    variants = {v[0].instrument for c in by_xm for v in c}
    lines = [f"{path.name} -> {json_path.name}", "",
             f"Tempo {result['beatsPerMinute']} (FT2 speed {played['base_speed']}, BPM {played['base_bpm']} = {PARTS_PER_ROW} parts per row, 4 rows per beat)",
             f"{bar_count} bars of {beats_per_bar} beats, rhythm grid {ticks_per_beat} per beat ({on_grid:.1%} of notes on the 1/4 beat grid)",
             f"{len(channels) - len(mod_channels)} note channels and {len(mod_channels)} modulation channels for vibrato and pan (from {song['channels']} XM channels)",
             f"{note_total} notes kept, {dropped} dropped because they were silent or shorter than one part",
             f"{pattern_total} unique patterns across {used_bars} used channel-bars ({1 - pattern_total / max(1, used_bars):.0%} reuse)",
             f"{aligned} of {len(played['order_starts'])} XM song positions start exactly on a bar line ({lead_in / PARTS_PER_BEAT:g} beats of silent lead-in added)",
             (f"{len(samples)} samples in {sample_folder.name}, so open this song in UltraBox through start_ultrabox.py, {len(variants)} instrument variants" if sample_url else
              f"{len(samples)} samples embedded, {sum(len(u) for u in samples) / 1e6:.2f} MB in the song file, {len(variants)} instrument variants"), "",
             "Channel map (XM instrument numbers, in the same order as each channel's instrument list):"]
    for slot in slots:
        merged = f" (also holds {', '.join(slot['merged'])})" if slot.get("merged") else ""
        lines.append(f"  {slot['name']:<8} instruments {', '.join(describe_instrument(i) for i in slot['instruments'])}{merged}")
    shifted = {f"{wave_name(*k)}": s for k, s in shifts.items() if s}
    if shifted:
        lines += ["", f"Moved by octaves to fit the pitch range: {shifted}"]
    if swept:
        lines += ["", f"Auto-vibrato sweep is a delay instead of a gradual fade-in, on instruments: {', '.join(map(str, swept))}"]
    if other_waves:
        lines += ["", f"Auto-vibrato uses a sine instead of another waveform, on instruments: {', '.join(map(str, other_waves))}"]
    if rough:
        lines += ["", f"Rough version: pan uses {ROUGH_PAN_POSITIONS} fixed positions, and {dropped_pieces} of {sum(len(p) for c in by_xm for _, p in c)} note pieces were left out to fit {MAX_CHANNELS} channels"]
    if detuned:
        lines += ["", f"{detuned} notes have small pitch shifts carried over as detuned instrument variants"]
    if halved:
        lines += ["", f"Samples at half their rate, so the song fits in a Firefox URL: {', '.join(wave_name(*k) for k in halved)}"]
    if sample_chars > SAMPLE_URL_BUDGET:
        warning = (f"WARNING: the samples take {sample_chars / 1e6:.2f} MB, over the {SAMPLE_URL_BUDGET / 1e6:.2f} MB that fits in a Firefox URL. "
                   "UltraBox in Firefox won't load them and will play its default wave instead. Chrome and Edge allow about 2 MB.")
        lines += ["", warning]
    lines += ["", "Effects not carried over:"]
    lines += [f"  {count:>6}  {name}" for name, count in played["ignored"].most_common()] or ["  none"]
    report_path = output_dir / f"{path.stem}.report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return json_path


def convert_at_best_bar_size(path, output_dir, beats_per_bar=None, **options):
    """Convert a song at the longest bar that fits its channels."""
    error = None
    for beats in ([beats_per_bar] if beats_per_bar else BAR_SIZES):
        try:
            return convert(path, output_dir, beats, **options)
        except TooManyChannels as problem:
            error = problem
    raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xm", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("output"))
    parser.add_argument("--beats-per-bar", type=int, help="by default, the longest bar that fits the song's channels")
    parser.add_argument("--rough", action="store_true", help="make an approximate version of a song too big to convert normally")
    parser.add_argument("--sample-url", help=f"serve samples from here instead of embedding them (--rough uses {DEFAULT_SAMPLE_URL})")
    args = parser.parse_args()
    sample_url = args.sample_url or (DEFAULT_SAMPLE_URL if args.rough else None)
    try:
        convert_at_best_bar_size(args.xm, args.output, args.beats_per_bar, rough=args.rough, sample_url=sample_url)
    except TooManyChannels as problem:
        raise SystemExit(str(problem))


if __name__ == "__main__":
    main()

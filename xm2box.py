#!/usr/bin/env python3
"""Convert a FastTracker II .xm module into UltraBox song JSON.

Carries over notes, timing, song order, volume, vibrato, pitch slides and the module's
own samples, which are stored inside the song file. The report written next to the song
lists what was left out.

Usage: python xm2box.py song.xm [-o output_dir] [--beats-per-bar 16]
"""
import argparse
import base64
import collections
import json
import math
import pathlib
import statistics
import struct

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
LOOP_FORWARD, LOOP_PING_PONG, LOOP_NONE = 0, 1, 2  # UltraBox's chipWaveLoopMode values
SEMITONES_PER_PORTA_STEP = 1 / 16  # linear periods: 1xx, 2xx, 3xx and E1x move 4 of the 64 units in a semitone
SEMITONES_PER_EXTRA_FINE_STEP = 1 / 64
VIBRATO_RAW_PER_SEMITONE = 25
VIBRATO_DEPTH_RAW_PER_STEP = VIBRATO_RAW_PER_SEMITONE / 8  # FT2 peaks at y/8 semitones, UltraBox at raw/25
VIBRATO_RAW_SPEED_HZ = 0.1 / 0.14  # UltraBox's rate per raw speed: 0.1 of its 0.14 second vibrato period
VIBRATO_TICKS_PER_CYCLE = 64  # FT2's vibrato takes this many ticks at speed 1
MAX_VIBRATO_DEPTH = 50
MAX_VIBRATO_SPEED = 30
MOD_SLOTS = 6
MOD_NO_CHANNEL = -2
MOD_VIBRATO_DEPTH, MOD_VIBRATO_SPEED = 16, 18  # positions in UltraBox's list of modulators
HANDLED_EFFECTS = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 33}
HANDLED_EXTENDED = {1, 2, 6, 10, 11, 12, 13, 14}
EFFECT_NAMES = {0: "arpeggio", 7: "tremolo", 8: "set pan",
                16: "global volume", 17: "global volume slide", 20: "key off at tick",
                21: "envelope position", 25: "pan slide", 27: "retrigger", 29: "tremor"}
EXTENDED_NAMES = {3: "glissando control", 4: "vibrato waveform",
                  5: "set finetune", 7: "tremolo waveform", 8: "set pan", 9: "retrigger"}


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
        instrument = dict(name=read_string(d[p + 4:p + 26]), samples=[], keymap=None, volume_envelope=None, pan_envelope=False, auto_vibrato=None)
        if sample_count:
            sample_header = struct.unpack_from("<I", d, p + 29)[0]
            instrument["keymap"] = d[p + 33:p + 129]
            point_count, _, sustain, loop_start, loop_end, _, _, _, kind, pan_kind, vibrato_kind, sweep, depth, rate = struct.unpack_from("<14B", d, p + 225)
            if kind & 1 and point_count:
                values = struct.unpack_from(f"<{point_count * 2}H", d, p + 129)
                instrument["volume_envelope"] = dict(points=list(zip(values[::2], values[1::2])),
                                                     sustain=sustain if kind & 2 else None,
                                                     loop=(loop_start, loop_end) if kind & 4 else None,
                                                     fadeout=struct.unpack_from("<H", d, p + 239)[0])
            instrument["pan_envelope"] = bool(pan_kind & 1)
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


def wav_data_url(sample, name, root_key):
    """Build an UltraBox custom sample URL holding the sample as an embedded WAV file.

    UltraBox names a sample after whatever follows the last "/" in its URL. So the
    WAV ends with an unused chunk whose base64 text spells "/<name>", which makes
    that name appear in UltraBox's wave menu. The name must be 3 less than a multiple
    of 4 long, so that the "/" and the name fill whole base64 groups.
    """
    width = 2 if sample["wide"] else 1
    audio = decode_sample(sample)
    if sample["loop"] != LOOP_NONE:
        audio = audio[:sample["loop_end"] * width]  # FT2 never plays past the end of a loop
    if len(audio) % 2:
        audio += b"\0"
    label = base64.b64decode("/" + name)
    filler = 0
    while (12 + 16 + 8 + 8 + len(audio) + 8 + filler) % 3 or (filler + len(label)) % 2:
        filler += 1
    chunk = b"\0" * filler + label
    body = (b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * width, width, 8 * width)
            + b"data" + struct.pack("<I", len(audio)) + audio + b"JUNK" + struct.pack("<I", len(chunk)) + chunk)
    encoded = base64.b64encode(b"RIFF" + struct.pack("<I", len(body)) + body).decode("ascii")
    return f"!s{SAMPLE_RATE},r{root_key:g}!data:audio/wav;base64,{encoded}"


class Voice:
    def __init__(self, instrument, pitch, start, end_limit):
        self.instrument = instrument  # (XM instrument, sample within it, start offset in frames)
        self.pitch = pitch
        self.start = start
        self.end = end_limit
        self.points = []
        self.path = [(start, 0.0)]  # (time, semitones away from the note's pitch)
        self.vibrato = []  # (start, end, raw speed, raw depth)
        self.envelope = []  # (time, volume envelope and fadeout as a share of full volume)

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


class VolumeEnvelope:
    """A channel's place in an instrument's volume envelope, stepped tick by tick as FT2 does."""

    def __init__(self, envelope):
        self.envelope = envelope
        self.tick = 0
        self.fade = FADE_START
        self.released = False

    def value(self):
        return _value_at(self.envelope["points"], self.tick) / 64 * self.fade / FADE_START

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
    return speeds.most_common(1)[0][0], bpms.most_common(1)[0][0]


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
                if effect == 15 and param:
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
                if column >> 4 in (0xC, 0xD, 0xE):
                    ignored["volume column pan"] += 1
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
                    envelopes[ch] = VolumeEnvelope(envelope) if envelope else None
                is_porta = effect in (3, 5) or column >> 4 == 0xF
                if note == 97:
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
                if instrument_index and last_sample[ch]:
                    volumes[ch] = float(last_sample[ch]["volume"])
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
                if effect == 14 and sub == 12 and param & 15 < speed:
                    volumes[ch] = 0.0
                    finish(ch, now + (param & 15) * tick)
            for ch, voice in enumerate(voices):
                if voice and envelopes[ch]:
                    for step in range(speed * (1 + pattern_delay)):
                        time = now + step * tick
                        if time >= voice.start:
                            voice.envelope.append((time, envelopes[ch].value()))
                            envelopes[ch].advance()
            now += speed * (1 + pattern_delay) * tick
            if jump_back is not None:
                row = jump_back
                continue
            if jump is not None or row_break is not None:
                break
            row += 1
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
    sizes = [(t, (v / 64) ** (2 / 3) * NOTE_SIZE_MAX) for t, v in volumes]
    kept = set()
    spans = [(0, len(sizes) - 1)]
    while spans:
        first, last = spans.pop()
        if last - first < 2:
            continue
        (t0, s0), (t1, s1) = sizes[first], sizes[last]
        errors = [(abs(sizes[i][1] - (s0 + (s1 - s0) * (sizes[i][0] - t0) / (t1 - t0))), i) for i in range(first + 1, last)]
        error, worst = max(errors)
        if error > VOLUME_TOLERANCE:
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


def pitch_volume_offset(pitches):
    """UltraBox plays higher notes more quietly and FT2 doesn't, so offset that for an instrument's usual pitch."""
    offset = round((statistics.median(pitches) - NATIVE_PITCH) * VOLUME_PER_PITCH)
    return max(-MAX_VOLUME, min(MAX_VOLUME, INSTRUMENT_VOLUME + offset))


def instrument_json(instrument, song, volume=INSTRUMENT_VOLUME, vibrato_speed=None):
    index, sample_index, offset = instrument
    sample = song["instruments"][index]["samples"][sample_index]
    frames = sample["loop_end"] if sample["loop"] != LOOP_NONE else sample["frames"]
    result = dict(type="chip", volume=volume, wave=wave_name(index, sample_index), eqFilter=[], effects=[],
                  fadeInSeconds=0, fadeOutTicks=-3, chord="simultaneous", envelopes=[], unison="none",
                  isUsingAdvancedLoopControls=True, chipWaveLoopMode=sample["loop"], chipWavePlayBackwards=False,
                  chipWaveLoopStart=sample["loop_start"] if sample["loop"] != LOOP_NONE else 0,
                  chipWaveLoopEnd=frames, chipWaveStartOffset=offset)
    if sample["pan"] != 128:
        result["effects"] = ["panning"]
        result["pan"] = round((sample["pan"] - 128) / 128 * 100)
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


def build_channel(slot, song, shifts, bar_count, ticks_per_beat, bar_parts, vibrato_speeds):
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
            instrument_pitches[voice.instrument].append(pitch)
            points = []
            for time, volume, bend in piece:
                level = to_json_volume(volume)
                bend = round(bend)
                if (len(points) >= 2 and points[-1]["volume"] == level and points[-2]["volume"] == level
                        and points[-1]["pitchBend"] == bend and points[-2]["pitchBend"] == bend):
                    points.pop()
                points.append(dict(tick=(time - bar * bar_parts) * ticks_per_beat / PARTS_PER_BEAT, pitchBend=bend, volume=level))
            note = dict(pitches=[pitch], points=points)
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
                   instruments=[instrument_json(i, song, pitch_volume_offset(instrument_pitches[i]), vibrato_speeds.get(i)) for i in instruments])
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
        result[start // bar_parts].append((start, end, depth))
        if index + 1 == len(runs) or runs[index + 1][0] != end:
            result[end // bar_parts].append((end, end + 1, resting_depth))
    return result


def build_mod_channels(slots, song, bar_count, ticks_per_beat, bar_parts):
    """Build the modulation channels that switch vibrato on and off, as FT2's effects do row by row.

    Returns the channels and each channel's {instrument: resting vibrato speed}. A second
    target for speed is only used where one instrument on one channel uses several speeds.
    """
    targets = []
    speeds = []
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
                targets.append((*where, MOD_VIBRATO_SPEED, {b: [(s, e, v) for s, e, v, _ in row] for b, row in bars.items()}))
        speeds.append(rest)
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
                for start, end, value in member[3].get(bar, []):
                    ticks = [(t - bar * bar_parts) * ticks_per_beat / PARTS_PER_BEAT for t in (start, end)]
                    notes.append((start, dict(pitches=[MOD_SLOTS - 1 - slot_index],
                                              points=[dict(tick=t, pitchBend=0, volume=value, forMod=True) for t in ticks])))
            if not notes:
                sequence.append(0)
                continue
            pattern = dict(instruments=[1], notes=[n for _, n in sorted(notes, key=lambda n: n[0])])
            key = json.dumps(pattern, sort_keys=True)
            if key not in lookup:
                patterns.append(pattern)
                lookup[key] = len(patterns)
            sequence.append(lookup[key])
        channels.append(dict(type="mod", name=f"Vibrato {len(channels) + 1}", instruments=[instrument], patterns=patterns, sequence=sequence))
    return channels, speeds


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
    played["order_starts"] = {k: t + lead_in for k, t in played["order_starts"].items()}
    played["length"] += lead_in
    played["loop_start"] += lead_in
    return lead_in


def custom_samples(song, shifts):
    """One embedded sample per (instrument, sample) that gets played."""
    urls = []
    for index, sample_index in sorted(shifts):
        sample = song["instruments"][index]["samples"][sample_index]
        root_key = ROOT_KEY + shifts[(index, sample_index)] - sample["finetune"] / 128
        urls.append(wav_data_url(sample, wave_name(index, sample_index), root_key))
    return urls


def describe_instrument(instrument):
    index, sample_index, offset = instrument
    text = str(index)
    if sample_index:
        text += f" sample {sample_index + 1}"
    if offset:
        text += f" from frame {offset}"
    return text


def convert(path, output_dir, beats_per_bar):
    song = parse_xm(path)
    played = simulate(song)
    bar_parts = beats_per_bar * PARTS_PER_BEAT
    lead_in = add_lead_in(played, bar_parts)
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
    slots = assign_channels(by_xm, bar_parts)
    bar_count = math.ceil(round(played["length"]) / bar_parts)
    for instrument in song["instruments"].values():
        if instrument["auto_vibrato"]:
            instrument["auto_settings"] = auto_vibrato_settings(instrument["auto_vibrato"], played["base_bpm"], played["parts_per_tick"])
    mod_channels, vibrato_speeds = build_mod_channels(slots, song, bar_count, ticks_per_beat, bar_parts)
    channels = [build_channel(s, song, shifts, bar_count, ticks_per_beat, bar_parts, v) for s, v in zip(slots, vibrato_speeds)]
    if len(channels) > MAX_CHANNELS or len(mod_channels) > MAX_CHANNELS:
        raise SystemExit(f"{path.name} needs {len(channels)} note and {len(mod_channels)} modulation channels; UltraBox allows {MAX_CHANNELS} of each")
    channels += mod_channels
    intro = int(played["loop_start"] // bar_parts)
    samples = custom_samples(song, shifts)
    result = {"format": "UltraBox", "version": 5, "name": song["title"] or path.stem, "scale": "Free", "key": "C",
              "introBars": intro, "loopBars": bar_count - intro, "beatsPerBar": beats_per_bar,
              "ticksPerBeat": ticks_per_beat, "beatsPerMinute": round(played["tempo"]),
              "layeredInstruments": False, "patternInstruments": True, "customSamples": samples, "channels": channels}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{path.stem}.ultrabox.json"
    json_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")

    slid = [v for c in by_xm for v, _ in c if any(o for _, o in v.path)]
    fine_only = sum(1 for v in slid if max(abs(o) for _, o in v.path) < 0.5)
    played_instruments = sorted({v.instrument[0] for c in by_xm for v, _ in c})
    pan_enveloped = [i for i in played_instruments if song["instruments"][i]["pan_envelope"]]
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
             f"{len(channels) - len(mod_channels)} note channels and {len(mod_channels)} vibrato modulation channels (from {song['channels']} XM channels)",
             f"{note_total} notes kept, {dropped} dropped because they were silent or shorter than one part",
             f"{pattern_total} unique patterns across {used_bars} used channel-bars ({1 - pattern_total / max(1, used_bars):.0%} reuse)",
             f"{aligned} of {len(played['order_starts'])} XM song positions start exactly on a bar line ({lead_in / PARTS_PER_BEAT:g} beats of silent lead-in added)",
             f"{len(samples)} samples embedded, {sum(len(u) for u in samples) / 1e6:.2f} MB in the song file, {len(variants)} instrument variants", "",
             "Channel map (XM instrument numbers, in the same order as each channel's instrument list):"]
    for slot in slots:
        merged = f" (also holds {', '.join(slot['merged'])})" if slot.get("merged") else ""
        lines.append(f"  {slot['name']:<8} instruments {', '.join(describe_instrument(i) for i in slot['instruments'])}{merged}")
    shifted = {f"{wave_name(*k)}": s for k, s in shifts.items() if s}
    if shifted:
        lines += ["", f"Moved by octaves to fit the pitch range: {shifted}"]
    if pan_enveloped:
        lines += ["", f"Panning envelopes not carried over, on instruments: {', '.join(map(str, pan_enveloped))}"]
    if swept:
        lines += ["", f"Auto-vibrato sweep is a delay instead of a gradual fade-in, on instruments: {', '.join(map(str, swept))}"]
    if other_waves:
        lines += ["", f"Auto-vibrato uses a sine instead of another waveform, on instruments: {', '.join(map(str, other_waves))}"]
    if fine_only:
        lines += ["", f"{fine_only} notes have fine pitch shifts under half a semitone, which BeepBox's whole-semitone bends can't hold"]
    lines += ["", "Effects not carried over:"]
    lines += [f"  {count:>6}  {name}" for name, count in played["ignored"].most_common()] or ["  none"]
    report_path = output_dir / f"{path.stem}.report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xm", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("output"))
    parser.add_argument("--beats-per-bar", type=int, default=16)
    args = parser.parse_args()
    convert(args.xm, args.output, args.beats_per_bar)


if __name__ == "__main__":
    main()

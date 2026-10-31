#!/usr/bin/env python3
"""Convert a FastTracker II .xm module into a regular BeepBox song, which is an approximation.

BeepBox can't play samples, so each sample is imitated by a built-in instrument: harmonics for
a tonal sound, spectrum for a noisy one, and a drumset slot for a short hit. Each note's volume
follows its sample's loudness, and the quietest parts are left out to fit BeepBox's channels.
The report written next to the song lists how it went.

Usage: python xm2beepbox.py song.xm [-o output_dir]
"""
import argparse
import collections
import json
import math
import pathlib

import numpy as np

import xm2box

MAX_PITCH_CHANNELS = 10
MAX_NOISE_CHANNELS = 5
MAX_BARS = 128
MAX_TEMPO = 300
MIN_TEMPO = 30
MAX_PITCH = 84
NOTE_SIZE_MAX = 3
QUIETEST_VOLUME = 6  # BeepBox's instrument volume runs from 0 down to this, in steps of 3 dB, and the next step is silent
VOLUME_STEP_LOG2 = 0.5
C4_HZ = 261.6256
BEEPBOX_PITCH_OF_C4 = 48
ANALYSIS_FRAMES = 8192
MIN_ANALYSIS_FRAMES = 2048
LOUDNESS_WINDOW = 256  # frames, for following a sample's loudness
CURVE_STEP = 2  # parts between loudness points on a note
SILENCE_SHARE = 0.015  # a one-shot sample has ended once it's this far below its peak
MAX_HARMONICS = 64
CONTROL_POINTS_HARMONICS = 28
CONTROL_POINTS_SPECTRUM = 30
LEVELS = 7  # the highest control value for harmonics and spectrum points
DRUM_SLOTS = 12
DRUM_BASE_PITCH = 24
SPECTRUM_POINTS_PER_OCTAVE = 7
SPECTRUM_TWEAK = (0, 1 / 7, math.log2(5 / 4), 3 / 7, math.log2(3 / 2), 5 / 7, 6 / 7)  # BeepBox nudges these points onto harmonic intervals
SPECTRUM_WAVE_LENGTH = 1 << 15
TONAL_SHARE = 0.5  # the share of a sample's power on a harmonic series that makes it tonal
DRUM_MAX_PITCHES = 4
DRUM_MAX_SECONDS = 0.45  # a short one-shot on only a few pitches can be a drum even if it has a tone
PITCH_DAMPING_HARMONICS = 48
PITCH_DAMPING_SPECTRUM = 28
PITCH_DAMPING_DRUMS = 48
PITCH_REFERENCE_HARMONICS = 16
PITCH_REFERENCE_SPECTRUM = 24
# How loud each instrument type comes out at full note size and volume 0, given its wave's calculated level. Measured from BeepBox's synth.
HARMONICS_UNIT = 0.253
SPECTRUM_UNIT = 1.87
DRUM_UNIT = 2.45
GAIN_PERCENTILE = 90  # an instrument's full note size is set near its loudest notes
MIX_TARGET_RMS = 0.13  # about where FT2's render of these songs sits
MIX_MAX_GAIN = 1e3
SAME_VOLUME_RANGE = 3  # drums within this many volume steps can share a channel's instrument volume
PAN_STEP = 25  # BeepBox can pan further in between, but more positions would split a sound across more channels
PAN_SWEEP_MIN = 5  # how far a pan envelope must swing, out of xm2box.MAX_PAN, to count as moving
VIBRATO_PRESETS = (("light", 0.15), ("heavy", 0.45))
VIBRATO_MIN_SHARE = 0.25  # the share of a note with vibrato before its instrument needs vibrato
MIN_DETUNE_CENTS = 4
SEMITONES_PER_RAW_VIBRATO = 1 / xm2box.VIBRATO_RAW_PER_SEMITONE


def to_mono_values(sample):
    width = 32768 if sample["wide"] else 128
    return np.array(xm2box.as_values(xm2box.sample_audio(sample), sample["wide"]), dtype=float) / width


def loudness_curve(values):
    """A sample's loudness over its length, one value for each window."""
    windows = max(1, len(values) // LOUDNESS_WINDOW)
    trimmed = values[:windows * LOUDNESS_WINDOW].reshape(windows, -1) if len(values) >= LOUDNESS_WINDOW else values.reshape(1, -1)
    return np.sqrt((trimmed ** 2).mean(axis=1))


def average_spectrum(values):
    """A power spectrum at 8363 frames per second, scaled so that its sum is the signal's power."""
    size = ANALYSIS_FRAMES
    if len(values) < size:
        values = np.concatenate([values, np.zeros(size - len(values))]) if len(values) >= MIN_ANALYSIS_FRAMES else np.resize(values, size)
    window = np.hanning(size)
    power = np.zeros(size // 2 + 1)
    count = 0
    for at in range(0, len(values) - size + 1, size // 2):
        power += np.abs(np.fft.rfft(values[at:at + size] * window)) ** 2
        count += 1
    power *= 2 / (size * (window ** 2).sum() * max(1, count))
    power[0] = 0
    return power


def harmonic_share(power, f0):
    """How much of a spectrum's power sits on the harmonics of f0."""
    hertz_per_bin = xm2box.SAMPLE_RATE / ANALYSIS_FRAMES
    total = power.sum()
    if total <= 0:
        return 0.0
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    kept = 0.0
    last = len(power) - 1
    for k in range(1, MAX_HARMONICS + 1):
        centre = k * f0 / hertz_per_bin
        if centre >= last:
            break
        reach = max(2.0, 0.004 * centre)
        kept += cumulative[min(last + 1, int(centre + reach) + 1)] - cumulative[max(0, int(centre - reach))]
    return kept / total


def find_fundamental(power):
    """The highest pitch whose harmonics hold nearly as much of the sound as any lower pitch's do."""
    candidates = 20 * 2 ** (np.arange(0, 6.5 * 48) / 48)
    candidates = candidates[candidates < xm2box.SAMPLE_RATE / 4]
    shares = np.array([harmonic_share(power, f) for f in candidates])
    best = shares.max() if len(shares) else 0.0
    if best <= 0:
        return 0.0, 0.0
    index = max(i for i, s in enumerate(shares) if s >= 0.85 * best)
    while index + 1 < len(shares) and shares[index + 1] > shares[index]:
        index += 1
    fine = candidates[index] * 2 ** (np.linspace(-1, 1, 41) / 48)
    fine_shares = [harmonic_share(power, f) for f in fine]
    pick = int(np.argmax(fine_shares))
    return float(fine[pick]), float(fine_shares[pick])


def harmonic_level(value):
    """How strong BeepBox makes a harmonic from its control value."""
    return 2.0 ** (value - LEVELS + 1) * math.sqrt(value / LEVELS) if value > 0 else 0.0


def spectrum_level(value):
    share = value / LEVELS
    return 2.0 ** ((share - 1) * LEVELS + 1) * share if value > 0 else 0.0


def nearest_control(wanted, level):
    """The control value whose level is closest to what's wanted, on a log scale."""
    if wanted <= level(1) / math.sqrt(2):
        return 0
    return min(range(1, LEVELS + 1), key=lambda v: abs(math.log(level(v) / wanted)))


def fit_harmonics(power, f0):
    """Turn a spectrum into BeepBox harmonics values, with how loud the resulting wave comes out."""
    hertz_per_bin = xm2box.SAMPLE_RATE / ANALYSIS_FRAMES
    last = len(power) - 1
    strengths = []
    for k in range(1, CONTROL_POINTS_HARMONICS + 1):
        centre = k * f0 / hertz_per_bin
        if centre >= last:
            strengths.append(0.0)
            continue
        reach = max(2.0, 0.004 * centre)
        strengths.append(math.sqrt(2 * power[max(0, int(centre - reach)):int(centre + reach) + 1].sum()))
    wanted = [s * (k + 1) ** 0.25 for k, s in enumerate(strengths)]  # BeepBox tilts harmonics down by this
    top = max(wanted) or 1.0
    controls = [nearest_control(w / top * harmonic_level(LEVELS), harmonic_level) for w in wanted]
    return controls, harmonics_wave_level(controls)


def harmonics_wave_level(controls):
    """Calculate the level of BeepBox's harmonics wave, as its code builds it."""
    combined = 1.0
    total = 0.0
    for index in range(MAX_HARMONICS):
        value = controls[index] if index < CONTROL_POINTS_HARMONICS else controls[-1] * (1 - (index - CONTROL_POINTS_HARMONICS) / (MAX_HARMONICS - CONTROL_POINTS_HARMONICS))
        amplitude = 2.0 ** (value - LEVELS + 1) * math.sqrt(value / LEVELS) if value > 0 else 0.0
        if index < CONTROL_POINTS_HARMONICS:
            combined += amplitude
        total += (amplitude * (index + 1) ** -0.25) ** 2 / 2
    return math.sqrt(total) / combined ** 0.7


def spectrum_point_octave(point):
    return math.floor(point / SPECTRUM_POINTS_PER_OCTAVE) + SPECTRUM_TWEAK[point % SPECTRUM_POINTS_PER_OCTAVE]


def fit_spectrum(power, playback, base_hertz):
    """Turn a spectrum, played back this many times faster, into BeepBox spectrum values above a base frequency."""
    hertz = np.arange(len(power)) * xm2box.SAMPLE_RATE / ANALYSIS_FRAMES * playback
    wanted = []
    for point in range(CONTROL_POINTS_SPECTRUM):
        low = base_hertz * 2 ** (spectrum_point_octave(point) - 0.5 / SPECTRUM_POINTS_PER_OCTAVE)
        high = base_hertz * 2 ** (spectrum_point_octave(point) + 0.5 / SPECTRUM_POINTS_PER_OCTAVE)
        inside = (hertz >= low) & (hertz < high)
        if not inside.any():
            nearest = int(np.argmin(np.abs(hertz - (low + high) / 2)))
            if nearest == 0 or abs(hertz[nearest] - (low + high) / 2) > (high - low):
                wanted.append(0.0)
                continue
            inside = np.zeros(len(power), dtype=bool)
            inside[nearest] = True
        density = power[inside].mean()
        wanted.append(math.sqrt(density) * math.sqrt((low + high) / 2))  # BeepBox tilts these down by the root of the frequency
    top = max(wanted) or 1.0
    return [nearest_control(w / top * spectrum_level(LEVELS), spectrum_level) for w in wanted]


def spectrum_wave_level(controls, lowest_octave):
    """Calculate the level of BeepBox's spectrum wave, as its code builds it."""
    combined = 1.0
    total = 0.0

    def octave_of(point):
        return lowest_octave + spectrum_point_octave(point)

    spans = []
    for i in range(CONTROL_POINTS_SPECTRUM + 1):
        value1 = 0 if i <= 0 else controls[i - 1]
        value2 = controls[-1] if i >= CONTROL_POINTS_SPECTRUM else controls[i]
        low, high = octave_of(i - 1), octave_of(i)
        if i >= CONTROL_POINTS_SPECTRUM:
            high = 14 + (high - 14) * 0.25
        spans.append((value1, value2, low, high))
    if controls[-1] > 0:
        spans.append((controls[-1], 0, 14 + (octave_of(CONTROL_POINTS_SPECTRUM) - 14) * 0.25, 14))
    for value1, value2, low, high in spans:
        if value1 == 0 and value2 == 0:
            continue
        first = int(2 ** low)
        last = min(SPECTRUM_WAVE_LENGTH >> 1, int(2 ** high))
        if last <= first or high <= low:
            continue
        bins = np.arange(first, last, dtype=float)
        shares = (value1 + (value2 - value1) * (np.log2(bins) - low) / (high - low)) / LEVELS
        amplitudes = 2.0 ** ((shares - 1) * LEVELS + 1) * shares * (bins / 2048) ** -0.5
        combined += 0.02 * amplitudes.sum()
        total += (amplitudes ** 2).sum() / 2
    return 5.0 / (math.sqrt(SPECTRUM_WAVE_LENGTH) * combined ** 0.75) * math.sqrt(total)


def drum_base_hertz(slot):
    return 440.0 * 2 ** ((DRUM_BASE_PITCH + slot * 6 - 69) / 12)


def drum_lowest_octave(slot):
    return 15 + math.log2(drum_base_hertz(slot) / 44100)


def best_drum_slot(power, playback):
    """The drumset slot whose frequency range covers the most of a sound's power."""
    hertz = np.arange(len(power)) * xm2box.SAMPLE_RATE / ANALYSIS_FRAMES * playback
    def covered(slot):
        low = drum_base_hertz(slot)
        high = low * 2 ** spectrum_point_octave(CONTROL_POINTS_SPECTRUM - 1)
        return power[(hertz >= low) & (hertz < high)].sum()
    scores = [covered(slot) for slot in range(DRUM_SLOTS)]
    top = max(scores) or 1.0
    # Of the slots that cover it about as well, the lowest one gives a low sound the most detail.
    return next(slot for slot in range(DRUM_SLOTS) if scores[slot] >= 0.98 * top)


class SampleSound:
    """What a sample sounds like and how BeepBox will imitate it."""

    def __init__(self, sample, voices):
        self.sample = sample
        self.values = to_mono_values(sample)
        self.curve = loudness_curve(self.values)
        self.pitches = collections.Counter(v.pitch for v in voices)
        median_pitch = sorted(self.pitches.elements())[sum(self.pitches.values()) // 2]
        self.playback_rate = lambda pitch: 2 ** ((pitch - BEEPBOX_PITCH_OF_C4 + sample["finetune"] / 128) / 12)
        seconds = sorted(v.end - v.start for v in voices)[len(voices) // 2] / max(1e-9, voices[0].parts_per_second)
        frames = int(seconds * xm2box.SAMPLE_RATE * self.playback_rate(median_pitch))
        region = self.values[:max(ANALYSIS_FRAMES, frames)] if sample["loop"] == xm2box.LOOP_NONE else self.values[:max(ANALYSIS_FRAMES * 2, frames)]
        self.power = average_spectrum(region)
        self.f0, self.share = find_fundamental(self.power)
        self.seconds = len(self.values) / xm2box.SAMPLE_RATE / self.playback_rate(median_pitch)
        bendy = sum(len(v.path) > 1 for v in voices) / len(voices)
        one_shot = sample["loop"] == xm2box.LOOP_NONE
        few_pitches = len(self.pitches) <= DRUM_MAX_PITCHES and bendy < 0.2
        if one_shot and few_pitches and (self.share < TONAL_SHARE or self.seconds < DRUM_MAX_SECONDS):
            self.kind = "drum"
        elif self.share >= TONAL_SHARE:
            self.kind = "harmonics"
        else:
            self.kind = "spectrum"
        self.median_pitch = median_pitch
        self.semitones = 0
        self.cents = 0
        if self.kind == "harmonics":
            exact = sample["finetune"] / 128 + 12 * math.log2(self.f0 / C4_HZ)
            self.semitones = round(exact)
            self.cents = round((exact - self.semitones) * 100)
            self.controls, self.wave_level = fit_harmonics(self.power, self.f0)
        elif self.kind == "spectrum":
            # A noisy sound has no pitch to match, so its lowest strong frequency sets where its spectrum starts.
            hertz = np.arange(len(self.power)) * xm2box.SAMPLE_RATE / ANALYSIS_FRAMES * self.playback_rate(median_pitch)
            running = np.cumsum(self.power) / max(self.power.sum(), 1e-12)
            low = max(32.7, float(hertz[int(np.searchsorted(running, 0.02))]))
            target = round(BEEPBOX_PITCH_OF_C4 + 12 * math.log2(low / C4_HZ))
            self.semitones = target - median_pitch
            base = C4_HZ * 2 ** ((target - BEEPBOX_PITCH_OF_C4) / 12)
            self.controls = fit_spectrum(self.power, self.playback_rate(median_pitch), base)
            self.wave_level = spectrum_wave_level(self.controls, 8)
        self.reference = 0.0  # the note level that gets full note size, set once all notes are known

    def loudness_at(self, frame):
        """A sample's loudness at some frame, following its loop."""
        sample = self.sample
        if sample["loop"] != xm2box.LOOP_NONE and frame >= sample["loop_end"]:
            start, end = sample["loop_start"], sample["loop_end"]
            length = max(1, end - start)
            passed = (frame - start) % (length * (2 if sample["loop"] == xm2box.LOOP_PING_PONG else 1))
            frame = start + (passed if passed < length else 2 * length - passed)
        return self.curve[min(len(self.curve) - 1, int(frame // LOUDNESS_WINDOW))]

    def ends_at(self):
        """The frame where a one-shot sample has gone quiet."""
        loud = np.nonzero(self.curve >= self.curve.max() * SILENCE_SHARE)[0]
        return (int(loud[-1]) + 1) * LOUDNESS_WINDOW if len(loud) else len(self.values)


def describe_sound(key, sound):
    name = xm2box.wave_name(*key)
    if sound.kind == "drum":
        return f"  {name}: drum, {len(sound.pitches)} pitch(es), harmonic share {sound.share:.0%}"
    if sound.kind == "spectrum":
        return f"  {name}: spectrum (noisy), harmonic share {sound.share:.0%}"
    return f"  {name}: harmonics at {sound.f0:.1f} Hz, harmonic share {sound.share:.0%}"


def follow_sample_loudness(voice, sound):
    """Add a sample's own loudness to a note's volume shape."""
    offset = voice.instrument[2] if len(voice.instrument) > 2 else 0
    rate = xm2box.SAMPLE_RATE * sound.playback_rate(voice.pitch) / voice.parts_per_second
    shape = []
    time = voice.start
    while time < voice.end:
        shape.append((time, sound.loudness_at(offset + (time - voice.start) * rate)))
        time += CURVE_STEP
    shape.append((voice.end, sound.loudness_at(offset + (voice.end - voice.start) * rate)))
    if voice.envelope:
        points = sorted(dict(voice.envelope).items())
        shape = [(t, level * xm2box._value_at(points, t)) for t, level in shape]
    voice.envelope = shape


def vibrato_depth(voice, auto_settings):
    """How deep a note's vibrato mostly is, in semitones, if it has it for long enough to matter."""
    length = max(1e-9, voice.end - voice.start)
    amount = sum(b - a for a, b, _, _ in voice.vibrato) / length
    depth = 0.0
    if amount >= VIBRATO_MIN_SHARE:
        depth = sum((b - a) * d for a, b, _, d in voice.vibrato) / max(1e-9, sum(b - a for a, b, _, _ in voice.vibrato)) * SEMITONES_PER_RAW_VIBRATO
    elif auto_settings:
        depth = auto_settings["depth"]
    return depth


def main_vibrato(voices, auto_settings):
    """A sample gets one vibrato besides none, the one its notes spend the longest near, so its notes can share channels."""
    time = collections.Counter()
    for voice in voices:
        depth = vibrato_depth(voice, auto_settings)
        if depth >= 0.5 * VIBRATO_PRESETS[0][1]:
            time[min(VIBRATO_PRESETS, key=lambda p: abs(p[1] - depth))[0]] += voice.end - voice.start
    return time.most_common(1)[0][0] if time else "none"


def pan_position(voice, sample):
    """A note's fixed place in the stereo field. Pan that sweeps can't, so odd and even XM channels go to opposite sides of it."""
    if voice.pan:
        values = [v for _, v in voice.pan]
        middle = (sum(values) / len(values) - xm2box.MAX_PAN / 2) * 2
        swing = (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5 * 2
        if swing >= PAN_SWEEP_MIN:
            middle += swing if voice.xm_channel % 2 else -swing
    else:
        middle = (sample["pan"] / 255 * xm2box.MAX_PAN - xm2box.MAX_PAN / 2) * 2
    return max(-100, min(100, round(middle / PAN_STEP) * PAN_STEP))


def merge_doubles(voices):
    """Two channels playing one note the same way are one note in BeepBox. A little louder, since FT2 detunes them to a chorus."""
    first = {}
    kept = []
    for voice in sorted(voices, key=lambda v: v.start):
        key = (voice.instrument, voice.pitch, round(voice.start))
        twin = first.get(key)
        if twin is not None and abs(twin.end - voice.end) <= 1:
            mine = max((v for _, v in voice.points), default=0)
            other = max((v for _, v in twin.points), default=0)
            scale = math.sqrt(other ** 2 + mine ** 2) / other if other else 1.0
            twin.points = [(t, min(64.0 * 1.5, v * scale)) for t, v in twin.points]
            continue
        first[key] = voice
        kept.append(voice)
    return kept


def size_of(level, reference):
    """The note size that gives a level, since BeepBox raises note size to the power 1.5."""
    return NOTE_SIZE_MAX * (max(0.0, level) / reference) ** (2 / 3) if reference > 0 else 0.0


def note_points(piece, reference, bar_start, closes=False):
    """Write a note's points on BeepBox's 4 note sizes, placed where its smooth volume crosses one so that fades stay smooth.

    A note that closes ends at silence. A drumset rings on for a whole beat after a note that doesn't.
    """
    sized = [(t, min(float(NOTE_SIZE_MAX), size_of(level, reference)), bend) for t, level, bend in piece]
    if max(s for _, s, _ in sized) < 0.3:
        return None
    floor = 1 if min(round(s) for _, s, _ in sized) == 0 and max(round(s) for _, s, _ in sized) == 0 else 0  # still audible, so it can't round to silence
    points = []

    def add(time, size, bend):
        time = int(round(time))
        size = max(floor, int(round(size)))
        bend = int(round(bend))
        if points and time <= points[-1][0]:
            return
        if len(points) >= 2 and points[-1][1:] == (size, bend) and points[-2][1:] == (size, bend):
            points.pop()
        points.append((time, size, bend))

    for (t0, s0, b0), (t1, s1, b1) in zip(sized, sized[1:]):
        add(t0, s0, b0)
        low, high = sorted((s0, s1))
        crossings = range(math.ceil(low), math.floor(high) + 1)
        for k in (crossings if s1 > s0 else reversed(crossings)):
            if k not in (s0, s1) and s1 != s0:
                share = (k - s0) / (s1 - s0)
                add(t0 + share * (t1 - t0), k, b0 + share * (b1 - b0))
    last = sized[-1]
    time = int(round(last[0]))
    if len(points) >= 2 and time <= points[-1][0]:
        points.pop()  # the end takes the place of a point on the same part, since going past it could overlap the next note
    points.append((time, 0 if closes else max(floor, int(round(last[1]))), int(round(last[2]))))
    if len(points) < 2 or all(p[1] == 0 for p in points):
        return None
    return [dict(tick=t - bar_start, pitchBend=b, volume=round(s * 100 / NOTE_SIZE_MAX)) for t, s, b in points]


def percentile_level(levels, weights):
    order = np.argsort(levels)
    running = np.cumsum(np.array(weights)[order])
    return float(np.array(levels)[order][int(np.searchsorted(running, running[-1] * GAIN_PERCENTILE / 100))])


def fold_pitch(pitch):
    """Move a note by octaves into BeepBox's range. Going up or down is better than going silent."""
    while pitch > MAX_PITCH:
        pitch -= 12
    while pitch < 0:
        pitch += 12
    return pitch


def volume_for(json_level):
    """Write BeepBox's instrument volume, which it reads as 5 - value / 20."""
    return (5 - json_level) * 20


def tonal_instrument(key, sound, volume, arpeggio):
    _, _, pan, vibrato = key
    effects = []
    result = dict(type=sound.kind, volume=volume_for(volume), eqFilter=[], envelopes=[], fadeInSeconds=0, fadeOutTicks=-3, transition="normal", chord="simultaneous")
    if pan:
        effects.append("panning")
        result["pan"] = pan
    if abs(sound.cents) >= MIN_DETUNE_CENTS:
        effects.append("detune")
        result["detuneCents"] = sound.cents
    if vibrato != "none":
        effects.append("vibrato")
        result["vibrato"] = vibrato
    if arpeggio:
        effects.append("chord type")
        result["chord"] = "arpeggio"
    result["effects"] = effects
    points = [round(100 * c / LEVELS) for c in sound.controls]
    if sound.kind == "harmonics":
        result.update(harmonics=points, unison="none")
    else:
        result["spectrum"] = points
    return result


def pitch_expression(pitch, sound):
    """BeepBox plays higher notes more quietly by itself."""
    midi = pitch + 12
    if sound.kind == "harmonics":
        return 2.0 ** (-(midi - PITCH_REFERENCE_HARMONICS) / PITCH_DAMPING_HARMONICS)
    return 2.0 ** (-(midi - PITCH_REFERENCE_SPECTRUM) / PITCH_DAMPING_SPECTRUM)


def unit_level(sound):
    """How loud a sound's instrument is at its typical pitch, at full note size and volume 0."""
    expression = pitch_expression(sound.median_pitch + sound.semitones, sound)
    return (HARMONICS_UNIT if sound.kind == "harmonics" else SPECTRUM_UNIT) * sound.wave_level * expression


def drum_unit_level(controls, slot):
    expression = 2.0 ** (-(slot * 6) / PITCH_DAMPING_DRUMS)
    return DRUM_UNIT * spectrum_wave_level(controls, drum_lowest_octave(slot)) * expression


def assign_drums(voices, sounds, bar_parts):
    """Share out drums between the noise channels. Each channel is one drumset, so its drums can all play in one bar.

    A channel plays one hit at a time, so drums go to a channel where they rarely overlap and suit its volume. A new hit cuts off the last.
    """
    groups = collections.defaultdict(list)
    for voice in voices:
        groups[(voice.instrument[0], voice.instrument[1], voice.pitch)].append(voice)
    length = int(max((v.end for v in voices), default=0)) + 2
    drums = {}
    for drum, members in groups.items():
        sound = sounds[drum[:2]]
        playback = sound.playback_rate(drum[2])
        slot = best_drum_slot(sound.power, playback)
        controls = fit_spectrum(sound.power, playback, drum_base_hertz(slot))
        levels = [max((v for _, v, _ in piece), default=0) for voice in members for _, piece, _ in voice.pieces]
        loudest = percentile_level(levels, [1] * len(levels)) if levels else 0.0
        drums[drum] = dict(slot=slot, controls=controls, loudest=loudest, voices=members,
                           energy=sum(xm2box.note_energy(piece) for voice in members for _, piece, _ in voice.pieces))
    channels = []
    for drum, info in sorted(drums.items(), key=lambda d: -d[1]["energy"]):
        need = info["loudest"] / max(1e-12, drum_unit_level(info["controls"], info["slot"]))
        best, best_clash = None, math.inf
        for channel in channels:
            if len(channel["drums"]) >= DRUM_SLOTS:
                continue
            if abs(2 * math.log2(max(need, 1e-9) / max(channel["need"], 1e-9))) > SAME_VOLUME_RANGE and len(channels) < MAX_NOISE_CHANNELS:
                continue
            clash = sum(int(channel["busy"][int(v.start):int(v.end) + 1].sum()) for v in info["voices"])
            if clash < best_clash and (clash == 0 or len(channels) >= MAX_NOISE_CHANNELS):
                best, best_clash = channel, clash
        if best is None:
            if len(channels) < MAX_NOISE_CHANNELS:
                best = dict(drums={}, busy=np.zeros(length, dtype=bool), need=need)
                channels.append(best)
            else:
                best = min((c for c in channels if len(c["drums"]) < DRUM_SLOTS), key=lambda c: sum(int(c["busy"][int(v.start):int(v.end) + 1].sum()) for v in info["voices"]), default=None)
                if best is None:
                    continue
        taken = {d["slot"] for d in best["drums"].values()}
        info["slot_in_channel"] = min((s for s in range(DRUM_SLOTS) if s not in taken), key=lambda s: abs(s - info["slot"]))
        if info["slot_in_channel"] != info["slot"]:
            info["slot"] = info["slot_in_channel"]
            info["controls"] = fit_spectrum(sounds[drum[:2]].power, sounds[drum[:2]].playback_rate(drum[2]), drum_base_hertz(info["slot"]))
        best["drums"][drum] = info
        best["need"] = max(best["need"], info["loudest"] / max(1e-12, drum_unit_level(info["controls"], info["slot"])))
        for voice in info["voices"]:
            best["busy"][int(voice.start):int(voice.end) + 1] = True
    return channels


def drum_channel_json(channel, name, bar_parts, bar_count, gain):
    """Build a noise channel holding one drumset, with each drum in its slot."""
    drums = [dict(filterEnvelope="none", spectrum=[0] * CONTROL_POINTS_SPECTRUM) for _ in range(DRUM_SLOTS)]
    units = {}
    for drum, info in channel["drums"].items():
        drums[info["slot"]]["spectrum"] = [round(100 * c / LEVELS) for c in info["controls"]]
        units[drum] = drum_unit_level(info["controls"], info["slot"])
    volume_steps = max(0, min(QUIETEST_VOLUME, round(-2 * math.log2(max(1e-12, channel["need"] * gain)))))
    scale = 2.0 ** (-volume_steps * VOLUME_STEP_LOG2)
    hits = []
    for drum, info in channel["drums"].items():
        for voice in info["voices"]:
            hits.append((voice.start, drum, voice))
    hits.sort(key=lambda h: h[0])
    for (_, _, voice), following in zip(hits, hits[1:]):
        voice.end = min(voice.end, following[0])
    by_bar = collections.defaultdict(list)
    placed = 0
    for _, drum, voice in hits:
        pieces = xm2box.build_note_pieces(voice, bar_parts)
        for index, (bar, piece, continues) in enumerate(pieces):
            reference = units[drum] * scale / gain
            points = note_points(piece, reference, bar * bar_parts, closes=index == len(pieces) - 1)
            if not points:
                continue
            note = dict(pitches=[channel["drums"][drum]["slot"]], points=points)
            if points[0]["tick"] == 0:
                note["continuesLastPattern"] = continues
            by_bar[bar].append(note)
            placed += 1
    patterns, lookup, sequence = [], {}, []
    for bar in range(bar_count):
        notes = by_bar.get(bar)
        if not notes:
            sequence.append(0)
            continue
        pattern = dict(instruments=[1], notes=notes)
        key = json.dumps(pattern, sort_keys=True)
        if key not in lookup:
            patterns.append(pattern)
            lookup[key] = len(patterns)
        sequence.append(lookup[key])
    instrument = dict(type="drumset", volume=volume_for(volume_steps), eqFilter=[], effects=[], envelopes=[], fadeInSeconds=0, fadeOutTicks=-3, transition="normal", chord="simultaneous", drums=drums)
    return dict(type="drum", name=name, instruments=[instrument], patterns=patterns, sequence=sequence, octaveScrollBar=0), placed


def pitch_channel_json(slot, sounds, gain, volumes, bar_parts, bar_count, song):
    instruments = slot["instruments"]
    patterns, lookup, sequence = [], {}, []
    pitches = []
    placed = 0
    for bar in range(bar_count):
        pieces = sorted(slot["notes"].get(bar, []), key=lambda n: n[1][0][0])
        if not pieces:
            sequence.append(0)
            continue
        notes = []
        for voice, piece, continues in pieces:
            sound = sounds[voice.instrument[:2]]
            pitch = fold_pitch(voice.pitch + sound.semitones)
            scale = 2.0 ** (-volumes[voice.instrument[:2]] * VOLUME_STEP_LOG2) * pitch_expression(pitch, sound) / pitch_expression(sound.median_pitch + sound.semitones, sound)
            reference = unit_level(sound) * scale / gain
            points = note_points(piece, reference, bar * bar_parts)
            if not points:
                continue
            note = dict(pitches=[pitch] + [pitch + i for i in voice.chord if pitch + i <= MAX_PITCH], points=points)
            if points[0]["tick"] == 0:
                note["continuesLastPattern"] = continues
            notes.append(note)
            pitches.append(pitch)
            placed += 1
        if not notes:
            sequence.append(0)
            continue
        pattern = dict(instruments=[instruments.index(pieces[0][0].instrument) + 1], notes=notes)
        key = json.dumps(pattern, sort_keys=True)
        if key not in lookup:
            patterns.append(pattern)
            lookup[key] = len(patterns)
        sequence.append(lookup[key])
    channel = dict(type="pitch", name=slot["name"], patterns=patterns, sequence=sequence,
                   instruments=[tonal_instrument(i, sounds[i[:2]], volumes[i[:2]], song["instruments"][i[0]].get("chord_arpeggio")) for i in instruments])
    if pitches:
        channel["octaveScrollBar"] = max(0, min(5, (sorted(pitches)[len(pitches) // 2] - 6) // 12)) - 1
    return channel, placed


def simulate_to_fit(song):
    """Play the song at a tempo BeepBox can hold, and at few enough bars. Half the tempo holds twice the song in a bar."""
    max_tempo = MAX_TEMPO
    while True:
        played = xm2box.simulate(song, max_tempo=max_tempo)
        beats = played["length"] / xm2box.PARTS_PER_BEAT
        for beats_per_bar in (4, 8, 16):
            if math.ceil((beats + beats_per_bar) / beats_per_bar) <= MAX_BARS:
                return played, beats_per_bar
        if played["tempo"] / 2 < MIN_TEMPO:
            return played, 16
        max_tempo = played["tempo"] / 2 + 1e-6


def convert(path, output_dir):
    song = xm2box.parse_xm(path)
    played, beats_per_bar = simulate_to_fit(song)
    bar_parts = beats_per_bar * xm2box.PARTS_PER_BEAT
    lead_in = xm2box.add_lead_in(played, bar_parts)
    parts_per_second = played["tempo"] * xm2box.PARTS_PER_BEAT / 60
    voices_by_sample = collections.defaultdict(list)
    all_voices = 0
    for xm_channel, channel in enumerate(played["notes"]):
        ordered = sorted(channel, key=lambda v: v.start)
        for voice, following in zip(ordered, ordered[1:] + [None]):
            if following is not None:
                voice.end = min(voice.end, following.start)
            voice.end = min(voice.end, played["length"])
            voice.parts_per_second = parts_per_second
            voice.xm_channel = xm_channel
            all_voices += 1
            if voice.end > voice.start:
                voices_by_sample[voice.instrument[:2]].append(voice)
    sounds = {key: SampleSound(song["instruments"][key[0]]["samples"][key[1]], voices) for key, voices in voices_by_sample.items()}
    for instrument in song["instruments"].values():
        if instrument["auto_vibrato"]:
            instrument["auto_settings"] = xm2box.auto_vibrato_settings(instrument["auto_vibrato"], played["base_bpm"], played["parts_per_tick"])

    tonal = collections.defaultdict(list)
    drum_voices = []
    for key, voices in voices_by_sample.items():
        sound = sounds[key]
        auto = song["instruments"][key[0]].get("auto_settings")
        vibrato = main_vibrato(voices, auto)
        for voice in voices:
            if sound.sample["loop"] == xm2box.LOOP_NONE:
                end_frame = sound.ends_at() - (voice.instrument[2] if len(voice.instrument) > 2 else 0)
                voice.end = min(voice.end, voice.start + max(1, end_frame) / (xm2box.SAMPLE_RATE * sound.playback_rate(voice.pitch)) * parts_per_second)
            follow_sample_loudness(voice, sound)
            if sound.kind == "drum":
                drum_voices.append(voice)
            else:
                has_vibrato = vibrato_depth(voice, auto) >= 0.5 * VIBRATO_PRESETS[0][1]
                voice.instrument = (key[0], key[1], pan_position(voice, sound.sample), vibrato if has_vibrato else "none")
                voice.pan = []
                tonal[key].append(voice)

    by_xm = []
    for key in sorted(tonal):
        kept = []
        for voice in merge_doubles(tonal[key]):
            pieces = xm2box.build_note_pieces(voice, bar_parts)
            if pieces:
                voice.pieces = pieces
                kept.append((voice, pieces))
        by_xm.append(kept)
    for voice in drum_voices:
        voice.pieces = xm2box.build_note_pieces(voice, bar_parts)
    drum_voices = [v for v in drum_voices if v.pieces]

    # An instrument's volume sets the level of its full note size, and the whole song's gain is its quietest fit.
    wanted = {}
    for kept in by_xm:
        if not kept:
            continue
        key = kept[0][0].instrument[:2]
        levels, weights = [], []
        for voice, pieces in kept:
            for _, piece, _ in pieces:
                for a, b in zip(piece, piece[1:]):
                    levels.append(max(a[1], b[1]))
                    weights.append(b[0] - a[0])
        wanted[key] = percentile_level(levels, weights) / unit_level(sounds[key])
    noise_channels = assign_drums(drum_voices, sounds, bar_parts)
    needs = list(wanted.values()) + [c["need"] for c in noise_channels]
    gain = min(MIX_MAX_GAIN, 1 / max(needs)) if needs else 1.0
    volumes = {key: max(0, min(QUIETEST_VOLUME, round(-2 * math.log2(max(1e-12, need * gain))))) for key, need in wanted.items()}

    slots, left_out = xm2box.assign_tracks(by_xm, max_channels=MAX_PITCH_CHANNELS)
    bar_count = min(MAX_BARS, math.ceil(round(played["length"]) / bar_parts))
    channels, tonal_notes = [], 0
    for slot in slots:
        channel, placed = pitch_channel_json(slot, sounds, gain, volumes, bar_parts, bar_count, song)
        channels.append(channel)
        tonal_notes += placed
    drum_notes = 0
    for index, channel in enumerate(noise_channels):
        built, placed = drum_channel_json(channel, f"Drums {index + 1}", bar_parts, bar_count, gain)
        channels.append(built)
        drum_notes += placed

    intro = min(bar_count - 1, int(played["loop_start"] // bar_parts))
    result = {"format": "BeepBox", "version": 9, "name": song["title"] or path.stem, "scale": "expert", "key": "C",
              "introBars": intro, "loopBars": max(1, bar_count - intro), "beatsPerBar": beats_per_bar, "ticksPerBeat": xm2box.PARTS_PER_BEAT,
              "beatsPerMinute": round(played["tempo"]), "layeredInstruments": False, "patternInstruments": True, "channels": channels}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{path.stem}.beepbox.json"
    json_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")

    total_pieces = sum(len(p) for kept in by_xm for _, p in kept)
    energy_kept = sum(xm2box.note_energy(item[1]) for slot in slots for items in slot["notes"].values() for item in items)
    energy_all = sum(xm2box.note_energy(piece) for kept in by_xm for _, pieces in kept for _, piece, _ in pieces)
    cut = max(0, math.ceil(round(played["length"]) / bar_parts) - bar_count)
    lines = [f"{path.name} -> {json_path.name}", "",
             "This is an approximation. BeepBox can't play samples, so the sounds are imitations.",
             f"Tempo {result['beatsPerMinute']}, {bar_count} bars of {beats_per_bar} beats ({lead_in / xm2box.PARTS_PER_BEAT:g} beats of silent lead-in added)",
             f"{len(slots)} pitch channels and {len(noise_channels)} noise channels (from {song['channels']} XM channels)",
             f"{left_out} of {total_pieces} tonal note pieces left out to fit {MAX_PITCH_CHANNELS} channels, {1 - energy_kept / max(1e-12, energy_all):.1%} of their sound energy",
             f"{tonal_notes} tonal and {drum_notes} drum note pieces placed", "", "How each sample is imitated:"]
    lines += [describe_sound(key, sounds[key]) for key in sorted(sounds)]
    if cut:
        lines += ["", f"WARNING: {cut} bars at the end were cut off to fit BeepBox's {MAX_BARS} bars"]
    lines += ["", "Effects not carried over:"]
    lines += [f"  {count:>6}  {name}" for name, count in played["ignored"].most_common()] or ["  none"]
    report_path = output_dir / f"{path.stem}.beepbox.report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xm", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("output"))
    args = parser.parse_args()
    convert(args.xm, args.output)


if __name__ == "__main__":
    main()

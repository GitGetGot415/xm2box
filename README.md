# XM2Box

Convert FastTracker II `.xm` files to UltraBox or BeepBox songs.

## Requirements

- Python 3
- Tkinter (usually included with Python)
- NumPy, only for BeepBox conversion:

```sh
python -m pip install numpy
```

## Run the app

From this folder, run:

```sh
python convert_xm.py
```

Choose an `.xm` file, select UltraBox or BeepBox, and click **Convert**. Converted files are saved in `output/`.

## Command line

Convert to UltraBox:

```sh
python xm2ultrabox.py "xm/song.xm"
```

Convert to BeepBox:

```sh
python xm2ultrabox.py "xm/song.xm" --beepbox
```

For UltraBox songs that use external samples, run the local player and import the song from `output/`:

```sh
python start_ultrabox.py
```


# XM2Box

XM2Box converts FastTracker II `.xm` files into UltraBox or BeepBox songs.

## Requirements

- Python 3
- Tkinter (usually included with Python)
- NumPy for BeepBox conversion:

```sh
python -m pip install numpy
```

## Run the app

From this folder, run:

```sh
python convert_xm.py
```

Choose an `.xm` file, select UltraBox or BeepBox, and click **Convert**. The converted files are saved in `output/`.

No songs are included with this project. You can download `.xm` files, such as Anvil's songs, from [Modland](https://modland.com/pub/modules/Fasttracker%202/Anvil/). The command-line examples below assume the files are in a folder named `xm/`. These songs belong to their authors and are not covered by this project's license.

If you plan to use a program that reads `.ultrabox.json` files directly, such as [Greater Diplomacy 5](https://github.com/GitGetGot415/Greater-Diplomacy-5), select **Keep full-quality samples in one file** before converting to UltraBox. Normal conversion may halve some samples to fit the song in UltraBox's URL. A file with full-quality samples may be too large to open in the UltraBox editor.

## Command line

Convert to UltraBox:

```sh
python xm2ultrabox.py "xm/song.xm"
```

Convert to BeepBox:

```sh
python xm2ultrabox.py "xm/song.xm" --beepbox
```

For UltraBox songs that use external samples, start the local player and import the song from `output/`:

```sh
python start_ultrabox.py
```

For a program that reads the song file directly, keep all samples in one file at full quality. Add `--rough` if the song is too big to convert normally:

```sh
python xm2ultrabox.py "xm/song.xm" --rough --embed-samples
```

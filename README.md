# Graph Pathfinding from Images

Extract a directed graph straight out of a hand-drawn / rendered graph image and compute the minimum-cost path between its start and end nodes — no manual graph entry required.

The pipeline turns a picture like this:

```
391_before.png   -->   391_after.png (nodes & edges detected and overlaid)
```

into a plain edge list (`graph.txt`), which is then fed to a C++ solver that returns the minimum cost to travel from node `0` to node `n-1`.

## How it works

The project is split into two stages that mirror the two languages in the repo:

### 1. Image → Graph (`final.py`, Python)

1. **Node detection** – `cv2.HoughCircles` locates the circular nodes in the image.
2. **Node labeling (OCR)** – each detected circle is cropped, upscaled, thresholded, and passed through `pytesseract` to read the digit printed inside it, giving a `{(x, y): node_id}` mapping.
3. **Edge / arrow detection** – a custom-trained **YOLO** model (via `ultralytics`) detects two classes of objects in the image:
   - `edge` — the line segment connecting two nodes
   - `arrow` — the arrowhead indicating direction
4. **Edge de-duplication** – overlapping `edge` boxes (common with thick or double-rendered lines) are merged using an IoU threshold.
5. **Edge → node matching** – for every detected edge box, the two diagonals of its bounding box are tested against nearby node coordinates (within `DISTANCE_THRESHOLD`) to figure out which pair of nodes it actually connects.
6. **Direction resolution** – any `arrow` box that falls near one end of a matched edge determines that edge's direction (or both, if arrows are detected on both ends). Edges are added to a `networkx.DiGraph`.
7. **Output** – an annotated image (`final_graph_result.png`) plus a plain-text edge list (`graph.txt`), one `u v` pair per line, terminated by `-1`.

### 2. Graph → Shortest Path (`solution.cpp`, C++)

`solution.cpp` reads `graph.txt` and runs a **Dijkstra-style search over a doubled ("layered") state space** `(node, toggle_state)`:

- Moving along a detected edge costs `1`.
- "Flipping" the toggle state at the current node costs `n` (the number of nodes) — this models a state-dependent traversal cost (e.g. a one-time detour/penalty) on top of the raw graph.
- The program prints the minimum cost to reach node `n - 1` from node `0` across either state, or `-1` if unreachable.

A precompiled Windows binary (`solution.exe`) is included for convenience.

## Repository structure

| Path | Description |
|---|---|
| `final.py` | Main pipeline: detects nodes/edges/arrows in an input image and writes `graph.txt`. |
| `solution.cpp` / `solution.exe` | Reads `graph.txt` and computes the minimum-cost path. |
| `Strat 1/`, `Strat 2/` | Experiment folders for different YOLO detection strategies/model runs. |
| `Documentation.pdf`, `main.tex` | Write-up describing the approach and methodology. |
| `test.jpg`, `393.png`, `420.png` | Sample input graph images. |
| `391_before.png`, `391_after.png` | Example image before/after detection and annotation. |
| `391_before_preprocessing.png`, `391_after_preprocessing.png` | Example of the preprocessing step used for OCR/node detection. |

## Requirements

**Python** (for `final.py`):
- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [NumPy](https://pypi.org/project/numpy/)
- [pytesseract](https://pypi.org/project/pytesseract/) + a local install of the [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) engine
- [networkx](https://pypi.org/project/networkx/)
- [ultralytics](https://pypi.org/project/ultralytics/) (YOLO) + a trained detection weights file (`best.pt`) for the `arrow` / `edge` classes

```bash
pip install opencv-python numpy pytesseract networkx ultralytics
```

**C++** (for `solution.cpp`):
- Any C++17-compatible compiler (g++, clang++, MSVC)

## Usage

1. **Set up paths.** `final.py` currently hardcodes a few local paths — update these before running:
   ```python
   image_path = r"C:\path\to\your\graph_image.png"
   model_path = r"C:\path\to\your\yolo\best.pt"
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
2. **Run the detector:**
   ```bash
   python final.py
   ```
   This produces:
   - `final_graph_result.png` — the input image with detected nodes/edges drawn on top
   - `graph.txt` — the extracted edge list

3. **Compile and run the solver:**
   ```bash
   g++ -O2 -std=c++17 solution.cpp -o solution
   ./solution
   ```
   (or simply run the prebuilt `solution.exe` on Windows). The solver reads `graph.txt` from the working directory and prints the minimum cost from node `0` to the highest-numbered node.

## Suggested end-to-end deployment

As noted in the original project notes, a simple product wrapper around this pipeline would look like:

1. A front-end where a user uploads an image of a graph.
2. The backend runs `final.py` on the uploaded image to produce `graph.txt`.
3. The backend runs `solution.cpp`/`solution.exe` on `graph.txt`.
4. The resulting minimum cost is returned/displayed to the user.

## Notes & limitations

- Node detection assumes nodes are drawn as circles containing a single OCR-readable digit.
- Edge/arrow detection quality depends entirely on the YOLO model weights used (`best.pt`), which is not included in this repo and must be trained/supplied separately — see `Strat 1/` and `Strat 2/` for the experiment strategies used to train it.
- Detection thresholds (`DISTANCE_THRESHOLD`, `ARROW_MATCH_THRESHOLD`, `MERGE_IOU_THRESHOLD`) may need tuning for different image resolutions or drawing styles.
- Paths in `final.py` are currently absolute/Windows-specific and should be parameterized (e.g. via CLI arguments) for portability.

## Documentation

See `Documentation.pdf` (source: `main.tex`) for a full write-up of the approach.

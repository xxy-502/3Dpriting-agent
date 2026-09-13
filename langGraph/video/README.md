# LangGraph-ready video defect tools

This package wraps the bundled `langGraph/video/video_pipeline.py` implementation as
two independently callable, state-safe stages:

```text
video_yolo_detection
        |
        | defect_exists=true and render_result_video=true
        v
video_sam_rendering
```

The full per-frame detections remain in `yolo_defect_events.json`. LangGraph
state receives a compact JSON record containing event counts, class counts,
time intervals, confidence summaries, and artifact paths.

## Model paths

- YOLO: automatically selects a `.pt` checkpoint under `model/yolo`
- SAM: automatically selects a `.pt` checkpoint under `model/sam`
- SAM config: `configs/sam2.1/sam2.1_hiera_b+`

The preferred YOLO filename is present in this checkout. The default resolver
only emits a warning and uses `best.pt` when an older checkout lacks
`yolo_best.pt`. An explicitly supplied missing checkpoint never falls back
silently.

## Run the stages separately

From the repository root in the `pytorch` conda environment:

```powershell
conda run --no-capture-output -n pytorch python -m langGraph.video.video_tools yolo `
  "vos_dataset\video\fdm_3d_printing_defect.mp4" `
  --output-dir "langGraph\runs\example"

conda run --no-capture-output -n pytorch python -m langGraph.video.video_tools sam `
  "vos_dataset\video\fdm_3d_printing_defect.mp4" `
  "langGraph\runs\example\yolo_defect_events.json" `
  --output-dir "langGraph\runs\example"
```

## Run the LangGraph subgraph

YOLO only:

```powershell
conda run --no-capture-output -n pytorch python -m langGraph.video.video_graph `
  "vos_dataset\video\fdm_3d_printing_defect.mp4" `
  --output-dir "langGraph\runs\example"
```

YOLO followed by SAM only when a confirmed event exists:

```powershell
conda run --no-capture-output -n pytorch python -m langGraph.video.video_graph `
  "vos_dataset\video\fdm_3d_printing_defect.mp4" `
  --output-dir "langGraph\runs\example" `
  --render-result-video
```

## Embed in a parent graph

```python
from langGraph.video import build_video_detection_graph

video_graph = build_video_detection_graph()
result = video_graph.invoke(
    {
        "video_path": r"D:\videos\print.mp4",
        "output_dir": r"D:\results\print",
        "render_result_video": True,
    }
)

compact_detection = result["detection_result"]
complete_video_result = result["video_result"]
```

The parent multimodal graph can register the compiled graph directly as a node,
or call it from a normal node. Do not place model instances, Qdrant clients,
CUDA tensors, masks, or video frames in graph state.

## Tests

The unit tests use fake tools, so they validate routing without loading CUDA
models:

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover `
  -s langGraph\tests -v
```

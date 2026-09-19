"""TerraSem Gradio demo app.

See BUILD.md Phase 9.1 for full specification.
TODO (Phase 9): implement after model is exported to INT8 ONNX.

Requirements:
- Load INT8 ONNX model (CPU only — do not import torch)
- 4-6 bundled sample frames + user upload
- Three tabs: Segmentation overlay / Traversability heatmap / 3D voxel view
- Display inference latency per request
- Lazy-load model inside first request handler (not at import time)
"""

import gradio as gr


def _not_implemented(*args, **kwargs):
    raise NotImplementedError("Demo not yet implemented (Phase 9).")


with gr.Blocks(title="TerraSem") as demo:
    gr.Markdown("# 🛻 TerraSem — Off-Road Semantic Occupancy (coming soon)")
    gr.Markdown("TODO (Phase 9): implement demo after ONNX export in Phase 8.")

if __name__ == "__main__":
    demo.launch()

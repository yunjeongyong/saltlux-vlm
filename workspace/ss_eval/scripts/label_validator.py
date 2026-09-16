"""Validate kie_output/*/*.json against data/*/images/*.png side by side. Edit + save json."""

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
KIE_DIR = ROOT / "kie_output"
DATA_DIR = ROOT / "data"


def list_samples():
    """Return sorted list of (doc_type, stem) for every kie_output json that has a matching image."""
    samples = []
    for doc_dir in sorted(KIE_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_type = doc_dir.name
        img_dir = DATA_DIR / doc_type / "images"
        for json_path in sorted(doc_dir.glob("*.json")):
            stem = json_path.stem
            img_path = find_image(img_dir, stem)
            if img_path is not None:
                samples.append((doc_type, stem))
    return samples


def find_image(img_dir: Path, stem: str):
    if not img_dir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


SAMPLES = list_samples()
LABELS = [f"{doc_type}/{stem}" for doc_type, stem in SAMPLES]


def load_sample(index: int):
    doc_type, stem = SAMPLES[index]
    json_path = KIE_DIR / doc_type / f"{stem}.json"
    img_path = find_image(DATA_DIR / doc_type / "images", stem)
    text = json_path.read_text(encoding="utf-8")
    status = f"[{index + 1}/{len(SAMPLES)}] {doc_type}/{stem}"
    return str(img_path), text, status, index


def save_sample(index: int, text: str):
    doc_type, stem = SAMPLES[index]
    json_path = KIE_DIR / doc_type / f"{stem}.json"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return f"Invalid JSON, not saved: {e}"
    json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Saved {doc_type}/{stem}"


def go_to(index: int):
    index = max(0, min(index, len(SAMPLES) - 1))
    return load_sample(index)


def prev_sample(index: int):
    return go_to(index - 1)


def next_sample(index: int):
    return go_to(index + 1)


def jump_to_label(label: str):
    if label in LABELS:
        return go_to(LABELS.index(label))
    return go_to(0)


with gr.Blocks(title="KIE Label Validator") as demo:
    gr.Markdown("# KIE Label Validator")
    index_state = gr.State(0)

    with gr.Row():
        selector = gr.Dropdown(choices=LABELS, value=LABELS[0] if LABELS else None, label="Sample")
        prev_btn = gr.Button("Prev")
        next_btn = gr.Button("Next")
        save_btn = gr.Button("Save", variant="primary")

    status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        image_box = gr.Image(label="Image", type="filepath")
        json_box = gr.Code(label="Label JSON", language="json", lines=40)

    demo.load(lambda: load_sample(0), outputs=[image_box, json_box, status_box, index_state])
    selector.change(jump_to_label, inputs=selector, outputs=[image_box, json_box, status_box, index_state])
    prev_btn.click(prev_sample, inputs=index_state, outputs=[image_box, json_box, status_box, index_state])
    next_btn.click(next_sample, inputs=index_state, outputs=[image_box, json_box, status_box, index_state])
    save_btn.click(save_sample, inputs=[index_state, json_box], outputs=status_box)


if __name__ == "__main__":
    demo.launch()

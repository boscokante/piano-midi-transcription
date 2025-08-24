# app.py created by Bosco Kante
import os
import gradio as gr
import torch

# paste into your app.py (near imports)

def _move_object_to_device(obj, device):
    """
    Best-effort recursive mover:
    - If attr is nn.Module -> call .to(device)
    - If attr is a tensor -> move it
    - If attr is a container (dict/list/tuple) containing tensors/modules -> move contents
    Returns a dictionary of limited diagnostics.
    """
    moved = {"modules": 0, "tensors": 0, "errors": []}

    def try_move(x, name):
        try:
            if isinstance(x, torch.nn.Module):
                x.to(device)
                moved["modules"] += 1
                return True
            if isinstance(x, torch.Tensor):
                # replace if an attribute on obj
                try:
                    setattr(obj, name, x.to(device))
                except Exception:
                    pass
                moved["tensors"] += 1
                return True
        except Exception as e:
            moved["errors"].append(f"{name}: {repr(e)}")
        return False

    # Move top-level attrs
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue

        # direct modules/tensors
        try_move(attr, name)

        # if attr is dict/list/tuple, walk them
        try:
            if isinstance(attr, dict):
                for k, v in attr.items():
                    try_move(v, f"{name}.{k}")
            elif isinstance(attr, (list, tuple, set)):
                for i, v in enumerate(attr):
                    try_move(v, f"{name}[{i}]")
        except Exception:
            pass

    # Also try obj.__dict__ if present
    try:
        for k, v in getattr(obj, "__dict__", {}).items():
            try_move(v, k)
    except Exception:
        pass

    return moved

# Example get_transcriptor wrapper (replace your existing code)
from piano_transcription_inference import PianoTranscription
_TRANSCRIPTOR = None

def get_transcriptor():
    global _TRANSCRIPTOR
    if _TRANSCRIPTOR is None:
        # create on CPU (safer) and then attempt to move internals to MPS
        tr = PianoTranscription(device="cpu", checkpoint_path=None)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if device.type != "cpu":
            try:
                diag = _move_object_to_device(tr, device)
                # try moving .model explicitly if available
                if hasattr(tr, "model"):
                    try:
                        tr.model.to(device)
                    except Exception:
                        pass
                # if transcriptor has a device attribute used internally, set it
                try:
                    setattr(tr, "device", device)
                except Exception:
                    pass

                # verify
                param_device = None
                try:
                    param_device = next(tr.model.parameters()).device
                except Exception:
                    pass

                if param_device and param_device.type == device.type:
                    print(f"Moved transcriptor to {device} (params on {param_device}) diag={diag}")
                    _TRANSCRIPTOR = tr
                else:
                    print("Warning: could not move model parameters to MPS. Falling back to CPU. diag=", diag)
                    _TRANSCRIPTOR = tr  # still usable on CPU
            except Exception as e:
                print("Error while moving transcriptor to device:", e)
                _TRANSCRIPTOR = tr
        else:
            _TRANSCRIPTOR = tr
    return _TRANSCRIPTOR




# IMPORTANT: piano_transcription_inference expects librosa/soundfile installed.
# The package will download model weights on first use (checkpoint_path=None).
from piano_transcription_inference import sample_rate, load_audio

def transcribe_file(audio_path):
    """
    audio_path: local file path (gradio provides filepath when type='filepath').
    Returns: path to generated .mid file (so Gradio offers it for download).
    """
    if not audio_path:
        raise gr.Error("No file provided")

    # Load audio at library's expected sample rate (mono)
    audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcribe and write out to repo tests/ folder
    base = os.path.splitext(os.path.basename(audio_path))[0]
    root_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(root_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    out_path = os.path.join(tests_dir, f"{base}.mid")

    transcriptor = get_transcriptor()
    # This runs the actual inference and writes the MIDI to out_path
    transcriptor.transcribe(audio, out_path)

    return out_path

description = """
Upload a piano recording (mp3, wav, m4a, flac, ogg). The model transcribes it to a downloadable **MIDI (.mid)** file.
Tip: shorter clips (1–3 minutes) are much faster. The first run will download model weights.
"""

def create_demo():
    with gr.Blocks(title="Piano → MIDI", css="""
        body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    """) as demo:
        gr.Markdown("# 🎹 Piano → MIDI Transcription")
        gr.Markdown(description)

        with gr.Row():
            audio_in = gr.Audio(sources=["upload"], type="filepath", label="Upload audio (mp3/wav/m4a/flac/ogg)")
            btn = gr.Button("Transcribe", variant="primary")

        midi_out = gr.File(label="Download MIDI")

        btn.click(fn=transcribe_file, inputs=audio_in, outputs=midi_out)

        gr.Markdown(
            "Notes: ffmpeg must be available on the system for some audio formats. "
            "If you deploy to Spaces, include `apt.txt` with `ffmpeg` so it's installed."
        )
    return demo

if __name__ == "__main__":
    # Local dev port
    port = int(os.environ.get("PORT", 7860))
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=port)

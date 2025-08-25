# app.py created by Bosco Kante
import os
import tempfile
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
import json
from music21 import converter

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

def midi_to_musicxml_str(midi_path):
    """
    Convert midi -> MusicXML string using music21.
    This is synchronous and returns the MusicXML as a text string.
    """
    try:
        print(f"music21: parsing MIDI file {midi_path}")
        score = converter.parse(midi_path)  # parse midi into music21 stream
        print(f"music21: parsed successfully, parts: {len(score.parts)}")
        
        tmp_xml = tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml")
        tmp_xml.close()
        print(f"music21: writing to temp file {tmp_xml.name}")
        
        # write MusicXML
        score.write('musicxml', fp=tmp_xml.name)
        print("music21: MusicXML written successfully")
        
        with open(tmp_xml.name, 'r', encoding='utf-8') as f:
            xml_text = f.read()
        os.unlink(tmp_xml.name)
        print(f"music21: read {len(xml_text)} characters of MusicXML")
        return xml_text
    except Exception as e:
        print(f"music21 error: {e}")
        raise RuntimeError(f"music21 conversion failed: {e}")

def build_osmd_html(musicxml_text):
    """
    Build an HTML page fragment that loads OSMD from CDN and renders the MusicXML.
    We embed the xml as a JS string safely using json.dumps to escape it.
    """
    xml_js = json.dumps(musicxml_text)
    html = f"""
    <div id="osmd_container"></div>
    <div id="osmd_status" style="font-size:0.9em; color: #666; margin-top:6px;">Loading score…</div>
    <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.2.1/build/opensheetmusicdisplay.min.js"></script>
    <script>
    (async () => {{
      try {{
        const xmlText = {xml_js};
        const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmd_container", {{
          drawingParameters: "compact", // 'compact' or 'default'
          followCursor: false,
          drawTitle: true
        }});
        await osmd.load(xmlText);
        osmd.render();
        document.getElementById("osmd_status").innerText = "Rendered";
      }} catch (err) {{
        document.getElementById("osmd_status").innerText = "Error rendering score: " + err;
        console.error(err);
      }}
    }})();
    </script>
    <style>
      #osmd_container svg {{ max-width: 100%; height: auto; }}
    </style>
    """
    return html

def transcribe_and_show_score(audio_path):
    """
    Transcribe audio to MIDI and convert to score display.
    Returns: (midi_file_path, score_html)
    """
    if not audio_path:
        raise gr.Error("No file provided")
    
    # Transcribe to MIDI
    midi_path = transcribe_file(audio_path)
    
    # Convert to MusicXML and create score display
    try:
        print(f"Converting MIDI to MusicXML: {midi_path}")
        xml_text = midi_to_musicxml_str(midi_path)
        print(f"MusicXML conversion successful, length: {len(xml_text)}")
        html = build_osmd_html(xml_text)
        print("OSMD HTML generated successfully")
    except Exception as e:
        print(f"Error in score conversion: {e}")
        html = f"<div style='color:red'>Error converting MIDI to score: {e}</div>"
    
    return midi_path, html

description = """
Upload a piano recording (mp3, wav, m4a, flac, ogg). The model transcribes it to a downloadable **MIDI (.mid)** file and displays the musical score.
Tip: shorter clips (1–3 minutes) are much faster! The first run will download model weights.
"""

def create_demo():
    with gr.Blocks(title="Piano → MIDI", css="""
        body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    """) as demo:
        gr.Markdown("# 🎹 Piano → MIDI → Score")
        gr.Markdown(description)

        with gr.Row():
            audio_in = gr.Audio(sources=["upload"], type="filepath", label="Upload audio (mp3/wav/m4a/flac/ogg)")
            btn = gr.Button("Transcribe & Show Score", variant="primary")

        with gr.Row():
            midi_out = gr.File(label="Download MIDI")
        
        score_html = gr.HTML("<i>Score will appear here after transcription.</i>")

        btn.click(fn=transcribe_and_show_score, inputs=audio_in, outputs=[midi_out, score_html])

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

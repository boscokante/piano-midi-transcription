# test_infer.py
import time, os, torch
from piano_transcription_inference import load_audio, sample_rate
from app import get_transcriptor

# Quick env info
print("Torch:", torch.__version__, "MPS available:", torch.backends.mps.is_available())

audio_path = "short_piano_clip.mp3"  # put a 15-30s clip here
if not os.path.exists(audio_path):
    raise SystemExit("Place a short audio file at short_piano_clip.mp3 to test")

tr = get_transcriptor()
print("Transcriptor object:", tr)
print("Model param device (if available):", end=" ")
try:
    print(next(tr.model.parameters()).device)
except Exception as e:
    print("inspect failed:", e)

audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)
root_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(root_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)
out_path = os.path.join(tests_dir, "test_out.mid")

t0 = time.time()
tr.transcribe(audio, out_path)
# on MPS, synchronize to ensure all ops finished before timing
if torch.backends.mps.is_available():
    try:
        torch.mps.synchronize()
    except Exception:
        pass
t_total = time.time() - t0

print(f"Transcription finished in {t_total:.2f}s, output: {out_path}")

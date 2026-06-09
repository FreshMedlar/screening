import soundfile as sf
from datasets import load_dataset, Audio

# 1. Load the dataset structure
dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100")

# 2. Force Hugging Face to drop automatic decoding hooks
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. Pull a sample metadata entry safely
sample = dataset[0]
print("Transcript:", sample["text"])
print("Audio Metadata:", sample["audio"])

# 4. Read the audio manually using standard file IO
# (Hugging Face stores the raw FLAC bytes inside the 'bytes' key)
import io
audio_data, sample_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]))
print("Audio Array Shape:", audio_data.shape)
print("Sample Rate:", sample_rate)

"""One-time tone color training for the cabal character.

OpenVoice clones a voice by extracting a "tone color embedding" (SE) from
reference vocals.  All recordings in ``cabal_source/`` belong to the same
person, so this module extracts the embedding from every VAD segment of
every recording, averages them and saves the result to
``assets/se/cabal.pth``.  The bot loads that tiny ``.pth`` file at runtime
instead of re-running this step for every message.
"""

import logging
import tempfile
import urllib.request
from glob import glob
from pathlib import Path

import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_DIR = PROJECT_ROOT / "checkpoints" / "converter"
SOURCE_DIR = PROJECT_ROOT / "cabal_source"
SE_DIR = PROJECT_ROOT / "assets" / "se"
CHARACTER = "cabal"

CONVERTER_FILES = {
    "config.json": (
        "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json"
    ),
    "checkpoint.pth": (
        "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth"
    ),
}

SEGMENT_SECONDS = 10.0
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_converter() -> None:
    """Download the OpenVoice V2 converter files if they are missing."""
    CONVERTER_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in CONVERTER_FILES.items():
        dest = CONVERTER_DIR / name
        if dest.is_file():
            continue
        logger.info("Downloading %s from %s", name, url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)


def ensure_silero_trusted() -> None:
    """Pre-approve the silero-vad repo so torch.hub never prompts."""
    trusted = Path(torch.hub.get_dir()) / "trusted_list"
    trusted.parent.mkdir(parents=True, exist_ok=True)
    entry = "snakers4_silero-vad\n"
    if not trusted.is_file() or entry not in trusted.read_text():
        with trusted.open("a") as f:
            f.write(entry)


def load_converter(device: str = DEFAULT_DEVICE) -> ToneColorConverter:
    """Load the OpenVoice tone color converter (V2)."""
    ensure_converter()
    converter = ToneColorConverter(str(CONVERTER_DIR / "config.json"), device=device)
    converter.watermark_model = None
    converter.load_ckpt(str(CONVERTER_DIR / "checkpoint.pth"))
    return converter


def extract_tone_color(converter: ToneColorConverter, out_path: Path) -> torch.Tensor:
    """Extract and save the averaged tone color embedding of the character."""
    sources = sorted(SOURCE_DIR.glob("*.wav"))
    if not sources:
        raise FileNotFoundError(f"No reference vocals found in {SOURCE_DIR}")

    ensure_silero_trusted()
    segments: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cabal-train-") as tmp:
        for wav in sources:
            wavs_folder = se_extractor.split_audio_vad(
                str(wav), wav.stem, target_dir=tmp
            )
            segments.extend(sorted(glob(f"{wavs_folder}/*.wav")))
        if not segments:
            raise RuntimeError(f"No speech segments found in {SOURCE_DIR}")

        logger.info("Extracting tone color from %d segments", len(segments))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        se = converter.extract_se(segments, se_save_path=str(out_path))

    logger.info("Saved tone color embedding to %s", out_path)
    return se


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    converter = load_converter()
    extract_tone_color(converter, SE_DIR / f"{CHARACTER}.pth")


if __name__ == "__main__":
    main()

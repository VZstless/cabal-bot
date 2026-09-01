import asyncio
import io
import logging
import subprocess
import tempfile
from datetime import timedelta
from glob import glob
from pathlib import Path

import torch
from openvoice import se_extractor
from telegram import ForceReply, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from training import CHARACTER, SE_DIR, ensure_silero_trusted, load_converter

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
SAMPLE_RATE = 22050
MIN_VAD_SECONDS = 5.0
OPUS_BITRATE = "64k"


def run_ffmpeg(*args: str) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], check=True
    )


def convert_to_cabal(
    input_ogg: Path, duration: float, converter, tgt_se: torch.Tensor
) -> io.BytesIO:
    """Convert one voice message to the cabal tone color, return OGG bytes."""
    with tempfile.TemporaryDirectory(prefix="cabal-msg-") as tmp:
        wav = Path(tmp) / "src.wav"
        run_ffmpeg("-i", str(input_ogg), "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav))

        if duration >= MIN_VAD_SECONDS:
            wavs_folder = se_extractor.split_audio_vad(str(wav), "src", target_dir=tmp)
            segments = sorted(glob(f"{wavs_folder}/*.wav"))
        else:
            segments = []
        src_se = converter.extract_se(segments or [str(wav)])

        out_wav = Path(tmp) / "out.wav"
        converter.convert(str(wav), src_se, tgt_se, output_path=str(out_wav))

        out_ogg = Path(tmp) / "out.ogg"
        run_ffmpeg(
            "-i", str(out_wav), "-c:a", "libopus", "-b:a", OPUS_BITRATE, str(out_ogg)
        )
        return io.BytesIO(out_ogg.read_bytes())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    message = update.message
    if user is None or message is None:
        return
    await message.reply_html(
        rf"Hi {user.mention_html()}!\nJust send me a voice message!",
        reply_markup=ForceReply(selective=True),
    )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ping whether the bot is alive"""
    message = update.message
    if message is None:
        return
    await message.reply_text("Pong! After 114514 ms.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    message = update.message
    if message is None:
        return
    await message.reply_text("Just send me a voice message.")


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Convert the received voice message to the cabal voice."""
    message = update.message
    if message is None:
        return
    audio = message.voice or message.audio
    if audio is None:
        return
    duration = audio.duration
    if isinstance(duration, timedelta):
        duration = duration.total_seconds()

    await message.reply_chat_action(ChatAction.RECORD_VOICE)
    src = Path(tempfile.gettempdir()) / f"cabal_{audio.file_unique_id}.ogg"
    file = await audio.get_file()
    await file.download_to_drive(src)

    try:
        async with context.bot_data["conversion_lock"]:
            out = await asyncio.to_thread(
                convert_to_cabal,
                src,
                float(duration),
                context.bot_data["converter"],
                context.bot_data["tgt_se"],
            )
    except Exception:
        logger.exception("Voice conversion failed")
        await message.reply_text("转换失败了，换一条语音试试吧。")
        return
    finally:
        src.unlink(missing_ok=True)

    await message.reply_voice(out, duration=int(duration))


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    with open(".env", "r") as file:
        TOKEN = file.readline().strip()
    application = Application.builder().token(TOKEN).build()

    logger.info("Loading voice converter...")
    ensure_silero_trusted()
    converter = load_converter()
    application.bot_data["converter"] = converter
    application.bot_data["tgt_se"] = torch.load(
        SE_DIR / f"{CHARACTER}.pth", weights_only=True
    ).to(converter.device)
    application.bot_data["conversion_lock"] = asyncio.Lock()
    logger.info("Voice converter ready.")

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ping", ping))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, voice_handler)
    )

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

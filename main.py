import discord
import io
import tempfile
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
from pywhispercpp.model import Model

import librosa
import soundfile as sf  # For saving the WAV file
import numpy as np

bot = discord.Bot()
connections = {}
load_dotenv()

model = Model('base.en')
print('Loaded model')


def downsample_audio(input_bytes, target_sr=16000):
    """
    Read WAV bytes, resample to `target_sr`, convert to mono, and return WAV bytes.

    - Uses soundfile to read the incoming bytes.
    - Uses librosa.resample for high-quality resampling.
    - Converts multi-channel audio to mono by averaging channels.
    - Writes out a WAV into an in-memory buffer and returns its bytes.

    On error the original bytes are returned.
    """
    try:
        # Read input bytes into numpy array and detect sample rate
        data, sr = sf.read(io.BytesIO(input_bytes))
        print(f"sample rate {sr}")
        # Ensure data is a numpy array
        data = np.asarray(data)

        # If multi-channel, resample each channel then average to mono
        if sr != target_sr:

            if data.ndim == 1:
                y = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            else:
                # resample each column (channel) then stack
                resampled_channels = [
                    librosa.resample(data[:, ch].astype(float), orig_sr=sr, target_sr=target_sr)
                    for ch in range(data.shape[1])
                ]
                # align lengths then average to mono
                min_len = min(len(c) for c in resampled_channels)
                stacked = np.stack([c[:min_len] for c in resampled_channels], axis=1)
                y = np.mean(stacked, axis=1)
        else:
            # sample rate already matches target
            if data.ndim == 1:
                y = data
            else:
                # convert to mono by averaging channels
                y = np.mean(data, axis=1)

        # Ensure float32 for writing
        y = y.astype(np.float32)

        # Write to in-memory WAV and return bytes
        out = io.BytesIO()
        sf.write(out, y, target_sr, format="WAV")
        out.seek(0)
        return out.read()

    except Exception as e:
        # On failure, log and return original bytes so caller can fallback
        print(f"downsample_audio failed: {e}")
        return input_bytes


@bot.command()
async def record(ctx):  # If you're using commands.Bot, this will also work.
    voice = ctx.author.voice

    if not voice:
        await ctx.respond("You aren't in a voice channel!")

    vc = await voice.channel.connect()  # Connect to the voice channel the author is in.
    connections.update({ctx.guild.id: vc})  # Updating the cache with the guild and channel.

    vc.start_recording(
        discord.sinks.WaveSink(),  # The sink type to use.
        once_done,  # What to do once done.
        ctx.channel  # The channel to disconnect from.
    )
    await ctx.respond("Started recording!")


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):  # Our voice client already passes these in.
    recorded_users = [  # A list of recorded users
        f"<@{user_id}>"
        for user_id, audio in sink.audio_data.items()
    ]
    await sink.vc.disconnect()  # Disconnect from the voice channel.
    files = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]  # List down the files.
    await channel.send(f"finished recording audio for: {', '.join(recorded_users)}.", files=files)  # Send a message with the accumulated files.
    
    await channel.send(f"transcribing...")
    # Play each user's WAV bytes (non-blocking for the event loop)
    for user_id, audio in sink.audio_data.items():
        audio.file.seek(0)
        wav_bytes = audio.file.read()
         # Resample to 16k mono (required by the model)
        TARGET_RATE = 16000
        wav_bytes_for_model = downsample_audio(wav_bytes, target_sr=TARGET_RATE)

        try:
            buf = io.BytesIO(wav_bytes_for_model)
            buf.seek(0)
            filename = f"{user_id}_16k.wav"
            await channel.send(content=f"Downsampled audio for <@{user_id}>:", file=discord.File(buf, filename=filename))
        except Exception as e:
            print(f"Failed to send downsampled audio for {user_id}: {e}")

        # model.transcribe may accept raw bytes or a filename; try bytes first, fallback to temp file
        try:
            segments = model.transcribe(wav_bytes_for_model)
        except Exception:
            tmp_path = None
            try:
                # Create a temporary path and use soundfile to write a valid WAV there.
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)  # Close the low-level fd so soundfile can open the path on Windows
                # Read in-memory WAV bytes to numpy data + samplerate, then write a proper WAV file using soundfile
                data, sr = sf.read(io.BytesIO(wav_bytes_for_model))
                sf.write(tmp_path, data, sr, format="WAV", subtype="PCM_16")
                segments = model.transcribe(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        
        # segments is library-specific; convert to a reasonable string for quick feedback
        try:
            pretty = "\n".join(getattr(s, "text", str(s)) for s in segments)
        except Exception:
            pretty = str(segments)

        await channel.send(f"transcription:\n{pretty}")


@bot.command()
async def stop_recording(ctx):
    if ctx.guild.id in connections:  # Check if the guild is in the cache.
        vc = connections[ctx.guild.id]
        vc.stop_recording()  # Stop recording, and call the callback (once_done).
        del connections[ctx.guild.id]  # Remove the guild from the cache.
        await ctx.delete()  # And delete.
    else:
        await ctx.respond("I am currently not recording here.")  # Respond with this if we aren't recording.


bot.run(env.get("DISCORD_BOT_TOKEN"))
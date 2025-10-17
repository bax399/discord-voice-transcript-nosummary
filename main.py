import discord
import io
import tempfile
import pathlib
import os
import time
import pydub  # pip install pydub==0.25.1
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
from pywhispercpp.model import Model
from pydub import AudioSegment


import librosa
import soundfile as sf  # For saving the WAV file
import numpy as np

import stable_whisper

bot = discord.Bot()
connections = {}
load_dotenv()

#model = Model('medium.en',n_threads=5)
model = stable_whisper.load_model('medium.en')
print('Loaded model')
channelstash = {}

def downsample_audio_to_ndarray(input_bytes, target_sr=16000) -> np.ndarray:
    """
    Read WAV bytes, resample to `target_sr`, convert to mono, and return a float32 1-D numpy array.

    - Uses soundfile to read the incoming bytes into float32 samples.
    - Converts multi-channel audio to mono by averaging channels.
    - Uses librosa.resample to resample to target_sr (high quality).
    - Returns a 1-D numpy.ndarray (dtype float32) with the resampled audio at `target_sr`.

    On error returns None.
    """
    try:
        # Read input bytes into numpy array (as float32) and detect sample rate
        data, sr = sf.read(io.BytesIO(input_bytes), dtype="float32")
        data = np.asarray(data)

        # Convert multi-channel to mono
        if data.ndim > 1:
            print(f"multi channel detected: {data.ndim}")
            data = np.mean(data, axis=1)

        # If required, resample to target_sr
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

        # Ensure float32 1-D ndarray
        arr = np.asarray(data, dtype=np.float32).flatten()
        return arr
    except Exception as e:
        print(f"downsample_audio_to_ndarray failed: {e}")
        return None


@bot.command()
async def record(ctx):  # If you're using commands.Bot, this will also work.
    voice = ctx.author.voice

    if not voice:
        await ctx.respond("You aren't in a voice channel!")

    vc = await voice.channel.connect()  # Connect to the voice channel the author is in.
    connections.update({ctx.guild.id: vc})  # Updating the cache with the guild and channel.

    # stash the guild object so once_done can look up members later
    channelstash[ctx.guild.id] = ctx.guild

    vc.start_recording(
        discord.sinks.MP3Sink(),  # The sink type to use.
        once_done,  # What to do once done.
        ctx.channel,  # The channel to disconnect from
        sync_start=True
    )
    await ctx.respond("Started recording!")


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    recorded_users = [
        f"<@{user_id}>"
        for user_id, audio in sink.audio_data.items()
    ]
    await sink.vc.disconnect()
    files = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]
    #DISABLE FILE SENDING
    await channel.send(f"finished recording audio for: {', '.join(recorded_users)}.", files=files)

    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "recorded_wavs"

    segments_list = []
    # resolve the guild we stashed (fallback to channel.guild)
    guild = channelstash.get(channel.guild.id) or channel.guild

    for user_id, audio in sink.audio_data.items():
        try:
            # ensure pointer at start
            try:
                audio.file.seek(0)
            except Exception:
                pass

            wav_bytes = audio.file.read()
            if not wav_bytes:
                print(f"No bytes for user {user_id}, skipping save.")
                continue

            filename = f"{int(time.time())}_{user_id}.mp3"
            out_path = out_dir / filename
            out_path.write_bytes(wav_bytes)
            print(f"Saved recorded WAV for user {user_id} to {out_path}")
        except Exception as e:
            print(f"Failed to save WAV for user {user_id}: {e}")
        
        try:
            #segments = model.transcribe(str(out_path), token_timestamps=True, max_len=1, split_on_word=True, suppress_blank=False)
            result = model.transcribe_minimal(str(out_path), suppress_blank=False, word_timestamps=True)
            print(vars(result))
        except Exception as e:
            print(f"failed transcribe {e}")
        
        # Convert segments to a readable string (library dependent)
        speaker_display = f"<@{user_id}>"
        try:
            member = await guild.fetch_member(int(user_id))
            # prefer nickname, then display_name
            speaker_display = member.nick or member.display_name or str(member)
        except Exception:
            # member fetch failed (not in guild or API error) keep mention string
            pass
        for segment in result:
            for word in segment:
                new_segment = {
                    "speaker" : speaker_display,
                    "user" : user_id,
                    "text" : word.word,
                    "start" : word.start,
                    "end" : word.end
                }
                segments_list.append(new_segment)

        # cleanup temporary file
        if os.path.exists(str(out_path)):
            os.remove(str(out_path))

    segments_list.sort(key=lambda x: x["start"])
    print(segments_list)

    transcript = ""
    current_speaker = None

    for segment in segments_list:
        if "speaker" in segment and segment["speaker"] != current_speaker:
            transcript += f"\n{segment['speaker']}: "
            current_speaker = segment["speaker"]
        transcript += f"{segment['text']} "

    transcript = transcript.strip()

    await channel.send(
        f"Finished recording audio for: {', '.join(recorded_users)}. \n Here is the transcript: \n\n{transcript}\n"
    )    


@bot.command()
async def stop_recording(ctx):
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.delete()
    else:
        await ctx.respond("I am currently not recording here.")


bot.run(env.get("DISCORD_BOT_TOKEN"))
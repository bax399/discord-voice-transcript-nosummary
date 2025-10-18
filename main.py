import discord
import io
import tempfile
import pathlib
import os
import time
import pydub
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
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

# periodically start_recording then stop recording after a set time, fire off a new 

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
    # TODO if stop recording hasn't been called, bot should reconnect automatically!
    recorded_users = [
        f"<@{user_id}>"
        for user_id, audio in sink.audio_data.items()
    ]
    await sink.vc.disconnect()
    files = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]
    #DISABLE FILE SENDING
    # await channel.send(f"finished recording audio for: {', '.join(recorded_users)}.", files=files)

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
    
        # Cleanup: remove all files under `recorded_wavs`
    try:
        for p in out_dir.iterdir():
            if p.is_file():
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Failed to remove {p}: {e}")
        print(f"Cleaned up files under {out_dir}")
    except Exception as e:
        print(f"Failed cleanup of {out_dir}: {e}")



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
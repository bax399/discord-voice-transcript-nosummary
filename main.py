import discord
import wave
import io    
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
#from whisper_cpp_python import Whisper
from pywhispercpp.model import Model
from pydub import AudioSegment
_HAS_PYDUB = True   

bot = discord.Bot()
connections = {}
load_dotenv()

model = Model('base.en')
print('Loaded model')
#not needed on windows
#discord.opus.load_opus("libopus-0.x64.dll")

@bot.command()
async def record(ctx):
    voice = ctx.author.voice

    if not voice:
        await ctx.respond("⚠️ You aren't in a voice channel!")

    vc = await voice.channel.connect()
    connections.update({ctx.guild.id: vc})

    vc.start_recording(
        discord.sinks.WaveSink(),
        once_done,
        ctx.channel,
    )
    await ctx.respond("🔴 Listening to this conversation.")


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
    await sink.vc.disconnect()
    print("discord vc decoder sampling rate:", sink.vc.decoder.SAMPLING_RATE)
    sampling_rate = sink.vc.decoder.SAMPLING_RATE

    count = len(recorded_users)
    print(f"Recorded {count} user(s): {', '.join(recorded_users) if count else 'none'}")
    await channel.send(f"Recorded {count} user(s): {', '.join(recorded_users) if count else 'none'}")

    words_list = []    

    last_wav_bytes = None
    last_user_id_for_test = None

    for user_id, audio in sink.audio_data.items():
        # ensure WaveSink has formatted the stream and pointer is at the start
        try:
            audio.file.seek(0)
        except Exception:
            pass

        wav_bytes = audio.file.read()

        last_wav_bytes = wav_bytes
        last_user_id_for_test = user_id

        # inspect WAV header (in-memory)
        wav_rate = None
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                wav_rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
            print(f"User {user_id} WAV: {wav_rate} Hz, {channels} channels, sampwidth={sampwidth}")
        except wave.Error as e:
            print(f"Failed to read WAV header for user {user_id}: {e}")

        # If the model insists on 16000 Hz, either set the constant or resample.
        TARGET_RATE = 16000
        if wav_rate is not None and wav_rate != TARGET_RATE:
            print(f"Warning: WAV is {wav_rate} Hz but model expects {TARGET_RATE} Hz.")
            if _HAS_PYDUB:
                try:
                    seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
                    seg = seg.set_frame_rate(TARGET_RATE)

                    # pad with silence to meet min duration required by whisper_full_with_state (1000 ms)
                    MIN_MS = 1500
                    duration_ms = len(seg)
                    if duration_ms < MIN_MS:
                        pad_ms = MIN_MS - duration_ms
                        seg = seg + AudioSegment.silent(duration=pad_ms)
                        print(f"Padded user {user_id} audio by {pad_ms} ms to reach {MIN_MS} ms total.")

                    buf = io.BytesIO()
                    seg.export(buf, format="wav")
                    wav_bytes_for_model = buf.getvalue()
                    print(f"Resampled user {user_id} to {TARGET_RATE} Hz using pydub.")
                except Exception as e:
                    print("Resampling with pydub failed:", e)
                    wav_bytes_for_model = wav_bytes
            else:
                print("pydub not available — model may reject non-16000 Hz WAVs.")
                wav_bytes_for_model = wav_bytes
        else:
            wav_bytes_for_model = wav_bytes

        # write bytes to a temporary .wav file (use delete=False for Windows compatibility)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav_bytes_for_model)
                tf.flush()
                tmp_path = tf.name

            # pass the filename to the transcriber (many libs accept a path)
            segments = model.transcribe(tmp_path)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        for segment in segments:
            print(segment.text)
        words = segments

        for word in words:
            # if speaker is not 0, then it's someone else, set the user ID to that.
            ## This is to make sure that the dearize work. if multiple people are speaking from the same user ID
            if word["speaker"] != 0:
                user_id = word["speaker"]

            new_word = {
                "word": word["word"],
                "start": word["start"],
                "end": word["end"],
                "confidence": word["confidence"],
                "punctuated_word": word["punctuated_word"],
                "speaker": user_id,
                "speaker_confidence": word["speaker_confidence"],
            }
            words_list.append(new_word)

    # Write the last wav bytes to repository-root/testwav for inspection (if any)
    if last_wav_bytes is not None:
        repo_root = Path(__file__).resolve().parent
        testwav_dir = repo_root / "testwav"
        try:
            testwav_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{last_user_id_for_test}.wav"
            out_path = testwav_dir / filename
            out_path.write_bytes(last_wav_bytes)
            print(f"Wrote last WAV to {out_path}")
        except Exception as e:
            print(f"Failed to write test wav: {e}")

    words_list.sort(key=lambda x: x["start"])

    print(words_list)

    transcript = ""
    current_speaker = None

    for word in words_list:
        if "speaker" in word and word["speaker"] != current_speaker:
            transcript += f"\n\nSpeaker <@{word['speaker']}>: "
            current_speaker = word["speaker"]
        transcript += f"{word['punctuated_word']} "

    transcript = transcript.strip()
    
    await channel.send(
        f"finished recording audio for: {', '.join(recorded_users)}. Here is the transcript: \n\n{transcript}\n\nHere is the summary: \n\n"
    )


@bot.command()
async def stop_recording(ctx):
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.delete()
    else:
        await ctx.respond("🚫 Not recording here")


bot.run(env.get("DISCORD_BOT_TOKEN"))

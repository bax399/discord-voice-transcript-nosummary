import discord
import wave
import io
import tempfile
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
from pywhispercpp.model import Model
_HAS_PYDUB = True

import librosa
import soundfile as sf # For saving the WAV file

def downsample_audio(input_path, output_path, target_sr=16000):
    """
    Downsamples an audio WAV file to a target sample rate.

    Args:
        input_path (str or bytes): Path to the input WAV file, or raw WAV bytes.
        output_path (str): Path to save the downsampled WAV file.
        target_sr (int): The desired target sample rate (e.g., 16000 for 16kHz).
    """
    if isinstance(input_path, bytes):
        # If input_path is bytes, read it from a BytesIO object
        audio_data, sr = librosa.load(io.BytesIO(input_path), sr=None)
    else:
        # Otherwise, assume it's a file path
        audio_data, sr = librosa.load(input_path, sr=None)

    # If the audio is already at the target sample rate, just save it
    if sr == target_sr:
        print(f"Audio is already at {target_sr} Hz. Saving directly.")
        sf.write(output_path, audio_data, sr)
        return

    # Downsample the audio
    downsampled_audio = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)

    # Save the downsampled audio to the output path
    sf.write(output_path, downsampled_audio, target_sr)
    print(f"Audio downsampled from {sr} Hz to {target_sr} Hz and saved to {output_path}")
    # --- Return as bytes ---
    # Create an in-memory binary stream
    buffer = io.BytesIO()
    # Write the downsampled audio to the buffer as a WAV file
    sf.write(buffer, downsampled_audio, target_sr, format='WAV')
    # Get the bytes from the buffer
    downsampled_wav_bytes = buffer.getvalue()
    buffer.close() # Close the buffer

    return downsampled_wav_bytes


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


    # Play each user's WAV bytes (non-blocking for the event loop)
    for user_id, audio in sink.audio_data.items():
        wav_bytes = audio.file.read()
         # Resample to 16k mono (required by the model)
        TARGET_RATE = 16000
        wav_bytes_for_model = downsample_audio(wav_bytes, "temp_input.wav", target_sr=TARGET_RATE)

        # model.transcribe may accept raw bytes or a filename; try bytes first, fallback to temp file
        try:
            segments = model.transcribe(wav_bytes_for_model)
        except Exception:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(wav_bytes_for_model)
                    tf.flush()
                    tmp_path = tf.name
                segments = model.transcribe(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        for segment in segments:
            print('Segment? '+segment.text)
        #words = segments

        #for word in words:
            # if speaker is not 0, then it's someone else, set the user ID to that.
            ## This is to make sure that the dearize work. if multiple people are speaking from the same user ID
            #if word["speaker"] != 0:
            #    user_id = word["speaker"]

            #new_word = {
            #    "word": word["word"],
            #    "start": word["start"],
            #    "end": word["end"],
            #    "confidence": word["confidence"],
            #    "punctuated_word": word["punctuated_word"],
            #    "speaker": user_id,
            #    "speaker_confidence": word["speaker_confidence"],
            #}
            #words_list.append(new_word)
        
    await channel.send(
        f"finished recording audio for: {', '.join(recorded_users)}."
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

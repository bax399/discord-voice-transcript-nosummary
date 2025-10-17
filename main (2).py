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

        # pass the filename to the transcriber (many libs accept a path)
        segments = model.transcribe(wave_bytes, language='en', fp16=False, word_timestamps=True, diarization=True)

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

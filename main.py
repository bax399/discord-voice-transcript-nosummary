import discord
from dotenv import load_dotenv
from os import environ as env
#from whisper_cpp_python import Whisper
from pywhispercpp.model import Model

bot = discord.Bot()
connections = {}
load_dotenv()

model = Model('base.en')

discord.opus.load_opus("/usr/local/opt/opus/lib/libopus.0.dylib")

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

    words_list = []
    whisper = Whisper(model_path="./models/ggml-tiny.bin")

    for user_id, audio in sink.audio_data.items():
        payload: FileSource = {
            "buffer": audio.file.read(),
        }

        #output = whisper.transcribe(payload["buffer"])
        #print(output)
        segments = model.transcribe(payload["buffer"])
        print("Got segments?")
        for segment in segments:
            print(segment.text)
        words = segments

        words = [word.to_dict() for word in words]

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

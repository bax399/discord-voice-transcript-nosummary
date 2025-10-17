# A fully self-hosted Discord Bot that listens to a voice channel and transcribes the audio

Yoinking https://github.com/Dhravya/discord-voice-transcript-for-teams

- Removing chat summarization for now as I don't want it
- Using self-hosted whisper model to do the transcribing, paired with stable-ts to keep timestamps somewhat diarized between speaker audio files (its not perfect)

## How to use

1. Create a discord bot and invite it to your server. [Guide](https://discordpy.readthedocs.io/en/latest/discord.html)
2. Clone this repo and install the requirements `pip install -r requirements.txt`
3. Create a `.env` file and add the following variables:
    - `DISCORD_BOT_TOKEN`: The token of your discord bot

4. Run the bot `python main.py`

## Commands

- `/record` - Starts recording the audio in the voice channel
- `/stop_recording` - Stops recording the audio in the voice channel

## Tools used


- [Pycord](https://pycord.dev) - Python wrapper for discord api
- Whisper (Transcription)
- https://github.com/jianfch/stable-ts/tree/main For stable transcription timestamping

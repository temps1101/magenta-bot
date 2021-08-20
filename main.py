from os import path, remove

import discord
from discord.ext import commands
from midi2audio import FluidSynth

from magenta_tools import model


# prepair AI stuffs
unconditional_generator = model.UnconditionalGenerator()

# prepair fluidsynth
fs = FluidSynth(sound_font='/content/Yamaha-C5-Salamander-JNv5.1.sf2')


def is_playing(ctx):
    if ctx.message.guild.voice_client:
        if ctx.message.guild.voice_client.is_playing():
            return True

        else:
                return False
    else:
        return False


async def generate(mode):
    if mode == 'uc':
        midi_filename = unconditional_generator.generate()
        wav_filename = path.join(path.dirname(midi_filename), '{}.wav'.format(path.splitext(path.basename(midi_filename))[0]))
        fs.midi_to_audio(midi_filename, wav_filename)
        remove(midi_filename)

        return wav_filename


# main bit
intents = discord.Intents().default()
intents.members = True
client = commands.Bot(command_prefix='!', intents=intents)

busy = False


@client.command(name='tf')
async def temps_finger(ctx, arg):
    global busy
    channel = ctx.channel
    if arg.lower() == 'uc':
        if not busy:
            if not is_playing(ctx):
                busy = True
                await channel.send('Generating unconditional sequence...')
                filename = await generate('uc')
                await channel.send('Done!')

                if ctx.author.voice:
                    try:
                        vc = await ctx.message.author.voice.channel.connect()

                    except discord.errors.ClientException:
                        vc = ctx.message.guild.voice_client

                    vc.play(discord.FFmpegPCMAudio(source=filename))

                    busy = False

            else:
                await channel.send('still playing!')
        else:
            await channel.send('still busy!')

    if arg.lower() == 'dc':
        if ctx.message.guild.voice_client:
            if not ctx.message.guild.voice_client.is_playing():
                await ctx.message.guild.voice_client.disconnect()
                await channel.send('disconnected from vc')

            else:
                await channel.send('still plaing!')
        else:
            await channel.send('Not connected to the vc')


# run the bot
TOKEN = input("Please type your discord bot token : ")
client.run(TOKEN)

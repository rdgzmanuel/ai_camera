import json
from typing import Dict, Literal
import os

import requests
from gtts import gTTS
import src.config as config


# === CONFIGURATION ===

LLMProvider = Literal["openai", "anthropic", "mistral", "google", "groq"]
TTSProvider = Literal["gtts", "edge", "elevenlabs", "voicemaker"]

SELECTED_LLM: LLMProvider = "groq"
SELECTED_TTS: TTSProvider = "voicemaker"

# === API KEYS ===
OPENAI_API_KEY: str = config.OPENAI_API_KEY
ANTHROPIC_API_KEY: str = config.ANTHROPIC_API_KEY
GROQ_API_KEY: str = config.GROQ_API_KEY
ELEVENLABS_API_KEY: str = config.ELEVENLABS_API_KEY
HF_API_KEY: str = config.HF_API_KEY
GOOGLE_API_KEY: str = config.GOOGLE_API_KEY
VOICEMAKER_API_KEY: str = config.VOICE_MAKER_API_KEY

# === FUNCTIONS ===

def load_match_data(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["data"]


def build_prompt(data: Dict) -> str:
    # return f"""
    # Generate a 1-minute spoken introduction in Spanish (Spain) for a live football stream. 
    # Mention the stadium '{data['stadium']['name']}', its features, and today's weather. 
    # Talk about the local team '{data['local_team']['name']}', their league position, top scorer {data['local_team']['top_scorer']['name']} with {data['local_team']['top_scorer']['goals']} goals, and coach. 
    # Do the same for the visiting team '{data['visiting_team']['name']}'.
    # Make it exciting and informative for a new viewer joining the stream.
    # """

    return f"""
    Generate a 1-minute spoken introduction in Spanish (Spain) for a live football stream. 
    Mention the stadium, its features, and today's weather. 
    Talk about the local team, their league position, top scorer with number of goals, and coach. 
    Do the same for the visiting team.
    Make it exciting and informative for a new viewer joining the stream.
    This is the data: {data}. No need to mention everything. Give me directly and only the commentary text.
    """

# === LLM PROVIDERS ===

def generate_text_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()


def generate_text_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def generate_text_mistral(prompt: str) -> str:
    HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}  # Replace with your token
    payload = {"inputs": prompt}

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    output = response.json()
    if isinstance(output, list) and "generated_text" in output[0]:
        return output[0]["generated_text"]
    else:
        raise ValueError("Unexpected response format from Hugging Face API.")


def generate_text_google(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-flash" if you prefer
    response = model.generate_content(prompt)
    return response.text


def generate_text_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=1024,
        top_p=1,
        stream=False
    )

    return response.choices[0].message.content.strip()


def generate_text(prompt: str, model: LLMProvider) -> str:
    if model == "openai":
        return generate_text_openai(prompt)
    elif model == "anthropic":
        return generate_text_anthropic(prompt)
    elif model == "mistral":
        return generate_text_mistral(prompt)
    elif model == "google":
        return generate_text_google(prompt)
    elif model == "groq":
        return generate_text_groq(prompt)
    else:
        raise ValueError(f"Unsupported LLM model: {model}")


# === TTS PROVIDERS ===
def tts_gtts(text: str, output_path: str):
    tts = gTTS(text, lang="es")
    tts.save(output_path)


async def tts_edge(text: str, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice="es-ES-AlvaroNeural")
    await communicate.save(output_path)


def tts_elevenlabs(text: str, output_path: str):
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    elevenlabs = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )

    response = elevenlabs.text_to_speech.convert(
        voice_id="TZ3pgF19H1pelFonL8Zq", # Pablo: TZ3pgF19H1pelFonL8Zq   Gabo: o0SveC0zgHFuCsEO3vHR
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        # Optional voice settings that allow you to customize the output
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    with open(output_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    
    print(f"{output_path}: A new audio file was saved successfully!")


def tts_voicemaker(text: str, output_path: str):
    url = "https://developer.voicemaker.in/voice/api"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VOICEMAKER_API_KEY}"
    }

    payload = {
        "Engine": "neural",
        "VoiceId": "ai3-es-ES-Lorenzo",         # You can change to other voices
        "LanguageCode": "es-ES",
        "Text": text,
        "OutputFormat": "mp3",
        "SampleRate": "48000",
        "Effect": "default",           # Options: default, soft, whispered, etc.
        "MasterVolume": "0",           # -10 to 10
        "MasterSpeed": "0",            # -10 to 10
        "MasterPitch": "0"             # -10 to 10
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "path" in result:
            audio_url = result["path"]
            audio_response = requests.get(audio_url)
            if audio_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(audio_response.content)
                print(f"{output_path}: A new audio file was saved successfully!")
            else:
                print(f"Error downloading audio: {audio_response.status_code}")
        else:
            print("Error: Unexpected response data:", result)
    else:
        print(f"Error {response.status_code}: {response.text}")
    


def run_tts(text: str, provider: TTSProvider, output_path: str):
    if provider == "gtts":
        tts_gtts(text, output_path)
    elif provider == "elevenlabs":
        tts_elevenlabs(text, output_path)
    elif provider == "edge":
        import asyncio
        asyncio.run(tts_edge(text, output_path))
    elif provider == "voicemaker":
        tts_voicemaker(text, output_path)
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")


# === MAIN EXECUTION ===

def main(data_info_path: str):
    match_data = load_match_data(data_info_path)
    prompt = build_prompt(match_data)
    print("Generating commentary text...")
    # commentary = generate_text(prompt, SELECTED_LLM)
    # print("Generated text:\n", commentary)

    commentary = """ ¡Bienvenidos a la retransmisión en directo del partido entre la Agrupación Deportiva Colmenar Viejo y el Club Deportivo Sanse!  Desde el espectacular estadio Alberto Ruiz en Colmenar Viejo, Madrid, un campo con capacidad para 5000 espectadores y con gradas cubiertas, marcador electrónico… ¡y un techo amarillo que lo hace inconfundible!  Hoy, con un sol radiante y 20 grados, se espera un ambientazo.
    El Colmenar Viejo, líder de la liga, llega con cinco victorias y un empate en sus últimos seis partidos.  Su estrella, Juan Pérez Bonilla, con 12 goles, buscará aumentar su cuenta, guiado por el experimentado entrenador Carlos García.
    Enfrente, el Club Deportivo Sanse, segundo clasificado, un rival directo con un juego muy ofensivo.  Miguel Torres Sánchez, su pichichi con 14 goles, será una amenaza constante.  La entrenadora Laura Fernández Ortega, con 3 años de experiencia, buscará dar la sorpresa.
     ¡Comenzamos!"""

    print("Converting to audio...")
    run_tts(commentary, SELECTED_TTS, "commentary/commentary.mp3")
    print("✅ Audio commentary saved as 'commentary/commentary.mp3'.")


if __name__ == "__main__":
    data_info_path: str = "commentary/data.json"
    
    main(data_info_path)

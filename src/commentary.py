import json
from typing import Literal
from ollama import chat
from ollama import ChatResponse

import requests
from gtts import gTTS
import src.config as config


# === CONFIGURATION ===

LLMProvider = Literal["openai", "anthropic", "mistral", "google", "groq"]
TTSProvider = Literal["gtts", "edge", "elevenlabs", "voicemaker"]

SELECTED_LLM: LLMProvider = "openai"
SELECTED_TTS: TTSProvider = "elevenlabs"

# === API KEYS ===
OPENAI_API_KEY: str = config.OPENAI_API_KEY
ANTHROPIC_API_KEY: str = config.ANTHROPIC_API_KEY
GROQ_API_KEY: str = config.GROQ_API_KEY
ELEVENLABS_API_KEY: str = config.ELEVENLABS_API_KEY
HF_API_KEY: str = config.HF_API_KEY
GOOGLE_API_KEY: str = config.GOOGLE_API_KEY
VOICEMAKER_API_KEY: str = config.VOICE_MAKER_API_KEY


def load_match_data(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["data"]


def generate_prompt(data_path: str) -> str:
    """
    Generates a prompt based on the match data.
    Args:
        data_path (str): Path to the match data JSON file.
    Returns:
        str: Formatted prompt string.
    """

    match_data: dict = load_match_data(data_path)
    prompt = f"""Generate a 50-second spoken introduction in Spanish (Spain) for a live football stream. 
    Mention the stadium, its features, and today's weather. 
    Talk about the local team, their league position, top scorer with number of goals, and coach. 
    Do the same for the visiting team.
    Make it exciting and informative for a new viewer joining the stream.
    This is the data: {match_data}. No need to mention everything. Give me directly and only the commentary text.
    Make sure every character is easily readable.\n"""
    return prompt


# === LLM PROVIDERS ===

def generate_text_openai(prompt: str, model: str = "gpt-oss:20b") -> str:
    """
    Generates a response from the LLM based on the provided prompt.
    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The model to use for generating the response.
    Returns:
        str: The generated response from the LLM.
    """

    response: ChatResponse = chat(model=model, messages=[
    {
        "role": "user",
        "content": prompt,
    },
    ])
    # print(response["message"]["content"])
    content: str = response["message"]["content"].strip()
    # print(response.message.content)

    return content


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
        voice_id="TZ3pgF19H1pelFonL8Zq",  # Pablo TZ3pgF19H1pelFonL8Zq    Gabo: o0SveC0zgHFuCsEO3vHR
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.2,           # keeps some variety for excitement
            similarity_boost=1.0,    # stay close to chosen voice
            style=0.85,              # adds drama and energy
            use_speaker_boost=True,  # richer, more present sound
            speed=1.12,              # faster pacing like commentary
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
        "VoiceId": "ai3-es-ES-Lorenzo",
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
    prompt = generate_prompt(data_info_path)
    print("Generating commentary text...")
    commentary = generate_text(prompt, SELECTED_LLM)
    print("Generated text:\n", commentary)

    # commentary = "¡Buenas tardes, espectadores! Nos encontramos en el estadio Alberto Ruiz, uno de los más grandes de la liga con 5 000 plazas, en Colmenar Viejo. Hoy hace 20 grados bajo un cielo despejado. En casa, la Agrupación Deportiva Colmenar Viejo, lidera la tabla con un sólido record de victorias y empates; su máximo goleador es Juan Pérez Bonilla con 12 tantos, y el equipo está a cargo de Carlos García, con 5 años de experiencia. Del otro lado, el Club Deportivo Sanse ocupa el segundo puesto; su estrella es Miguel Torres Sánchez, 14 goles, y su entrenador, Laura Fernández Ortega, lleva 3 años guiando al equipo. ¡Que comience el partido!"

    print("Converting to audio...")
    run_tts(commentary, SELECTED_TTS, "commentary/commentary.mp3")
    print("✅ Audio commentary saved as 'commentary/commentary.mp3'.")


if __name__ == "__main__":
    data_info_path: str = "commentary/data.json"
    
    main(data_info_path)

from ollama import chat
from ollama import ChatResponse
import json


def generate_prompt(data_path: str) -> str:
    """
    Generates a prompt based on the match data.
    Args:
        data_path (str): Path to the match data JSON file.
    Returns:
        str: Formatted prompt string.
    """

    match_data: dict = load_match_data(data_path)
    prompt = f"""Generate a 1-minute spoken introduction in Spanish (Spain) for a live football stream. 
    Mention the stadium, its features, and today's weather. 
    Talk about the local team, their league position, top scorer with number of goals, and coach. 
    Do the same for the visiting team.
    Make it exciting and informative for a new viewer joining the stream.
    This is the data: {match_data}. No need to mention everything. Give me directly and only the commentary text.\n"""
    return prompt


def generate_response(prompt: str, model: str = "gpt-oss:20b", streaming: bool = False) -> str:
    """
    Generates a response from the LLM based on the provided prompt.
    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The model to use for generating the response.
    Returns:
        str: The generated response from the LLM.
    """

    if streaming:
        stream = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)

    else:
        response: ChatResponse = chat(model=model, messages=[
        {
            "role": "user",
            "content": prompt,
        },
        ])
        print(response["message"]["content"])
        content: str = response["message"]["content"].strip()
        # print(response.message.content)

    return content


def load_match_data(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["data"]


def main(data_path: str) -> None:
    """
    Main function to generate a prompt and response based on match data.
    Args:
        data_path (str): Path to the match data JSON file.
    """
    prompt: str = generate_prompt(data_path)

    response: str = generate_response(prompt)

    return None


if __name__ == "__main__":
    data_info_path: str = "commentary/data.json"
    main(data_info_path)

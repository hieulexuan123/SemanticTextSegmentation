from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv('.env')
client = OpenAI(api_key=os.getenv('API_KEY'))


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content, response.usage.total_tokens, response.choices[0].finish_reason


def segment(content, model="gpt-3.5-turbo"):
    prompt = f"""
    Your task is to semantically divide a news article in Vietnamese into many sections.

    The news article is delimited by triple backticks.
    A cluster of sentences that support a common idea or subtopic form a section.

    The number of sections should be not too many and not too few.
    Make sure orginal text unchanged.

    Print list of sections in the json format. Each section item has 2 keys: topic and sentences. Sentences is a list of sentence in the section.

    Article: ```{content}```
    """
    response, total_tokens, finish_reason = get_completion(prompt, model)
    if finish_reason != "stop":
        return ""
    else:
        print(f"Total_tokens: {total_tokens}\n")
        print(f"Response: {response}")
        print("------------------------\n")
        return response

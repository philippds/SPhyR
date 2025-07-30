from openai import OpenAI
from google import genai
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


def run_openai(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OAI"))
    response = client.responses.create(model="gpt-4.1-2025-04-14", input=prompt)
    return response.output_text.strip()


def run_openai_4o(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OAI"))
    response = client.responses.create(model="gpt-4o-2024-08-06", input=prompt)
    return response.output_text.strip()


def run_openai_3_5_turbo(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OAI"))
    response = client.responses.create(model="gpt-3.5-turbo-0125", input=prompt)
    return response.output_text.strip()


def run_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEM"))
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06",
        contents=prompt,
    )
    return response.text


def run_gemini_1_5(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEM"))
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt,
    )
    return response.text


def run_claude(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANT"))
    message = client.messages.create(
        max_tokens=1024,
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_claude_opus(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANT"))
    message = client.messages.create(
        max_tokens=1024,
        model="claude-opus-4-20250514",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_deepkseek(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("R1"), base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def run_perplexity_sonar(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("PPLX"), base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_perplexity_sonar_reasoning(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("PPLX"), base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar-reasoning",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

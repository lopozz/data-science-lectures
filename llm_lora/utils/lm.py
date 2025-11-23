import os
import openai
import requests

from tqdm.contrib.concurrent import thread_map


def call_openai_api(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return chat_completion.choices[0].message.content


def call_vllm_api(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> str:
    vllm_host = "0.0.0.0"
    vllm_port = 8003
    client = openai.OpenAI(
        base_url=f"http://{vllm_host}:{vllm_port}/v1",
        api_key="NOT A REAL KEY",
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return chat_completion.choices[0].message.content


def call_ollama_api(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    ollama_host = "127.0.0.1"
    ollama_port = 11434  # Default port for Ollama server
    response = requests.post(
        f"http://{ollama_host}:{ollama_port}/v1/completions", json=payload
    )

    # Check if the response is successful
    if response.status_code == 200:
        response_data = response.json()
        return response_data["choices"][0]["text"]
    else:
        # Handle failure response
        return f"Error: Unable to get a response from Ollama (status code {response.status_code})"


def run_inference_batch(prompts: list[str], args) -> list[str]:
    """
    Run a list of prompts through the chosen backend (vllm or openai)
    and return the list of outputs.
    """
    if args.inference_method == "openai":
        outputs = thread_map(
            lambda p: call_openai_api(
                p,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            prompts,
            desc="Predicting with openai",
        )
    elif args.inference_method == "vllm":
        outputs = thread_map(
            lambda p: call_vllm_api(
                p,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            prompts,
            desc="Predicting with vllm",
        )
    elif args.inference_method == "ollama":
        outputs = thread_map(
            lambda p: call_ollama_api(
                p,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            prompts,
            desc="Predicting with Ollama",
        )
    else:
        raise ValueError(f"Unknown inference method: {args.inference_method}")

    return outputs

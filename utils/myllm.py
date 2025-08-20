import openai
import time
import numpy as np
import tiktoken
import requests
import time
import numpy as np
from transformers import AutoTokenizer

class MyChatGPT:
    def __init__(self, model_name: str, key: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=key)

    def adjust_max_tokens(self, prompt: str, buffer: int = 2000) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(enc.encode(prompt))

        model_max_tokens = {
            "gpt-4.1": 1000000,
            "gpt-4.1-mini": 1000000,
            "gpt-4.1-nano": 1000000,
            "gpt-4-turbo": 128000,
            "gpt-4-1106-preview": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0301": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-0125": 16385,
        }.get(self.model_name, 4096)

        max_tokens = max(200, min(buffer, model_max_tokens - input_tokens))

        return max_tokens

    def get_model_options(
        self,
        temperature: float = 0.0,
        per_example_max_decode_steps: int = 300,
        per_example_top_p: float = 1.0,
        n_sample: int = 1,
        prompt: str = None,
    ) -> dict:
        if prompt:
            max_tokens = self.adjust_max_tokens(prompt, per_example_max_decode_steps)
        else:
            max_tokens = per_example_max_decode_steps

        return {
            "temperature": temperature,
            "n": n_sample,
            "top_p": per_example_top_p,
            "max_tokens": max_tokens,
        }

    def generate_plus_with_score(self, prompt: str, options: dict = None, end_str: str = None) -> list:
        if options is None:
            options = self.get_model_options(prompt=prompt)

        messages = [
            {
                "role": "system",
                "content": "You are an expert in SQLite and table-based question answering. Please follow the given examples and complete the task.",
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        gpt_responses = None
        retry_num = 0
        retry_limit = 2
        error = None

        while gpt_responses is None:
            try:
                gpt_responses = self.client.chat.completions.create(
                    model=self.model_name,
                    seed = 42,
                    messages=messages,
                    stop=end_str,
                    **options
                )
                error = None
            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}", flush=True)
                error = str(e)

                if "This model's maximum context length is" in str(e):
                    print("Warning: Input exceeds max context length. Returning placeholder response.")
                    gpt_responses = {
                        "choices": [{"message": {"content": "PLACEHOLDER"}}]
                    }
                elif retry_num >= retry_limit:
                    print("Too many retry attempts. Returning placeholder response.")
                    gpt_responses = {
                        "choices": [{"message": {"content": "PLACEHOLDER"}}]
                    }
                else:
                    time.sleep(10)

                retry_num += 1

        if error:
            raise Exception(error)

        results = [(res.message.content, None) for res in gpt_responses.choices]

        return results

    def generate(self, prompt: str, options: dict = None, returnall: bool = False, end_str: str = None):
        if options is None:
            options = self.get_model_options(prompt=prompt)

        results = self.generate_plus_with_score(prompt, options, end_str)
        result_texts = [r[0] for r in results]

        if returnall:
            return result_texts
        else:
            return results[0][0]

    def generate_plus_with_score_text(self, prompt: str, options: dict = None, end_str: str = None) -> list:
        if options is None:
            options = self.get_model_options(prompt=prompt)

        messages = [
            {
                "role": "system",
                "content": "You are a semantic understanding specialist. Please follow the given examples and complete the task.",
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        gpt_responses = None
        retry_num = 0
        retry_limit = 2
        error = None

        while gpt_responses is None:
            try:
                gpt_responses = self.client.chat.completions.create(
                    model=self.model_name,
                    seed = 42,
                    messages=messages,
                    stop=end_str,
                    **options
                )
                error = None
            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}", flush=True)
                error = str(e)

                if "This model's maximum context length is" in str(e):
                    print("Warning: Input exceeds max context length. Returning placeholder response.")
                    gpt_responses = {
                        "choices": [{"message": {"content": "PLACEHOLDER"}}]
                    }
                elif retry_num >= retry_limit:
                    print("Too many retry attempts. Returning placeholder response.")
                    gpt_responses = {
                        "choices": [{"message": {"content": "PLACEHOLDER"}}]
                    }
                else:
                    time.sleep(10)

                retry_num += 1

        if error:
            raise Exception(error)

        results = [(res.message.content, None) for res in gpt_responses.choices]

        return results

    def generate_text(self, prompt: str, options: dict = None, returnall: bool = False, end_str: str = None):
        if options is None:
            options = self.get_model_options(prompt=prompt)

        results = self.generate_plus_with_score_text(prompt, options, end_str)
        result_texts = [r[0] for r in results]

        if returnall:
            return result_texts
        else:
            return results[0][0]
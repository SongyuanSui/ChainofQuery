# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import openai
import time
import numpy as np

class MyChatGPT:
    """
    A wrapper class for interacting with OpenAI's ChatGPT models.

    This class provides methods to generate text using OpenAI's GPT models,
    dynamically adjust token limits, and handle API errors efficiently.
    """
    def __init__(self, model_name: str, key: str):
        """
        Initializes the ChatGPT class with the specified model and API key.

        :param model_name: Name of the OpenAI GPT model (e.g., "gpt-4", "gpt-3.5-turbo").
        :param key: OpenAI API key for authentication.
        """
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=key)



    def get_model_options(
        self,
        temperature: float = 0.0,
        per_example_max_decode_steps: int = 300,
        per_example_top_p: float = 1.0,
        n_sample: int = 1,
        prompt: str = None,
    ) -> dict:
        """
        Generates the model configuration options for OpenAI API requests.

        :param temperature: Controls randomness; lower values make responses more deterministic (default: 0).
        :param per_example_max_decode_steps: Default max token limit before dynamic adjustment.
        :param per_example_top_p: Probability mass for top-p sampling (default: 1 for full range).
        :param n_sample: Number of responses to generate (default: 1).
        :param prompt: User input text (used for dynamic max_tokens calculation).
        :return: A dictionary containing model options.
        """
        if prompt:
            max_tokens = self.adjust_max_tokens(prompt, per_example_max_decode_steps)
        else:
            max_tokens = per_example_max_decode_steps  # Use default limit if no prompt is provided

        return {
            "temperature": temperature,
            "n": n_sample,
            "top_p": per_example_top_p,
            "max_tokens": max_tokens,
        }


    def generate_plus_with_score(self, prompt: str, options: dict = None, end_str: str = None) -> list:
        if options is None:
            options = self.get_model_options()
        messages = [
            {
                "role": "system",
                "content": "I will give you some examples, you need to follow the examples and complete the text, and no other content.",
            },
            {"role": "user", "content": prompt},
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
                    stop=end_str,  # Stop sequence (optional)
                    **options  # Pass API request options
                )
                error = None

            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}", flush=True)
                error = str(e)

                # Handle token limit errors gracefully
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
                    time.sleep(10)  # Wait for 10 seconds before retrying

                retry_num += 1  # Increment retry counter
        if error:
            raise Exception(error)
        results = []
        for i, res in enumerate(gpt_responses.choices):
            text = res.message.content
            fake_conf = (len(gpt_responses.choices) - i) / len(
                gpt_responses.choices
            )
            results.append((text, np.log(fake_conf)))

        return results

    def generate(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        options["n"] = 1
        result = self.generate_plus_with_score(prompt, options, end_str)[0][0]
        return result

class MyDeepSeek:
    """
    A wrapper class for interacting with OpenAI's ChatGPT models.

    This class provides methods to generate text using OpenAI's GPT models,
    dynamically adjust token limits, and handle API errors efficiently.
    """
    def __init__(self, model_name: str, key: str):
        """
        Initializes the ChatGPT class with the specified model and API key.

        :param model_name: Name of the OpenAI GPT model (e.g., "gpt-4", "gpt-3.5-turbo").
        :param key: OpenAI API key for authentication.
        """
        self.model_name = model_name
        self.client = openai.OpenAI(
          api_key=key,
          base_url="https://api.deepseek.com"  # 注意是 DeepSeek 的地址
        )


    def get_model_options(
        self,
        temperature: float = 0.0,
        per_example_max_decode_steps: int = 300,
        per_example_top_p: float = 1.0,
        n_sample: int = 1,
        prompt: str = None,
    ) -> dict:
        """
        Generates the model configuration options for OpenAI API requests.

        :param temperature: Controls randomness; lower values make responses more deterministic (default: 0).
        :param per_example_max_decode_steps: Default max token limit before dynamic adjustment.
        :param per_example_top_p: Probability mass for top-p sampling (default: 1 for full range).
        :param n_sample: Number of responses to generate (default: 1).
        :param prompt: User input text (used for dynamic max_tokens calculation).
        :return: A dictionary containing model options.
        """
        if prompt:
            max_tokens = self.adjust_max_tokens(prompt, per_example_max_decode_steps)
        else:
            max_tokens = per_example_max_decode_steps  # Use default limit if no prompt is provided

        return {
            "temperature": temperature,
            "n": n_sample,
            "top_p": per_example_top_p,
            "max_tokens": max_tokens,
        }


    def generate_plus_with_score(self, prompt: str, options: dict = None, end_str: str = None) -> list:
        if options is None:
            options = self.get_model_options()
        options["n"] = 1
        messages = [
            {
                "role": "system",
                "content": "I will give you some examples, you need to follow the examples and complete the text, and no other content.",
            },
            {"role": "user", "content": prompt},
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
                    stop=end_str,  # Stop sequence (optional)
                    **options  # Pass API request options
                )
                error = None

            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}", flush=True)
                error = str(e)

                # Handle token limit errors gracefully
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
                    time.sleep(10)  # Wait for 10 seconds before retrying

                retry_num += 1  # Increment retry counter
        if error:
            raise Exception(error)
        results = []
        for i, res in enumerate(gpt_responses.choices):
            text = res.message.content
            fake_conf = (len(gpt_responses.choices) - i) / len(
                gpt_responses.choices
            )
            results.append((text, np.log(fake_conf)))

        return results

    def generate(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        options["n"] = 1
        result = self.generate_plus_with_score(prompt, options, end_str)[0][0]
        return result

from typing import Any, Dict
import os
from openai import OpenAI
import json
import sys
import copy
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SimpleTemplatePrompt
from utils.utils import *


class NexusModel:
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("NEXUSFLOW_API_KEY"), base_url=os.getenv("NEXUSFLOW_BASE_URL"))

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None


class FunctionCallNexus(NexusModel):
    def __init__(self, model_name):
        super().__init__(None)
        self.model_name = model_name
        self.messages = []

    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048
            )
            return completion.choices[0].message
        except Exception as e:
            print(f"Exception: {e}")
            return None

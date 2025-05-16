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
    def __init__(self, model_name, is_reasoning=True):
        super().__init__(None)

        self.is_reasoning = is_reasoning
        if is_reasoning:
            self.reasoning_system_prompt = """Below is a reasoning template that you must strictly follow in your thinking process as you solve the problem. "
Use the template to structure your reasoning and ensure that you cover all the steps in the same order. 

Reasoning Template:
1. Problem Analysis: Restate the problem clearly and list known/unknown quantities.
2. Relevant Theorems and Knowledge: Explicitly mention relevant definitions, lemmas, or theorems needed.
3. Roadmap to Solution: Outline a clear and logical sequence of steps for solving the problem.
4. Step-by-Step Solution: Solve each step carefully, providing reasoning and intermediate results. You may use tools to perform symbolic or numerical computations when needed.
5. Verification: Check and confirm your final answer using at least one alternate method or computational verification.
"""
        else:
            self.reasoning_system_prompt = ""
        self.model_name = model_name
        self.messages = []

    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        if self.is_reasoning:
            self.messages.insert(0, {"role": "system", "content": self.reasoning_system_prompt})
        print("Hiiiii. in models/nexus.py, line 63")
        print(self.messages)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=0.0,
                tools=tools,
                tool_choice="auto",
            )
            print("Hiiiii. in models/nexus.py, line 66")
            print(completion.choices[0].message)
            return completion.choices[0].message
        except Exception as e:
            print(f"Exception: {e}")
            return None

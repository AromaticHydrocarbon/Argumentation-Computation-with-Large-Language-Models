# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
import openai
import groq
import time

class Llama3_8B:
    def __init__(self):
        self.model = "llama3-8b-8192"
        self.llm = groq.Groq(api_key="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")

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
                "response_format":{ "type": "json_object" },
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
            }
            
        return kwargs

    def completion(self, messages: List[Dict], return_json=False) -> Optional[str]:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))

        return rsp.choices[0].message.content

class Llama3_70B:
    def __init__(self):
        self.model = "llama3-70b-8192"
        self.llm = groq.Groq(api_key="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")

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
                "response_format":{ "type": "json_object" },
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
            }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> Optional[str]:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))

        return rsp.choices[0].message.content
    

class Claude3_Opus_API_tb:
    def __init__(self):
        self.model = "claude-3-opus-20240229"
        self.llm = openai.OpenAI(api_key='sk-Bbp8CePVaea06gK7Ac5bB22036Ab4577A2EbFdC8B1018a98', base_url="https://api.132999.xyz/v1")
        self.auto_max_tokens = False

    def _cons_kwargs(self, messages: List[dict]) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.3,
            "timeout": 60
        }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> Optional[str]:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages))
        except openai.RateLimitError as e: 
            print("OpenAI API request exceeded rate limit")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages))
        except openai.APITimeoutError as e:
            print("OpenAI API request timed out")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages))

        return rsp.choices[0].message.content

class OpenAI_GPT4_mini_API:
    def __init__(self):
        self.model = 'gpt-4o-mini'
        self.llm = openai.OpenAI(api_key='sk-Bbp8CePVaea06gK7Ac5bB22036Ab4577A2EbFdC8B1018a98', base_url="https://api.132999.xyz/v1")
        self.auto_max_tokens = False

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format":{ "type": "json_object" },
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> str:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.RateLimitError as e: 
            print("OpenAI API request exceeded rate limit")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APITimeoutError as e:
            print("OpenAI API request timed out")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APIConnectionError as e:
            print("OpenAI API connect error")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))


        return rsp.choices[0].message.content
    
class OpenAI_GPT4_API:
    def __init__(self):
        self.model = 'gpt-4o'
        self.llm = openai.OpenAI(api_key='sk-Bbp8CePVaea06gK7Ac5bB22036Ab4577A2EbFdC8B1018a98', base_url="https://api.132999.xyz/v1")
        self.auto_max_tokens = False

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format":{ "type": "json_object" },
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> str:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.RateLimitError as e: 
            print("OpenAI API request exceeded rate limit")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APITimeoutError as e:
            print("OpenAI API request timed out")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APIConnectionError as e:
            print("OpenAI API connect error")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))


        return rsp.choices[0].message.content

class OpenAI_GPT35_API:
    def __init__(self):
        self.model = 'gpt-3.5-turbo-0125'
        self.llm = openai.OpenAI(api_key='sk-Bbp8CePVaea06gK7Ac5bB22036Ab4577A2EbFdC8B1018a98', base_url="https://api.132999.xyz/v1")
        self.auto_max_tokens = False

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format":{ "type": "json_object" },
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> str:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.RateLimitError as e: 
            print("OpenAI API request exceeded rate limit")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APITimeoutError as e:
            print("OpenAI API request timed out")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APIConnectionError as e:
            print("OpenAI API connect error")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))


        return rsp.choices[0].message.content
    
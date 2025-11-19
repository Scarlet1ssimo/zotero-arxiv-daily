from llama_cpp import Llama
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig
from loguru import logger
from time import sleep

GLOBAL_LLM = None


class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
        if api_key:
            # self.llm = OpenAI(api_key=api_key, base_url=base_url)
            self.llm = genai.Client(api_key=api_key, http_options=types.HttpOptions(
                base_url=base_url
            ),)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict], model=None) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.responses.create(
                        input=messages, temperature=0, model=model or self.model,)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            return response.choices[0].message.content
        elif isinstance(self.llm, genai.Client):
            print("Using Google GenAI LLM")
            tools = [
                {"url_context": {}},
            ]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.models.generate_content(
                        contents=messages[1]['content'], model=model or self.model, config=GenerateContentConfig(
                            tools=tools,
                            temperature=0,
                            system_instruction=messages[0]['content'],
                        ))
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            print("GenAI response received")
            return response.text
        else:
            response = self.llm.create_chat_completion(
                messages=messages, temperature=0)
            return response["choices"][0]["message"]["content"]


def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url,
                     model=model, lang=lang)


def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info(
            "No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM

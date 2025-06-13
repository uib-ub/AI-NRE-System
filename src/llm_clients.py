import anthropic
import tiktoken
import requests
import logging

class ClaudeClient:
    """
    Client for interacting wit Anthropic Claude API
    """
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, prompt: str) -> str:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            prompt_token_count = len(enc.encode(prompt))
            print("--- Prompt Token Count ---: ", str(prompt_token_count))

            system="""You are an expert for understanding and analyzing medieval texts and manuscripts,
                    and you can markup all PROPER NOUNS in all kinds of medieval texts"""
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=3500,
                temperature=0.0,
                top_p=1.0,
                stream=False  
            )
            # DEBUG: 
            # print("--- Claude Response Text ---")
            # print(response.content[0].text)
            return response.content[0].text
        except Exception as e:
            logging.error("Claude API call failed: %s", e, exc_info=True)
            return "Claude API call failed"

class OllamaClient:
    """
    Client for interacting with Ollama via OpenWebUI API
    """
    def __init__(self, endpoint: str, token: str, model: str):
        self.endpoint = endpoint
        self.token = token
        self.model = model
    
    def call(self, prompt: str) -> str:
        try:
            # Prepare API payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p":1.0, 
                    "top_k":1,
                    "seed": 42
                }
            }
            # Prepare headers with authentication
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            }
            # Send request to OpenWebUI/Ollama
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=600
            )
            response.raise_for_status()
            # DEBUG: response 
            # print(response.json())
            return response.json().get("response", "")
        except Exception as e:
            logging.error("Ollama API call failed: %s", e, exc_info=True)
            return "Ollama API call failed"
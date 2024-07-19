import os
from openai import AzureOpenAI
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_TYPE = os.environ["AZURE_OPENAI_TYPE"]
AZURE_EMBEDDING_DEPLOYMENT = os.environ["AZURE_EMBEDDING_DEPLOYMENT"]
AZURE_EMBEDDING_MODEL = os.environ["AZURE_EMBEDDING_MODEL"]

client = AzureOpenAI(
            azure_endpoint = AZURE_ENDPOINT, 
            api_key=AZURE_OPENAI_KEY,  
            api_version=AZURE_OPENAI_API_VERSION
        )

prompt = f"""Hello
"""

message_text = [
    {"role":"system","content":"You are expert AI"},
    {"role":"user", "content":prompt}
]

completion = client.chat.completions.create(
model="VEGPT35",
messages = message_text,
temperature=0.1,
max_tokens=800,
top_p=0.95,
frequency_penalty=0,
presence_penalty=0,
stop=None
)
print(completion)
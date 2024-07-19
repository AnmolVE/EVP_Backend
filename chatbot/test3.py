import os
from openai import AzureOpenAI
 
client = AzureOpenAI(
  azure_endpoint = "https://stimulai-ve.openai.azure.com/", 
  api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_version="2024-02-15-preview"
)

message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
 
completion = client.chat.completions.create(
  model="VEGPT35", # model = "deployment_name"
  messages = message_text,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)

print(completion.choices)

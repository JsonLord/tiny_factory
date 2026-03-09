import time
import requests
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
    business_description="Selling dog toys",
    customer_profile="Dog owners",
    num_personas=1,
    api_key="sk-proj-test", # Fake key just to trigger the API loop
    api_name="/generate_personas"
)
print("Result:")
print(result)

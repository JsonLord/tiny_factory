from gradio_client import Client

# Note: We won't provide an api key here since we're just checking that the endpoint accepts the request.
# Without a valid openai API key, we should get an API Key error or a clean fallback error if one exists.
try:
    client = Client("AUXteam/UserSyncUI")
    result = client.predict(
        business_description="Selling dog toys",
        customer_profile="Dog owners",
        num_personas=1,
        api_key="sk-test",
        api_name="/generate_personas"
    )
    print("Result:")
    print(result)
except Exception as e:
    print(f"Exception: {e}")

import requests
import time
import json

def test_api():
    print("Testing call/generate_personas API locally")
    response = requests.post(
        "http://127.0.0.1:7860/gradio_api/call/generate_personas",
        json={"data": ["Selling dog toys", "Dog owners", 1, ""]}
    )
    if response.status_code == 200:
        event_id = response.json()["event_id"]
        print(f"Event ID: {event_id}")
        time.sleep(10)
        res2 = requests.get(f"http://127.0.0.1:7860/gradio_api/call/generate_personas/{event_id}")
        for line in res2.text.splitlines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    print(json.dumps(data, indent=2))
                except:
                    print(line)
    else:
        print("Failed to start")

test_api()

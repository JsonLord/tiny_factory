import sys
import os
import gradio as gr
import json
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.responses import RedirectResponse

@app.get("/api-docs")
def api_docs():
    return RedirectResponse(url="/docs")

class PersonaRequest(BaseModel):
    business_description: str
    customer_profile: str
    num_personas: int = 1

@app.post("/api/v1/generate_personas")
def generate_personas_api(req: PersonaRequest):
    return generate_personas(req.business_description, req.customer_profile, req.num_personas)

def extract_persona_parameters(business_description: str, customer_profile: str) -> dict:
    from tinytroupe.openai_utils import client

    system_prompt = """
    You are an expert persona parameter extractor.
    Based on the provided business description and customer profile, you must deduce and generate 10 specific parameters needed for a deep persona generator.
    The parameters are:
    - `age` (float): The age of the persona.
    - `gender` (str): The gender of the persona.
    - `occupation` (str): The occupation of the persona.
    - `city` (str): The city of the persona.
    - `country` (str): The country of the persona.
    - `custom_values` (str): The personal values of the persona.
    - `custom_life_attitude` (str): The life attitude of the persona.
    - `life_story` (str): A brief life story of the persona.
    - `interests_hobbies` (str): Interests and hobbies of the persona.
    - `attribute_count` (float): Attribute richness, default to 200.

    You must return a valid JSON object containing exactly these keys.
    """

    user_prompt = f"Business Description: {business_description}\nCustomer Profile: {customer_profile}\n\nReturn the 10 parameters as JSON."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    api_client = client()
    response = api_client.send_message(messages, response_format={"type": "json_object"})

    if response and "content" in response:
        try:
            # Attempt to parse it if the model returned string json
            import json
            import tinytroupe.utils as utils
            extracted_json = utils.extract_json(response["content"])

            # Ensure all keys are present
            required_keys = ['age', 'gender', 'occupation', 'city', 'country', 'custom_values', 'custom_life_attitude', 'life_story', 'interests_hobbies', 'attribute_count']

            # If extracting JSON list vs dict
            if isinstance(extracted_json, list) and len(extracted_json) > 0:
                extracted_json = extracted_json[0]

            for key in required_keys:
                if key not in extracted_json:
                    # provide defaults for missing ones
                    if key in ['age', 'attribute_count']:
                        extracted_json[key] = 200 if key == 'attribute_count' else 30
                    else:
                        extracted_json[key] = "Unknown"

            return extracted_json
        except Exception as e:
            print(f"Error parsing JSON from LLM: {e}")
            pass

    # Fallback
    return {
        "age": 30,
        "gender": "Non-binary",
        "occupation": "Professional",
        "city": "Metropolis",
        "country": "Country",
        "custom_values": "Innovation, Community",
        "custom_life_attitude": "Optimistic",
        "life_story": "A standard professional background with a passion for their field.",
        "interests_hobbies": "Technology, Reading",
        "attribute_count": 200
    }

def generate_personas(business_description, customer_profile, num_personas, blablador_api_key=None):
    """
    Generates a list of personas based on the provided inputs, utilizing a double
    sequential generation pipeline:
    1. Extract parameters from context via LLM.
    2. Generate persona using deeppersona-experience via gradio client.
    """
    api_key_to_use = blablador_api_key or os.getenv("BLABLADOR_API_KEY")

    if not api_key_to_use:
        return {"error": "BLABLADOR_API_KEY not found. Please provide it in your API call or set it as a secret in the Space settings."}

    original_key = os.getenv("BLABLADOR_API_KEY")
    os.environ["BLABLADOR_API_KEY"] = api_key_to_use
    
    try:
        from gradio_client import Client

        num_personas = int(num_personas)
        personas_data = []

        # Step 1: Extract 10 parameters based on the high-level inputs
        # For multiple personas, we could call this in a loop or once.
        # The prompt implies we want to do it in a pipeline. We'll do it per persona or once based on the prompt.
        # Let's do it per persona to generate distinct ones, passing an index or just relying on LLM variance.

        # Connect to gradio client
        # In a real scenario, the Hugging Face Token might be needed if the Space is private.
        # But deeppersona-experience is public or assumed accessible.
        client = Client("THzva/deeppersona-experience")

        for i in range(num_personas):
            # To get variety, we can append a note about variety to the profile
            profile_with_variance = customer_profile + f"\n\nMake this persona distinct. Persona {i+1} of {num_personas}."

            # Extract parameters using the LLM
            params = extract_persona_parameters(business_description, profile_with_variance)

            # Step 2: Call the Gradio API with the extracted parameters
            result = client.predict(
                age=float(params.get("age", 30)),
                gender=str(params.get("gender", "Non-binary")),
                occupation=str(params.get("occupation", "Professional")),
                city=str(params.get("city", "Metropolis")),
                country=str(params.get("country", "Country")),
                custom_values=str(params.get("custom_values", "Innovation, Community")),
                custom_life_attitude=str(params.get("custom_life_attitude", "Optimistic")),
                life_story=str(params.get("life_story", "A standard professional background with a passion for their field.")),
                interests_hobbies=str(params.get("interests_hobbies", "Technology, Reading")),
                attribute_count=float(params.get("attribute_count", 200)),
                api_name="/generate_persona"
            )

            # Note: The result from this API is a string (persona profile text)
            personas_data.append({
                "parameters_used": params,
                "persona_profile": result
            })

        return personas_data

    except Exception as e:
        return {"error": str(e)}

    finally:
        if original_key is None:
            if "BLABLADOR_API_KEY" in os.environ:
                del os.environ["BLABLADOR_API_KEY"]
        else:
            os.environ["BLABLADOR_API_KEY"] = original_key

with gr.Blocks() as demo:
    gr.Markdown("<h1>Tiny Persona Generator</h1>")
    with gr.Row():
        with gr.Column():
            business_description_input = gr.Textbox(label="What is your business about?", lines=5)
            customer_profile_input = gr.Textbox(label="Information about your customer profile", lines=5)
            num_personas_input = gr.Number(label="Number of personas to generate", value=1, minimum=1, step=1)
            
            blablador_api_key_input = gr.Textbox(
                label="Blablador API Key (for API client use)", 
                visible=False
            )

            generate_button = gr.Button("Generate Personas")
        with gr.Column():
            output_json = gr.JSON(label="Generated Personas")

    generate_button.click(
        fn=generate_personas,
        inputs=[business_description_input, customer_profile_input, num_personas_input, blablador_api_key_input],
        outputs=output_json,
        api_name="generate_personas"
    )

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

import sys
import os
import gradio as gr
import json
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.utils.semantics import select_best_persona
from huggingface_hub import hf_hub_download, upload_file
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn

HF_TOKEN = os.getenv("HF_TOKEN") # Ensure this is set in Space secrets
REPO_ID = "harvesthealth/tiny_factory"
PERSONA_BASE_FILE = "persona_base.json"

def load_persona_base():
    if not HF_TOKEN:
        print("HF_TOKEN not found, persistence disabled.")
        return []
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=PERSONA_BASE_FILE, repo_type="space", token=HF_TOKEN)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading persona base: {e}")
        return []

def save_persona_base(personas):
    if not HF_TOKEN:
        print("HF_TOKEN not found, skipping upload.")
        return
    with open(PERSONA_BASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(personas, f, indent=4)
    try:
        upload_file(
            path_or_fileobj=PERSONA_BASE_FILE,
            path_in_repo=PERSONA_BASE_FILE,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error saving persona base to Hub: {e}")

def generate_personas(business_description, customer_profile, num_personas, blablador_api_key=None):
    api_key_to_use = blablador_api_key or os.getenv("BLABLADOR_API_KEY")
    if not api_key_to_use:
        return {"error": "BLABLADOR_API_KEY not found. Please provide it in your API call or set it as a secret in the Space settings."}
    original_key = os.getenv("BLABLADOR_API_KEY")
    try:
        os.environ["BLABLADOR_API_KEY"] = api_key_to_use
        num_personas = int(num_personas)
        factory = TinyPersonFactory(
            context=business_description,
            sampling_space_description=customer_profile,
            total_population_size=num_personas
        )
        people = factory.generate_people(number_of_people=num_personas, parallelize=False)
        personas_data = [person._persona for person in people]
        current_base = load_persona_base()
        current_base.extend(personas_data)
        save_persona_base(current_base)
        return personas_data
    except Exception as e:
        return {"error": str(e)}
    finally:
        if original_key is None:
            if "BLABLADOR_API_KEY" in os.environ:
                del os.environ["BLABLADOR_API_KEY"]
        else:
            os.environ["BLABLADOR_API_KEY"] = original_key

def find_best_persona(criteria):
    personas = load_persona_base()
    if not personas:
        return {"error": "Persona base is empty. Generate some personas first!"}
    try:
        idx = select_best_persona(criteria=criteria, personas=personas)
        try:
            idx = int(idx)
        except (ValueError, TypeError):
            return {"error": f"LLM returned an invalid index: {idx}"}
        if idx >= 0 and idx < len(personas):
            return personas[idx]
        else:
            return {"error": f"No matching persona found for criteria: {criteria}"}
    except Exception as e:
        return {"error": f"Error during persona matching: {str(e)}"}

with gr.Blocks() as demo:
    gr.Markdown("<h1>Tiny Persona Generator</h1>")
    with gr.Row():
        with gr.Column():
            business_description_input = gr.Textbox(label="What is your business about?", lines=5)
            customer_profile_input = gr.Textbox(label="Information about your customer profile", lines=5)
            num_personas_input = gr.Number(label="Number of personas to generate", value=1, minimum=1, step=1)
            blablador_api_key_input = gr.Textbox(label="Blablador API Key (for API client use)", visible=False)
            generate_button = gr.Button("Generate Personas")
            gr.Markdown("---")
            gr.Markdown("<h3>Search Tresor</h3>")
            criteria_input = gr.Textbox(label="Criteria to find best matching persona", lines=2)
            find_button = gr.Button("Find Best Persona in Tresor")
        with gr.Column():
            output_json = gr.JSON(label="Output (Generated or Matched Persona)")
    generate_button.click(
        fn=generate_personas,
        inputs=[business_description_input, customer_profile_input, num_personas_input, blablador_api_key_input],
        outputs=output_json,
        api_name="generate_personas"
    )
    find_button.click(
        fn=find_best_persona,
        inputs=[criteria_input],
        outputs=output_json,
        api_name="find_best_persona"
    )

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api-docs")
def api_docs():
    return RedirectResponse(url="/docs")

# Disabling SSR to fix the port 7861 error in Spaces
try:
    app = gr.mount_gradio_app(app, demo, path="/", ssr=False)
except Exception:
    app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

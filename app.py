import sys
import os
import gradio as gr
import json
import concurrent.futures
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.utils.semantics import select_best_persona
from huggingface_hub import hf_hub_download, upload_file
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import tinytroupe.openai_utils as openai_utils

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
    temp_file = "persona_base_upload.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(personas, f, indent=4)
    try:
        upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=PERSONA_BASE_FILE,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN
        )
        print("Persona base saved successfully to Hub.")
    except Exception as e:
        print(f"Error saving persona base to Hub: {e}")

def render_personas_to_markdown(personas):
    if not personas:
        return ""
    md = "## 👤 Generated Personas\n\n"
    for i, p in enumerate(personas):
        name = p.get('name', 'Unknown')
        age = p.get('age', 'N/A')
        gender = p.get('gender', 'N/A')
        nationality = p.get('nationality', 'N/A')
        occupation = p.get('occupation', 'N/A')
        description = p.get('description', 'N/A')

        md += f"### {i+1}. {name}\n"
        md += f"**Age**: {age} | **Gender**: {gender} | **Nationality**: {nationality}\n\n"
        md += f"**Occupation**: {occupation}\n\n"
        md += f"**Description**: {description}\n\n"
        md += f"<details><summary>View Full JSON</summary>\n\n```json\n{json.dumps(p, indent=2)}\n```\n\n</details>\n\n"
        md += "---\n"
    return md

def generate_personas(business_description, customer_profile, num_personas, model_choice, blablador_api_key=None):
    api_key_to_use = blablador_api_key or os.getenv("BLABLADOR_API_KEY")
    if not api_key_to_use:
        yield {"error": "BLABLADOR_API_KEY missing"}, "### ❌ Error: BLABLADOR_API_KEY not found.", gr.update(visible=False)
        return

    # Set model choice in config
    openai_utils.config["OpenAI"]["MODEL"] = model_choice
    openai_utils.config["OpenAI"]["REASONING_MODEL"] = model_choice

    original_key = os.getenv("BLABLADOR_API_KEY")
    os.environ["BLABLADOR_API_KEY"] = api_key_to_use

    all_personas_data = []
    try:
        num_personas = int(num_personas)
        yield [], f"Initializing sampling plan using **{model_choice}**... ⏳", gr.update(visible=True)

        factory = TinyPersonFactory(
            context=business_description,
            sampling_space_description=customer_profile,
            total_population_size=num_personas
        )

        # Force initialization to show progress
        factory.initialize_sampling_plan()
        yield [], f"Sampling plan initialized. Generating {num_personas} personas in parallel... 🚀", gr.update(visible=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_personas, 5)) as executor:
            futures = [executor.submit(factory.generate_person) for _ in range(num_personas)]

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                person = future.result()
                if person:
                    persona_data = person._persona
                    all_personas_data.append(persona_data)

                status_msg = f"#### 🔄 Generated {completed}/{num_personas} personas... ⏳\n\n"
                yield all_personas_data, status_msg + render_personas_to_markdown(all_personas_data), gr.update(visible=True)

        yield all_personas_data, "### ✅ Generation complete! All personas saved to Tresor.\n\n" + render_personas_to_markdown(all_personas_data), gr.update(visible=False)

    except GeneratorExit:
        print("Generation cancelled by user.")
    except Exception as e:
        yield {"error": str(e)}, f"### ❌ Error\n{str(e)}", gr.update(visible=False)
    finally:
        if all_personas_data:
            print(f"Saving {len(all_personas_data)} personas...")
            try:
                current_base = load_persona_base()
                current_base.extend(all_personas_data)
                save_persona_base(current_base)
            except Exception as se:
                print(f"Error during final save: {se}")

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
    gr.Markdown("# 🏭 Tiny Persona Factory")
    gr.Markdown("Generate realistic personas for your business simulation. Results are automatically saved to the Tresor.")

    with gr.Row():
        with gr.Column(scale=1):
            business_description_input = gr.Textbox(label="Business Context", placeholder="e.g., A new coffee shop in Berlin", lines=3)
            customer_profile_input = gr.Textbox(label="Customer Profile", placeholder="e.g., Students and young professionals", lines=3)

            with gr.Row():
                num_personas_input = gr.Number(label="Number of Personas", value=1, minimum=1, step=1)
                model_choice = gr.Dropdown(
                    label="Model",
                    choices=["alias-fast", "alias-large", "alias-huge"],
                    value="alias-huge"
                )

            blablador_api_key_input = gr.Textbox(label="API Key (Optional)", type="password", visible=False)

            with gr.Row():
                generate_button = gr.Button("🚀 Generate", variant="primary")
                stop_button = gr.Button("🛑 Stop and Save", variant="stop", visible=False)

            gr.Markdown("---")
            gr.Markdown("### 🔍 Search Tresor")
            criteria_input = gr.Textbox(label="Criteria", placeholder="e.g., Find someone who likes dark roast", lines=2)
            find_button = gr.Button("🔍 Find Best Match")

        with gr.Column(scale=2):
            rendered_output = gr.Markdown("Personas will appear here...")
            output_json = gr.JSON(label="Data", visible=False)

    gen_event = generate_button.click(
        fn=generate_personas,
        inputs=[business_description_input, customer_profile_input, num_personas_input, model_choice, blablador_api_key_input],
        outputs=[output_json, rendered_output, stop_button],
        api_name="generate_personas"
    )

    stop_button.click(fn=None, cancels=[gen_event])

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

try:
    app = gr.mount_gradio_app(app, demo, path="/", ssr=False)
except Exception:
    app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

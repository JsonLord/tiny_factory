import sys
import os
import gradio as gr
import json
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.utils.semantics import select_best_persona
from tinytroupe.simulation_manager import SimulationManager, SimulationConfig
from tinytroupe.agent.social_types import Content
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN") # Ensure this is set in Space secrets
REPO_ID = "harvesthealth/tiny_factory"
PERSONA_BASE_FILE = "persona_base.json"

simulation_manager = SimulationManager()

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

# --- CHANGE 1: The function now accepts an optional API key. ---
def generate_personas(business_description, customer_profile, num_personas, blablador_api_key=None):
    """
    Generates a list of TinyPerson instances based on the provided inputs.
    It prioritizes the API key passed as an argument, but falls back to the
    environment variable if none is provided (for UI use).
    """
    # --- CHANGE 2: Logic to determine which key to use. ---
    # Use the key from the API call if provided, otherwise get it from the Space secrets.
    api_key_to_use = blablador_api_key or os.getenv("BLABLADOR_API_KEY")

    if not api_key_to_use:
        return {"error": "BLABLADOR_API_KEY not found. Please provide it in your API call or set it as a secret in the Space settings."}

    # Store the original state of the environment variable, if it exists
    original_key = os.getenv("BLABLADOR_API_KEY")
    
    try:
        # --- CHANGE 3: Securely set the correct environment variable for this request. ---
        # The underlying tinytroupe library will look for this variable.
        os.environ["BLABLADOR_API_KEY"] = api_key_to_use

        num_personas = int(num_personas)

        factory = TinyPersonFactory(
            context=business_description,
            sampling_space_description=customer_profile,
            total_population_size=num_personas
        )

        people = factory.generate_people(number_of_people=num_personas, parallelize=False)
        personas_data = [person._persona for person in people]
        
        # --- NEW: Update the Tresor ---
        current_base = load_persona_base()
        current_base.extend(personas_data)
        save_persona_base(current_base)
        # ------------------------------

        return personas_data

    except Exception as e:
        return {"error": str(e)}

    finally:
        # --- CHANGE 4: A robust cleanup using a 'finally' block. ---
        # This ensures the environment is always restored to its original state,
        # whether the function succeeds or fails.
        if original_key is None:
            # If the variable didn't exist originally, remove it.
            if "BLABLADOR_API_KEY" in os.environ:
                del os.environ["BLABLADOR_API_KEY"]
        else:
            # If it existed, restore its original value.
            os.environ["BLABLADOR_API_KEY"] = original_key


def find_best_persona(criteria):
    """
    Loads the persona base and finds the best matching persona based on criteria.
    """
    personas = load_persona_base()
    if not personas:
        return {"error": "Persona base is empty. Generate some personas first!"}

    try:
        # select_best_persona uses LLM to find the best index
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


def generate_social_network_api(name, persona_count, network_type):
    """
    Gradio API endpoint for generating a social network.
    """
    try:
        config = SimulationConfig(name=name, persona_count=int(persona_count), network_type=network_type)
        simulation = simulation_manager.create_simulation(config)
        return {
            "simulation_id": simulation.id,
            "name": simulation.config.name,
            "persona_count": len(simulation.personas),
            "network_metrics": simulation.network.get_metrics()
        }
    except Exception as e:
        return {"error": str(e)}


def predict_engagement_api(simulation_id, content_text):
    """
    Gradio API endpoint for predicting engagement.
    """
    try:
        content = Content(text=content_text)
        result = simulation_manager.run_simulation(simulation_id, content)
        return {
            "total_reach": result.total_reach,
            "expected_likes": result.expected_likes,
            "expected_comments": result.expected_comments,
            "expected_shares": result.expected_shares,
            "execution_time": result.execution_time
        }
    except Exception as e:
        return {"error": str(e)}


def generate_variants_api(content_text, num_variants):
    """
    Gradio API endpoint for generating content variants.
    """
    try:
        variants = simulation_manager.variant_generator.generate_variants(content_text, num_variants=int(num_variants))
        return [{"text": v.text, "strategy": v.strategy} for v in variants]
    except Exception as e:
        return {"error": str(e)}


def list_simulations_api():
    """
    Gradio API endpoint for listing simulations.
    """
    try:
        return simulation_manager.list_simulations()
    except Exception as e:
        return {"error": str(e)}


def list_personas_api(simulation_id):
    """
    Gradio API endpoint for listing personas in a simulation.
    """
    try:
        return simulation_manager.list_personas(simulation_id)
    except Exception as e:
        return {"error": str(e)}


def get_persona_api(simulation_id, persona_name):
    """
    Gradio API endpoint for getting persona details.
    """
    try:
        return simulation_manager.get_persona(simulation_id, persona_name)
    except Exception as e:
        return {"error": str(e)}


def delete_simulation_api(simulation_id):
    """
    Gradio API endpoint for deleting a simulation.
    """
    try:
        success = simulation_manager.delete_simulation(simulation_id)
        return {"success": success}
    except Exception as e:
        return {"error": str(e)}


def export_simulation_api(simulation_id):
    """
    Gradio API endpoint for exporting a simulation.
    """
    try:
        return simulation_manager.export_simulation(simulation_id)
    except Exception as e:
        return {"error": str(e)}


with gr.Blocks() as demo:
    gr.Markdown("<h1>Tiny Persona Generator</h1>")
    with gr.Row():
        with gr.Column():
            business_description_input = gr.Textbox(label="What is your business about?", lines=5)
            customer_profile_input = gr.Textbox(label="Information about your customer profile", lines=5)
            num_personas_input = gr.Number(label="Number of personas to generate", value=1, minimum=1, step=1)
            
            # --- CHANGE 5: The API key input is now INVISIBLE. ---
            # It still exists, so the API endpoint is created, but it's hidden from UI users.
            blablador_api_key_input = gr.Textbox(
                label="Blablador API Key (for API client use)", 
                visible=False
            )

            generate_button = gr.Button("Generate Personas")

            gr.Markdown("---")
            gr.Markdown("<h3>Search Tresor</h3>")
            criteria_input = gr.Textbox(label="Criteria to find best matching persona", lines=2)
            find_button = gr.Button("Find Best Persona in Tresor")

        with gr.Column():
            output_json = gr.JSON(label="Output (Generated or Matched Persona)")

    generate_button.click(
        fn=generate_personas,
        # --- CHANGE 6: Pass the invisible textbox to the function. ---
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

    # Invisible components to expose API endpoints
    # These won't be seen by regular UI users but will be available via /api
    with gr.Tab("Social Network API", visible=False):
        api_net_name = gr.Textbox(label="Network Name")
        api_net_count = gr.Number(label="Persona Count", value=10)
        api_net_type = gr.Dropdown(choices=["scale_free", "small_world"], label="Network Type")
        api_net_btn = gr.Button("Generate Network")
        api_net_out = gr.JSON()
        api_net_btn.click(generate_social_network_api, inputs=[api_net_name, api_net_count, api_net_type], outputs=api_net_out, api_name="generate_social_network")

    with gr.Tab("Engagement Prediction API", visible=False):
        api_pred_sim_id = gr.Textbox(label="Simulation ID")
        api_pred_content = gr.Textbox(label="Content Text")
        api_pred_btn = gr.Button("Predict Engagement")
        api_pred_out = gr.JSON()
        api_pred_btn.click(predict_engagement_api, inputs=[api_pred_sim_id, api_pred_content], outputs=api_pred_out, api_name="predict_engagement")

    with gr.Tab("Content Variants API", visible=False):
        api_var_content = gr.Textbox(label="Original Content")
        api_var_count = gr.Number(label="Number of Variants", value=5)
        api_var_btn = gr.Button("Generate Variants")
        api_var_out = gr.JSON()
        api_var_btn.click(generate_variants_api, inputs=[api_var_content, api_var_count], outputs=api_var_out, api_name="generate_variants")

    with gr.Tab("List Simulations API", visible=False):
        api_list_sim_btn = gr.Button("List Simulations")
        api_list_sim_out = gr.JSON()
        api_list_sim_btn.click(list_simulations_api, outputs=api_list_sim_out, api_name="list_simulations")

    with gr.Tab("List Personas API", visible=False):
        api_list_per_sim_id = gr.Textbox(label="Simulation ID")
        api_list_per_btn = gr.Button("List Personas")
        api_list_per_out = gr.JSON()
        api_list_per_btn.click(list_personas_api, inputs=[api_list_per_sim_id], outputs=api_list_per_out, api_name="list_personas")

    with gr.Tab("Get Persona API", visible=False):
        api_get_per_sim_id = gr.Textbox(label="Simulation ID")
        api_get_per_name = gr.Textbox(label="Persona Name")
        api_get_per_btn = gr.Button("Get Persona")
        api_get_per_out = gr.JSON()
        api_get_per_btn.click(get_persona_api, inputs=[api_get_per_sim_id, api_get_per_name], outputs=api_get_per_out, api_name="get_persona")

    with gr.Tab("Delete Simulation API", visible=False):
        api_del_sim_id = gr.Textbox(label="Simulation ID")
        api_del_btn = gr.Button("Delete Simulation")
        api_del_out = gr.JSON()
        api_del_btn.click(delete_simulation_api, inputs=[api_del_sim_id], outputs=api_del_out, api_name="delete_simulation")

    with gr.Tab("Export Simulation API", visible=False):
        api_exp_sim_id = gr.Textbox(label="Simulation ID")
        api_exp_btn = gr.Button("Export Simulation")
        api_exp_out = gr.JSON()
        api_exp_btn.click(export_simulation_api, inputs=[api_exp_sim_id], outputs=api_exp_out, api_name="export_simulation")

if __name__ == "__main__":
    demo.queue().launch()
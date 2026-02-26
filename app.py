import sys
import os
import gradio as gr
import json
import glob
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.utils.semantics import select_best_persona, select_relevant_personas_utility
from tinytroupe.simulation_manager import SimulationManager, SimulationConfig
from tinytroupe.agent.social_types import Content
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN") # Ensure this is set in Space secrets
REPO_ID = "AUXteam/tiny_factory"
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

        # Restricted to deep persona generation with double sequential API call
        people = factory.generate_people(number_of_people=num_personas, parallelize=False, deep_persona=True)
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


def load_example_personas():
    """
    Loads example personas from the tinytroupe library.
    """
    example_personas = []
    # Path to the agents folder in tinytroupe/examples
    agents_path = os.path.join("tinytroupe", "examples", "agents", "*.agent.json")
    for file_path in glob.glob(agents_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "persona" in data:
                    example_personas.append(data["persona"])
        except Exception as e:
            print(f"Error loading example persona from {file_path}: {e}")
    return example_personas


def identify_personas(context):
    """
    Identifies appropriate personas from the Tresor and example agents based on context.
    """
    try:
        # 1. Load Tresor personas (persisted JSON)
        tresor_personas = load_persona_base()

        # 2. Load Example personas from tinytroupe library
        example_personas = load_example_personas()

        all_available = tresor_personas + example_personas

        if not all_available:
            return {"error": "No personas available in Tresor or examples."}

        # 3. Use LLM to filter/select which ones match the 'context'
        # Returns a list of indices
        indices = select_relevant_personas_utility(context, all_available)

        selected = []
        if isinstance(indices, list):
            for i in indices:
                try:
                    idx = int(i)
                    if 0 <= idx < len(all_available):
                        selected.append(all_available[idx])
                except (ValueError, TypeError):
                    continue

        return selected
    except Exception as e:
        return {"error": str(e)}


def generate_social_network_api(name, persona_count, network_type, focus_group_name=None):
    """
    Gradio API endpoint for generating a social network.
    """
    try:
        config = SimulationConfig(name=name, persona_count=int(persona_count), network_type=network_type)
        simulation = simulation_manager.create_simulation(config, focus_group_name=focus_group_name)
        return {
            "simulation_id": simulation.id,
            "name": simulation.config.name,
            "persona_count": len(simulation.personas),
            "network_metrics": simulation.network.get_metrics()
        }
    except Exception as e:
        return {"error": str(e)}


def predict_engagement_api(simulation_id, content_text, format="text"):
    """
    Gradio API endpoint for predicting engagement.
    """
    try:
        content = Content(text=content_text, format=format)
        result = simulation_manager.run_simulation(simulation_id, content)
        return {
            "total_reach": result.total_reach,
            "expected_likes": result.expected_likes,
            "expected_comments": result.expected_comments,
            "expected_shares": result.expected_shares,
            "execution_time": result.execution_time,
            "avg_sentiment": result.avg_sentiment,
            "feedback_summary": result.feedback_summary
        }
    except Exception as e:
        return {"error": str(e)}


def start_simulation_async_api(simulation_id, content_text, format="text"):
    """
    Starts a simulation in the background.
    """
    try:
        content = Content(text=content_text, format=format)
        simulation_manager.run_simulation(simulation_id, content, background=True)
        return {"status": "started", "simulation_id": simulation_id}
    except Exception as e:
        return {"error": str(e)}


def get_simulation_status_api(simulation_id):
    """
    Checks the status and progress of a simulation.
    """
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return {"error": "Simulation not found"}

        status_data = {
            "status": sim.status,
            "progress": sim.progress
        }

        if sim.status == "completed" and sim.last_result:
            status_data["result"] = {
                "total_reach": sim.last_result.total_reach,
                "expected_likes": sim.last_result.expected_likes,
                "avg_sentiment": sim.last_result.avg_sentiment
            }

        return status_data
    except Exception as e:
        return {"error": str(e)}


def send_chat_message_api(simulation_id, sender, message):
    """
    Sends a message to the simulation chat.
    """
    try:
        return simulation_manager.send_chat_message(simulation_id, sender, message)
    except Exception as e:
        return {"error": str(e)}


def get_chat_history_api(simulation_id):
    """
    Gets the chat history for a simulation.
    """
    try:
        return simulation_manager.get_chat_history(simulation_id)
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


def get_network_graph_api(simulation_id):
    """
    Gradio API endpoint for getting network graph data.
    """
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return {"error": "Simulation not found"}

        nodes = []
        for p in sim.personas:
            nodes.append({
                "id": p.name,
                "label": p.name,
                "role": p._persona.get("occupation"),
                "location": p._persona.get("residence")
            })

        edges = []
        for edge in sim.network.edges:
            edges.append({
                "source": edge.connection_id.split('_')[0],
                "target": edge.connection_id.split('_')[1],
                "strength": edge.strength
            })

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"error": str(e)}


def list_focus_groups_api():
    """
    Gradio API endpoint for listing focus groups.
    """
    try:
        return simulation_manager.list_focus_groups()
    except Exception as e:
        return {"error": str(e)}


def save_focus_group_api(name, simulation_id):
    """
    Gradio API endpoint for saving a focus group from a simulation.
    """
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return {"error": "Simulation not found"}
        simulation_manager.save_focus_group(name, sim.personas)
        return {"status": "success", "name": name}
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

    with gr.Tab("Identify Personas API", visible=False):
        api_id_context = gr.Textbox(label="Context")
        api_id_btn = gr.Button("Identify Personas")
        api_id_out = gr.JSON()
        api_id_btn.click(identify_personas, inputs=[api_id_context], outputs=api_id_out, api_name="identify_personas")

    # Invisible components to expose API endpoints
    # These won't be seen by regular UI users but will be available via /api
    with gr.Tab("Social Network API", visible=False):
        api_net_name = gr.Textbox(label="Network Name")
        api_net_count = gr.Number(label="Persona Count", value=10)
        api_net_type = gr.Dropdown(choices=["scale_free", "small_world"], label="Network Type")
        api_net_focus = gr.Textbox(label="Focus Group Name (optional)")
        api_net_btn = gr.Button("Generate Network")
        api_net_out = gr.JSON()
        api_net_btn.click(generate_social_network_api, inputs=[api_net_name, api_net_count, api_net_type, api_net_focus], outputs=api_net_out, api_name="generate_social_network")

    with gr.Tab("Engagement Prediction API", visible=False):
        api_pred_sim_id = gr.Textbox(label="Simulation ID")
        api_pred_content = gr.Textbox(label="Content Text")
        api_pred_format = gr.Textbox(label="Format", value="text")
        api_pred_btn = gr.Button("Predict Engagement")
        api_pred_out = gr.JSON()
        api_pred_btn.click(predict_engagement_api, inputs=[api_pred_sim_id, api_pred_content, api_pred_format], outputs=api_pred_out, api_name="predict_engagement")

    with gr.Tab("Async Simulation API", visible=False):
        api_async_sim_id = gr.Textbox(label="Simulation ID")
        api_async_content = gr.Textbox(label="Content Text")
        api_async_format = gr.Textbox(label="Format", value="text")
        api_async_btn = gr.Button("Start Simulation")
        api_async_out = gr.JSON()
        api_async_btn.click(start_simulation_async_api, inputs=[api_async_sim_id, api_async_content, api_async_format], outputs=api_async_out, api_name="start_simulation_async")

        api_status_id = gr.Textbox(label="Simulation ID")
        api_status_btn = gr.Button("Check Status")
        api_status_out = gr.JSON()
        api_status_btn.click(get_simulation_status_api, inputs=[api_status_id], outputs=api_status_out, api_name="get_simulation_status")

    with gr.Tab("Chat API", visible=False):
        api_chat_sim_id = gr.Textbox(label="Simulation ID")
        api_chat_sender = gr.Textbox(label="Sender", value="User")
        api_chat_msg = gr.Textbox(label="Message")
        api_chat_send_btn = gr.Button("Send Message")
        api_chat_send_out = gr.JSON()
        api_chat_send_btn.click(send_chat_message_api, inputs=[api_chat_sim_id, api_chat_sender, api_chat_msg], outputs=api_chat_send_out, api_name="send_chat_message")

        api_chat_hist_btn = gr.Button("Get History")
        api_chat_hist_out = gr.JSON()
        api_chat_hist_btn.click(get_chat_history_api, inputs=[api_chat_sim_id], outputs=api_chat_hist_out, api_name="get_chat_history")

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

    with gr.Tab("Network Graph API", visible=False):
        api_graph_sim_id = gr.Textbox(label="Simulation ID")
        api_graph_btn = gr.Button("Get Graph Data")
        api_graph_out = gr.JSON()
        api_graph_btn.click(get_network_graph_api, inputs=[api_graph_sim_id], outputs=api_graph_out, api_name="get_network_graph")

    with gr.Tab("Focus Group API", visible=False):
        api_list_fg_btn = gr.Button("List Focus Groups")
        api_list_fg_out = gr.JSON()
        api_list_fg_btn.click(list_focus_groups_api, outputs=api_list_fg_out, api_name="list_focus_groups")

        api_save_fg_name = gr.Textbox(label="Focus Group Name")
        api_save_fg_sim_id = gr.Textbox(label="Simulation ID")
        api_save_fg_btn = gr.Button("Save Focus Group")
        api_save_fg_out = gr.JSON()
        api_save_fg_btn.click(save_focus_group_api, inputs=[api_save_fg_name, api_save_fg_sim_id], outputs=api_save_fg_out, api_name="save_focus_group")

if __name__ == "__main__":
    demo.queue().launch()
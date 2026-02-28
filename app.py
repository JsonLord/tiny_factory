import sys
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
import json
import glob
from deeppersona.factory import DeepPersonaFactory
from deeppersona.utils.semantics import select_best_persona, select_relevant_personas_utility
from deeppersona.simulation_manager import SimulationManager, SimulationConfig
from deeppersona.agent.social_types import Content
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
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
    if not HF_TOKEN: return
    try:
        with open(PERSONA_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=4)
        upload_file(
            path_or_fileobj=PERSONA_BASE_FILE,
            path_in_repo=PERSONA_BASE_FILE,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error saving persona base: {e}")

def generate_personas(business_description, customer_profile, num_personas, blablador_key=None):
    if blablador_key:
        os.environ["BLABLADOR_API_KEY"] = blablador_key

    context = f"Business: {business_description}\nTarget: {customer_profile}"
    factory = DeepPersonaFactory(context=context)

    # Generate personas
    generated = factory.generate_people(number_of_people=int(num_personas))

    # Store in base
    base = load_persona_base()
    for p in generated:
        base.append({
            "name": p.name,
            "persona": p._persona,
            "minibio": p.minibio()
        })
    save_persona_base(base)

    return [p._persona for p in generated]

def find_best_persona(criteria):
    base = load_persona_base()
    if not base: return {"error": "No personas in base"}

    # Simple semantic search using the utility
    best = select_best_persona(base, criteria)
    return best

def identify_personas(context):
    """
    Search for relevant personas across both Tresor and internal examples.
    """
    base = load_persona_base()
    # Add example agents from disk
    example_files = glob.glob("deeppersona/examples/agents/*.json")
    for ef in example_files:
        try:
            with open(ef, 'r') as f:
                data = json.load(f)
                base.append({
                    "name": data.get("name", "Unknown"),
                    "persona": data.get("persona", {}),
                    "minibio": "Example agent"
                })
        except: pass

    relevant = select_relevant_personas_utility(base, context)
    return relevant

# API Wrappers for SimulationManager
def generate_social_network_api(name, persona_count, network_type, focus_group_name=None):
    try:
        config = SimulationConfig(name=name, persona_count=int(persona_count), network_type=network_type)
        simulation = simulation_manager.create_simulation(config, focus_group_name=focus_group_name)
        return {"simulation_id": simulation.id, "status": "created"}
    except Exception as e:
        return {"error": str(e)}

def predict_engagement_api(simulation_id, content_text, format="text"):
    try:
        content = Content(text=content_text, format=format)
        results = simulation_manager.predict_engagement(simulation_id, content)
        return results
    except Exception as e:
        return {"error": str(e)}

def start_simulation_async_api(simulation_id, content_text, format="text"):
    try:
        content = Content(text=content_text, format=format)
        simulation_manager.run_simulation(simulation_id, content, background=True)
        return {"status": "started", "simulation_id": simulation_id}
    except Exception as e:
        return {"error": str(e)}

def get_simulation_status_api(simulation_id):
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return {"error": "Not found"}
        return {
            "status": sim.status,
            "progress": sim.progress,
            "result_ready": sim.result is not None
        }
    except Exception as e:
        return {"error": str(e)}

def send_chat_message_api(simulation_id, sender, message):
    try:
        res = simulation_manager.chat_with_simulation(simulation_id, sender, message)
        return res
    except Exception as e:
        return {"error": str(e)}

def get_chat_history_api(simulation_id):
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return {"error": "Not found"}
        return sim.chat_history
    except Exception as e:
        return {"error": str(e)}

def generate_variants_api(content_text, count=5):
    try:
        content = Content(text=content_text)
        variants = simulation_manager.generate_content_variants(content, int(count))
        return [v.text for v in variants]
    except Exception as e:
        return {"error": str(e)}

def list_simulations_api():
    return list(simulation_manager.simulations.keys())

def list_personas_api(simulation_id):
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return []
        return [p.name for p in sim.personas]
    except Exception as e:
        return {"error": str(e)}

def get_persona_api(simulation_id, persona_name):
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return None
        for p in sim.personas:
            if p.name == persona_name: return p._persona
        return None
    except Exception as e:
        return {"error": str(e)}

def delete_simulation_api(simulation_id):
    try:
        success = simulation_manager.delete_simulation(simulation_id)
        return {"success": success}
    except Exception as e:
        return {"error": str(e)}

def export_simulation_api(simulation_id):
    try:
        return simulation_manager.export_simulation(simulation_id)
    except Exception as e:
        return {"error": str(e)}

def get_network_graph_api(simulation_id):
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
    try:
        return simulation_manager.list_focus_groups()
    except Exception as e:
        return {"error": str(e)}

def save_focus_group_api(name, simulation_id):
    try:
        sim = simulation_manager.get_simulation(simulation_id)
        if not sim: return {"error": "Simulation not found"}
        simulation_manager.save_focus_group(name, sim.personas)
        return {"status": "success", "name": name}
    except Exception as e:
        return {"error": str(e)}

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1>Deep Persona Generator</h1>")
    with gr.Row():
        with gr.Column():
            business_description_input = gr.Textbox(label="What is your business about?", lines=5)
            customer_profile_input = gr.Textbox(label="Information about your customer profile", lines=5)
            num_personas_input = gr.Number(label="Number of personas to generate", value=1, minimum=1, step=1)
            blablador_api_key_input = gr.Textbox(label="Blablador API Key (for API client use)", visible=False)
            generate_button = gr.Button("Generate Deep Personas")
            gr.Markdown("---")
            gr.Markdown("<h3>Search Tresor</h3>")
            criteria_input = gr.Textbox(label="Criteria to find best matching persona", lines=2)
            find_button = gr.Button("Find Best Deep Persona in Tresor")
        with gr.Column():
            output_json = gr.JSON(label="Output (Generated or Matched Deep Persona)")

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

    with gr.Tab("Identify Deep Personas API", visible=False):
        api_id_context = gr.Textbox(label="Context")
        api_id_btn = gr.Button("Identify Deep Personas")
        api_id_out = gr.JSON()
        api_id_btn.click(identify_personas, inputs=[api_id_context], outputs=api_id_out, api_name="identify_personas")

    with gr.Tab("Social Network API", visible=False):
        api_net_name = gr.Textbox(label="Network Name")
        api_net_count = gr.Number(label="Deep Persona Count", value=10)
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

    with gr.Tab("List Deep Personas API", visible=False):
        api_list_per_sim_id = gr.Textbox(label="Simulation ID")
        api_list_per_btn = gr.Button("List Deep Personas")
        api_list_per_out = gr.JSON()
        api_list_per_btn.click(list_personas_api, inputs=[api_list_per_sim_id], outputs=api_list_per_out, api_name="list_personas")

    with gr.Tab("Get Deep Persona API", visible=False):
        api_get_per_sim_id = gr.Textbox(label="Simulation ID")
        api_get_per_name = gr.Textbox(label="Deep Persona Name")
        api_get_per_btn = gr.Button("Get Deep Persona")
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

# FastAPI App
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api-docs")
def api_docs():
    return RedirectResponse(url="/docs")

# Mount Gradio
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

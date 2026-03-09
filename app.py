import gradio as gr
import pandas as pd
import json
import random
import requests
import os
from datetime import datetime
from tinytroupe.simulation_manager import SimulationManager, SimulationConfig
from tinytroupe.agent.social_types import Content
from tinytroupe.agent import TinyPerson
import tinytroupe.openai_utils as openai_utils

# Initialize Simulation Manager
simulation_manager = SimulationManager()
REMOTE_BACKEND = "https://auxteam-tiny-factory.hf.space"

def generate_personas(business_description, customer_profile, num_personas, api_key=None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    import json
    import random
    from gradio_client import Client
    import openai
    from tinytroupe.agent import TinyPerson

    # Initialize the OpenAI client pointing to the Helmholtz endpoint
    client = openai.OpenAI(
        base_url="https://api.helmholtz-blablador.fz-juelich.de/v1",
        api_key=api_key or os.environ.get("BLABLADOR_API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
    )
    dp_client = Client("THzva/deeppersona-experience")

    personas = []

    for i in range(int(num_personas)):
        # 1. Generate initial parameters for the 200 API call
        prompt_1 = f"""
Given the following business description and customer profile:
Business: {business_description}
Customer: {customer_profile}

Generate realistic parameters for a persona. Return ONLY a valid JSON object with these EXACT keys (do not wrap in markdown blocks):
{{"Age": 30, "Gender": "Female", "Occupation": "Teacher", "City": "Berlin", "Country": "Germany", "Personal Values": "Hard work", "Life Attitude": "Optimistic", "Life Story": "Born in Munich", "Interests and Hobbies": "Reading"}}
"""
        try:
            response_1 = client.chat.completions.create(
                model="alias-fast",
                messages=[{"role": "user", "content": prompt_1}],
                temperature=0.7
            )
            raw_content = response_1.choices[0].message.content.strip()
            # Clean up markdown code blocks if the model hallucinates them
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()
            params_1 = json.loads(raw_content)
        except Exception as e:
            print("Fallback for params_1 due to error:", e)
            params_1 = {}

        # 2. Call DeepPersona with 200 attributes
        result_200 = dp_client.predict(
            age=params_1.get("Age", 30),
            gender=params_1.get("Gender", "Female"),
            occupation=params_1.get("Occupation", "Professional"),
            city=params_1.get("City", "New York"),
            country=params_1.get("Country", "USA"),
            custom_values=params_1.get("Personal Values", "Hardworking"),
            custom_life_attitude=params_1.get("Life Attitude", "Positive"),
            life_story=params_1.get("Life Story", "Grew up in the city"),
            interests_hobbies=params_1.get("Interests and Hobbies", "Reading"),
            attribute_count=200,
            api_name="/generate_persona"
        )

        # 3. Use LLM to extract specific truth/details from 200 output for the 400 call
        prompt_2 = f"""
Based on this generated persona output:
{result_200}

Extract and enhance specific details to create an updated set of parameters. Return ONLY a valid JSON object with these EXACT keys (do not wrap in markdown blocks):
{{"Age": 30, "Gender": "Female", "Occupation": "Teacher", "City": "Berlin", "Country": "Germany", "Personal Values": "Hard work", "Life Attitude": "Optimistic", "Life Story": "Born in Munich", "Interests and Hobbies": "Reading"}}
"""
        try:
            response_2 = client.chat.completions.create(
                model="alias-fast",
                messages=[{"role": "user", "content": prompt_2}],
                temperature=0.7
            )
            raw_content2 = response_2.choices[0].message.content.strip()
            if raw_content2.startswith("```json"):
                raw_content2 = raw_content2[7:-3].strip()
            elif raw_content2.startswith("```"):
                raw_content2 = raw_content2[3:-3].strip()
            params_2 = json.loads(raw_content2)
        except Exception as e:
            print("Fallback for params_2 due to error:", e)
            params_2 = params_1

        # 4. Call DeepPersona with 400 attributes
        result_400 = dp_client.predict(
            age=params_2.get("Age", 30),
            gender=params_2.get("Gender", "Female"),
            occupation=params_2.get("Occupation", "Professional"),
            city=params_2.get("City", "New York"),
            country=params_2.get("Country", "USA"),
            custom_values=params_2.get("Personal Values", "Hardworking"),
            custom_life_attitude=params_2.get("Life Attitude", "Positive"),
            life_story=params_2.get("Life Story", "Grew up in the city"),
            interests_hobbies=params_2.get("Interests and Hobbies", "Reading"),
            attribute_count=350,
            api_name="/generate_persona"
        )

        # 5. Extract final structured data for _persona output
        prompt_3 = f"""
Based on this final generated persona output:
{result_400}

Extract the persona details. Return ONLY a valid JSON object with these EXACT keys (do not wrap in markdown blocks):
{{"name": "John Doe", "age": 30, "nationality": "American", "country_of_residence": "USA", "occupation": "Teacher"}}
"""
        try:
            response_3 = client.chat.completions.create(
                model="alias-fast",
                messages=[{"role": "user", "content": prompt_3}],
                temperature=0.7
            )
            raw_content3 = response_3.choices[0].message.content.strip()
            if raw_content3.startswith("```json"):
                raw_content3 = raw_content3[7:-3].strip()
            elif raw_content3.startswith("```"):
                raw_content3 = raw_content3[3:-3].strip()
            final_persona = json.loads(raw_content3)
        except Exception as e:
            print("Fallback for final_persona due to error:", e)
            final_persona = {}

        # Transform output into a tinytroupe persona profile
        tp = TinyPerson(name=final_persona.get("name", f"Persona {i+1}"))
        tp._persona["age"] = final_persona.get("age", 30)
        tp._persona["nationality"] = final_persona.get("nationality", "Unknown")
        tp._persona["country_of_residence"] = final_persona.get("country_of_residence", "Unknown")
        tp._persona["residence"] = final_persona.get("country_of_residence", "Unknown")
        tp._persona["occupation"] = final_persona.get("occupation", "Professional")
        tp._persona["full_profile_text"] = result_400

        personas.append(tp._persona)

    return personas


def start_simulation(name, content_text, format_type, persona_count, network_type):
    config = SimulationConfig(name=name, persona_count=int(persona_count), network_type=network_type)
    sim = simulation_manager.create_simulation(config)
    content = Content(text=content_text, format=format_type)
    simulation_manager.run_simulation(sim.id, content)
    
    nodes = [{"id": p.name, "label": p.name, "title": f"<b>{p.name}</b><br>{p.minibio()}", "full_bio": json.dumps(p._persona, indent=2)} for p in sim.personas]
    edges = [{"from": e.connection_id.split('_')[0], "to": e.connection_id.split('_')[1]} for e in sim.network.edges]
    analysis_df = pd.DataFrame(sim.analysis_results)
    if analysis_df.empty: analysis_df = pd.DataFrame(columns=["persona_name", "opinion", "analysis", "implications"])
    
    return analysis_df, nodes, edges, sim.id

def get_persona_details(sim_id, persona_name):
    persona = simulation_manager.get_persona(sim_id, persona_name)
    return json.dumps(persona, indent=2) if persona else "Not found"

# API functions for backward compatibility
def generate_social_network_api(name, persona_count, network_type, focus_group_name=None):
    config = SimulationConfig(name=name, persona_count=int(persona_count), network_type=network_type)
    sim = simulation_manager.create_simulation(config, focus_group_name)
    return {"simulation_id": sim.id, "persona_count": len(sim.personas)}

def predict_engagement_api(simulation_id, content_text, format_type):
    sim = simulation_manager.get_simulation(simulation_id)
    if not sim: return {"error": "Simulation not found"}
    content = Content(text=content_text, format=format_type)
    results = []
    for p in sim.personas:
        reaction = p.predict_reaction(content)
        results.append({"persona": p.name, "will_engage": reaction.will_engage, "probability": reaction.probability})
    return results

def start_simulation_async_api(simulation_id, content_text, format_type):
    content = Content(text=content_text, format=format_type)
    simulation_manager.run_simulation(simulation_id, content, background=True)
    return {"status": "started", "simulation_id": simulation_id}

def get_simulation_status_api(simulation_id):
    sim = simulation_manager.get_simulation(simulation_id)
    if not sim: return {"error": "Simulation not found"}
    return {"status": sim.status, "progress": sim.progress}

def send_chat_message_api(simulation_id, sender, message):
    return simulation_manager.send_chat_message(simulation_id, sender, message)

def get_chat_history_api(simulation_id):
    return simulation_manager.get_chat_history(simulation_id)

def generate_variants_api(original_content, num_variants):
    variants = simulation_manager.variant_generator.generate_variants(original_content, int(num_variants))
    return [v.text for v in variants]

def list_simulations_api():
    return simulation_manager.list_simulations()

def list_personas_api(simulation_id):
    return simulation_manager.list_personas(simulation_id)

def get_persona_api(simulation_id, persona_name):
    return simulation_manager.get_persona(simulation_id, persona_name)

def delete_simulation_api(simulation_id):
    success = simulation_manager.delete_simulation(simulation_id)
    return {"success": success}

def export_simulation_api(simulation_id):
    return simulation_manager.export_simulation(simulation_id)

def get_network_graph_api(simulation_id):
    sim = simulation_manager.get_simulation(simulation_id)
    if not sim: return {"error": "Simulation not found"}
    nodes = [{"id": p.name, "label": p.name, "role": p._persona.get("occupation")} for p in sim.personas]
    edges = [{"source": e.connection_id.split('_')[0], "target": e.connection_id.split('_')[1]} for e in sim.network.edges]
    return {"nodes": nodes, "edges": edges}

def list_focus_groups_api():
    return simulation_manager.list_focus_groups()

def save_focus_group_api(name, simulation_id):
    sim = simulation_manager.get_simulation(simulation_id)
    if not sim: return {"error": "Simulation not found"}
    simulation_manager.save_focus_group(name, sim.personas)
    return {"status": "success", "name": name}

# UI Layout
with gr.Blocks(css=".big-input textarea { height: 300px !important; } #mesh-network-container { height: 600px; background: #101622; border-radius: 12px; }", title="Tiny Factory") as demo:
    gr.HTML('<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>')
    gr.Markdown("# 🌐 Tiny Factory: Social Simulation Dashboard")
    
    current_sim_id = gr.State()
    
    with gr.Tabs():
        with gr.Tab("Simulation Dashboard"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Content Input")
                    sim_name = gr.Textbox(label="Simulation Name", value="Market Pulse")
                    content_input = gr.Textbox(label="Content (Blog, LinkedIn, etc.)", lines=10, elem_classes="big-input")
                    content_format = gr.Dropdown(choices=["Blog Post", "LinkedIn Update", "Tweet", "Email"], label="Format", value="LinkedIn Update")
                    num_personas_sim = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="Number of Personas")
                    network_type_sim = gr.Dropdown(choices=["scale_free", "small_world"], label="Network Topology", value="scale_free")
                    run_btn = gr.Button("🚀 Run Simulation", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("### 🕸️ Persona Mesh Network (Hover for Bio, Click for Details)")
                    gr.HTML('<div id="mesh-network-container"></div>')
                    with gr.Accordion("Detailed Persona Profile", open=False):
                        detail_name = gr.Textbox(label="Name", interactive=False)
                        detail_json = gr.Code(label="Profile JSON", language="json")
            gr.Markdown("### 📊 Simulation Analysis & Implications (Helmholtz alias-huge)")
            analysis_table = gr.Dataframe(headers=["persona_name", "opinion", "analysis", "implications"], label="Analysis Results")

        with gr.Tab("Persona Generator"):
             with gr.Row():
                with gr.Column():
                    biz_desc = gr.Textbox(label="Business Description", lines=5)
                    cust_prof = gr.Textbox(label="Customer Profile", lines=5)
                    gen_count = gr.Number(label="Count", value=5)
                    blablador_key = gr.Textbox(label="API Key (Optional)", type="password")
                    gen_btn = gr.Button("Generate Personas")
                with gr.Column():
                    gen_out = gr.JSON(label="Generated Personas")

    nodes_state = gr.State([])
    edges_state = gr.State([])
    
    # Hidden components for JS interaction
    js_trigger = gr.Textbox(visible=False, elem_id="js_trigger_textbox")
    js_trigger_btn = gr.Button("trigger", visible=False, elem_id="js_trigger_btn")

    run_btn.click(
        fn=start_simulation,
        inputs=[sim_name, content_input, content_format, num_personas_sim, network_type_sim],
        outputs=[analysis_table, nodes_state, edges_state, current_sim_id]
    ).then(
        fn=None, inputs=[nodes_state, edges_state], outputs=None,
        js="""(nodes, edges) => {
            const container = document.getElementById('mesh-network-container');
            const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
            const options = { 
                nodes: { shape: 'dot', size: 25, font: { color: '#fff', size: 16 }, color: { background: '#135bec', border: '#fff' }, shadow: true }, 
                edges: { color: 'rgba(19,91,236,0.4)', width: 2, smooth: { type: 'continuous' } }, 
                physics: { enabled: true, stabilization: false, barnesHut: { gravitationalConstant: -3000 } } 
            };
            const network = new vis.Network(container, data, options);
            network.on("click", (params) => {
                if(params.nodes.length) {
                    const node = nodes.find(n => n.id === params.nodes[0]);
                    const trigger = document.getElementById('js_trigger_textbox').querySelector('input');
                    trigger.value = node.id;
                    trigger.dispatchEvent(new Event('input'));
                    document.getElementById('js_trigger_btn').click();
                }
            });
            setInterval(() => { network.stopSimulation(); network.startSimulation(); }, 4000);
        }"""
    )

    def on_persona_click(name, sim_id):
        details = simulation_manager.get_persona(sim_id, name)
        return name, json.dumps(details, indent=2)

    js_trigger_btn.click(on_persona_click, inputs=[js_trigger, current_sim_id], outputs=[detail_name, detail_json])

    gen_btn.click(generate_personas, inputs=[biz_desc, cust_prof, gen_count, blablador_key], outputs=gen_out, api_name="generate_personas")

    # API endpoints (backward compatibility)
    with gr.Tab("API", visible=False):
        gr.Button("find_best_persona").click(lambda x: {"message": "Searching: "+x}, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="find_best_persona")
        gr.Button("generate_social_network").click(generate_social_network_api, inputs=[gr.Textbox(), gr.Number(), gr.Dropdown(choices=["scale_free", "small_world"]), gr.Textbox()], outputs=gr.JSON(), api_name="generate_social_network")
        gr.Button("predict_engagement").click(predict_engagement_api, inputs=[gr.Textbox(), gr.Textbox(), gr.Textbox()], outputs=gr.JSON(), api_name="predict_engagement")
        gr.Button("start_simulation_async").click(start_simulation_async_api, inputs=[gr.Textbox(), gr.Textbox(), gr.Textbox()], outputs=gr.JSON(), api_name="start_simulation_async")
        gr.Button("get_simulation_status").click(get_simulation_status_api, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="get_simulation_status")
        gr.Button("send_chat_message").click(send_chat_message_api, inputs=[gr.Textbox(), gr.Textbox(), gr.Textbox()], outputs=gr.JSON(), api_name="send_chat_message")
        gr.Button("get_chat_history").click(get_chat_history_api, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="get_chat_history")
        gr.Button("generate_variants").click(generate_variants_api, inputs=[gr.Textbox(), gr.Number()], outputs=gr.JSON(), api_name="generate_variants")
        gr.Button("list_simulations").click(list_simulations_api, outputs=gr.JSON(), api_name="list_simulations")
        gr.Button("list_personas").click(list_personas_api, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="list_personas")
        gr.Button("get_persona").click(get_persona_api, inputs=[gr.Textbox(), gr.Textbox()], outputs=gr.JSON(), api_name="get_persona")
        gr.Button("delete_simulation").click(delete_simulation_api, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="delete_simulation")
        gr.Button("export_simulation").click(export_simulation_api, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="export_simulation")
        gr.Button("get_network_graph").click(get_network_graph_api, inputs=[gr.Textbox()], outputs=gr.JSON(), api_name="get_network_graph")
        gr.Button("list_focus_groups").click(list_focus_groups_api, outputs=gr.JSON(), api_name="list_focus_groups")
        gr.Button("save_focus_group").click(save_focus_group_api, inputs=[gr.Textbox(), gr.Textbox()], outputs=gr.JSON(), api_name="save_focus_group")

if __name__ == "__main__":
    demo.launch(show_error=True)

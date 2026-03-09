import re

with open('app.py', 'r') as f:
    content = f.read()

new_func = '''
def generate_personas(business_description, customer_profile, num_personas, api_key=None):
    if api_key:
        os.environ["BLABLADOR_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    import json
    from gradio_client import Client
    import openai

    client = openai.OpenAI()
    dp_client = Client("THzva/deeppersona-experience")

    personas = []

    for _ in range(int(num_personas)):
        # 1. Generate initial parameters for the 200 API call
        prompt_1 = f"""
Given the following business description and customer profile:
Business: {business_description}
Customer: {customer_profile}

Generate realistic parameters for a persona that fits this profile. Return ONLY a valid JSON object with these EXACT keys:
"Age" (number), "Gender" (string), "Occupation" (string), "City" (string), "Country" (string), "Personal Values" (string), "Life Attitude" (string), "Life Story" (string), "Interests and Hobbies" (string).
Keep the string fields concise (1-2 sentences).
"""
        response_1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_1}],
            response_format={"type": "json_object"}
        )
        params_1 = json.loads(response_1.choices[0].message.content)

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

Extract and enhance specific details to create an updated set of parameters. Return ONLY a valid JSON object with these EXACT keys:
"Age" (number), "Gender" (string), "Occupation" (string), "City" (string), "Country" (string), "Personal Values" (string), "Life Attitude" (string), "Life Story" (string), "Interests and Hobbies" (string).
"""
        response_2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_2}],
            response_format={"type": "json_object"}
        )
        params_2 = json.loads(response_2.choices[0].message.content)

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
            attribute_count=400,
            api_name="/generate_persona"
        )

        # 5. Extract final structured data for _persona output
        prompt_3 = f"""
Based on this final generated persona output:
{result_400}

Extract the persona details into a structured format. Return ONLY a valid JSON object with these EXACT keys:
"name" (string, make one up if not found), "age" (number), "nationality" (string), "country_of_residence" (string), "occupation" (string), "residence" (string).
"""
        response_3 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_3}],
            response_format={"type": "json_object"}
        )
        final_persona = json.loads(response_3.choices[0].message.content)
        final_persona["full_profile_text"] = result_400
        personas.append(final_persona)

    return personas
'''

# Find the def generate_personas function block
pattern = r"def generate_personas\(business_description, customer_profile, num_personas, api_key=None\):.*?(?=\ndef start_simulation)"
new_content = re.sub(pattern, new_func.strip(), content, flags=re.DOTALL)

with open('app.py', 'w') as f:
    f.write(new_content)

print("Patch applied")

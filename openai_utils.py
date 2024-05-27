# # import openai

# # openai.api_key = ""

# # def generate_response(prompt):
# #     response = openai.Completion.create(
# #         engine="text-davinci-003",
# #         prompt=prompt,
# #         max_tokens=4000,
# #         n=1,
# #         stop=None,
# #         temperature=0.5,
# #     )
# #     return response["choices"][0]["text"]




# # import openai

# # openai.api_key = ''

# # def generate_response(prompt):
# #     response = openai.ChatCompletion.create(
# #       model="gpt-3.5-turbo",
# #       messages=[
# #             {"role": "system", "content": "You are a helpful assistant."},
# #             {"role": "user", "content": prompt}
# #         ]
# #     )
# #     return response['choices'][0]['message']['content']



# import openai
# import os

# # Get the API key from the environment variable
# openai.api_key = os.getenv("")
# url = "https://api.openai.com/v1/threads/"+thread_id +"/runs"
# headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "appalication/json",
#         "OpenAI-Beta": "assistants=v2"
#     }
# data = {
#         "assistant_id": assistant_id
#     }
# response = requests.post(url, headers=headers, data=json.dumps(data))
# response= response.json()



# def generate_response(user_input):
#     response = openai.Completion.create(
#         engine="text-davinci-003",  # or whichever engine you are using
#         prompt=user_input,
#         max_tokens=150
#     )
#     return response.choices[0].text.strip()

a





































import requests
import openai

openai.api_key = ""
import os
import openai

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("SECRET_KEY")
import os

# Print all environment variables
for key, value in os.environ.items():
    print(f"{key}={value}")


def generate_response(prompt, thread_id):
    url = "https://api.openai.com/v1/threads/" + thread_id + "/runs"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


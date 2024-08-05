import openai

AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = "api71-api-834e5686-3caa-43c3-a013-77e924eac753"

client = openai.OpenAI(
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
)

# Simple invocation
try:
    response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat",
        messages=[
            {"role": "system", "content": "Talk like the Rock."},
            {"role": "user", "content": "Who are ya!"},
        ],
        max_tokens=100,
        temperature=0.7,
    )
    print(response.choices[0].message.content.strip())
except Exception as e:
    print(f"An error occurred: {str(e)}")
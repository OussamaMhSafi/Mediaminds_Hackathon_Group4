import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message["content"])

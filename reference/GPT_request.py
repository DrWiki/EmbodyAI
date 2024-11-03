from openai import OpenAI
client = OpenAI(api_key="")

response = client.chat.completions.create(
    messages=[{
        "role": "user",
        # "content": "Say this is a test, can you responce in Chinese?",
        "content": "请介绍一下日本历史",
    }],
    model="gpt-4o-mini",
)

print(response._request_id)
print(response.choices[0].message.content)

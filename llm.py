from openai import OpenAI

client = OpenAI(api_key="your-api-key-here")

def generate_insights_llm(summary):
    prompt = f"Analyze this dataset summary: {summary}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
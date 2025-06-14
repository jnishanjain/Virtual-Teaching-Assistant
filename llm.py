from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, docs):
    context = "\n\n".join([doc["text"] for doc in docs])
    links = [{"url": doc["url"], "text": doc["text"][:50]} for doc in docs]

    messages = [
        {"role": "system", "content": "You are a helpful TA for the TDS course."},
        {"role": "user", "content": f"Question: {question}\n\nContext: {context}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip(), links

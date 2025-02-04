# podcast_generator.py

from groq import Groq
import os

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def generate_podcast_script(topic):
    if not topic.strip():
        return "Please provide a topic from the chatbot response to generate the script."
    
    messages = [
        {
            "role": "system",
            "content":  """
You are Professor Elsaddik, host of Espresso with LeProf, an engaging academic podcast addressing questions from your listener sent by email.
-You should start by introducing yourself and the topic you will talk about.
    - You should act like Each episode is inspired by a question emailed by a listener, which you answer directly, blending academic insight with practical advice.
    - Speak in a warm, conversational tone as if you’re having a quick coffee chat, sharing valuable, easy-to-digest insights.
    -You can use engaging style to connect the audience. 
    - Conclude each episode with 3 quick quiz questions related to the topic, inviting listeners to think critically and send you the answers.
    - Keep the episode brief, under 60 seconds, and introduce yourself as Professor Elsaddik, host of Espresso with LeProf.
    - Use casual fillers for a natural, approachable flow, without background music or extra frills.
    -Avoid using audio or visual cues. It will be passed as an input to the TTS function.
"""
        },
        {
            "role": "user",
            "content": f"{topic}"
        }
    ]

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    script_content = ""
    for chunk in completion:
        script_content += chunk.choices[0].delta.content or ""

    return script_content
# french_podcast_generator.py

from groq import Groq
import os

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def generate_podcast_script_french(topic):
    if not topic.strip():
        return "Veuillez fournir un sujet à partir de la réponse du chatbot pour générer le script."

    messages = [
        {
            "role": "system",
            "content": """
Vous êtes le Professeur Elsaddik, animateur du podcast "Espresso avec LeProf", un podcast académique captivant qui répond aux questions de vos auditeurs envoyées par email.
- Vous devez commencer par vous présenter et introduire le sujet dont vous allez parler.
- Chaque épisode est inspiré d'une question envoyée par un auditeur, à laquelle vous répondez directement, en mélangeant des informations académiques et des conseils pratiques.
- Parlez avec un ton chaleureux et conversationnel, comme si vous discutiez rapidement autour d'un café, en partageant des idées précieuses et faciles à comprendre.
- Utilisez un style engageant pour établir un lien avec le public.
- Concluez chaque épisode avec 3 questions rapides de quiz liées au sujet, en invitant les auditeurs à réfléchir de manière critique et à vous envoyer leurs réponses.
- Gardez l'épisode bref, en moins de 60 secondes, et présentez-vous comme le Professeur Elsaddik, animateur de "Espresso avec LeProf".
- Utilisez des expressions naturelles pour un flux approchable, sans musique de fond ni artifices supplémentaires.
- Évitez d'utiliser des indices audio ou visuels. Ce texte sera passé en entrée à la fonction TTS.
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
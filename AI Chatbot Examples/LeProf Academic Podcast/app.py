import gradio as gr
from summarizer import summarize_topic
from podcast_generator import generate_podcast_script
from french_podcast import generate_podcast_script_french
from audio_generator import gtpodcast_script_to_audio
#from multpdf import upload_files, build_vector_db, respond
import os
from groq import Groq

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize conversation history
conversation_history = []

def chat_with_bot_stream(user_input):
    global conversation_history
    # Append the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Add a system message if the history is empty
    if len(conversation_history) == 1:
        conversation_history.insert(0, {
            "role": "system",
            "content": "You are an expert of the given topic. Analyze the provided text with a focus on the topic, identifying recent issues, recent insights, or improvements relevant to academic standards and effectiveness. Offer actionable advice for enhancing knowledge and suggest real-life examples."
        })

    # Get response from chatbot with streaming
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=conversation_history,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    response_content = ""
    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""
    
    # Append the bot's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response_content})
    
    # Return the updated conversation history
    return [(msg["content"] if msg["role"] == "user" else None, 
             msg["content"] if msg["role"] == "assistant" else None) 
            for msg in conversation_history]

#Use the podcast generation for user input onl

def generate_and_play_podcast(chat_history):
    # Extract only user queries from the chat history
    user_queries = [msg[0] for msg in chat_history if msg[0]]
    # Combine user queries into a single text
    conversation_text = "\n".join(user_queries)
    
    # Generate podcast script
    podcast_script = generate_podcast_script(conversation_text)
    # Convert the script to audio
    audio_path = gtpodcast_script_to_audio(podcast_script)
    # Return both the script and the audio file path
    return podcast_script, audio_path

def generate_and_play_podcast_french(chat_history):
    # Extract only user queries from the chat history
    user_queries = [msg[0] for msg in chat_history if msg[0]]
    # Combine user queries into a single text
    conversation_text = "\n".join(user_queries)
    
    # Generate podcast script
    podcast_script = generate_podcast_script_french(conversation_text)
    # Convert the script to audio
    audio_path = gtpodcast_script_to_audio(podcast_script)
    # Return both the script and the audio file path
    return podcast_script, audio_path

    #Use the podcast generation for the whole conversation

#def generate_and_play_podcast(chat_history):
    # Convert chat history into a readable string
    #conversation_text = "\n".join(
      #  f"User: {msg[0]}\nAssistant: {msg[1]}" 
        #for msg in chat_history if msg[0] or msg[1]
  #  )
    # Generate podcast script
    #podcast_script = generate_podcast_script(conversation_text)
    # Convert the script to audio
    #audio_path = gtpodcast_script_to_audio(podcast_script)
    # Return both the script and the audio file path
    #return podcast_script, audio_path

TITLE = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>☕️ Espresso with LeProf Lite</h1>
"""

TITLE_Podcast = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>🫗 I can brew podcast based on the previous chat</h1>
"""

TITLE_Custome_Podcast = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>🫗 I can brew podcast based on your given topic</h1>
"""


TITLE_Chat= """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>LitPie 📖🍕</h1>
"""


with gr.Blocks(theme=gr.themes.Glass(primary_hue="violet", secondary_hue="emerald", neutral_hue="stone")) as demo:
    with gr.Tabs():
        with gr.TabItem("💬Chat"):
            gr.HTML(TITLE)
            chatbot = gr.Chatbot(label="LeProf Chatbot")
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your question here...",
                    lines=1
                )
                send_button = gr.Button("✋Ask Question")

            # Chatbot functionality: Update chatbot and clear text input
            send_button.click(
                fn=chat_with_bot_stream,  # This should be defined in your actual application
                inputs=user_input,
                outputs=chatbot,
                queue=True  # Enables streaming responses
            ).then(
                fn=lambda: "",  # Clear the input box after sending
                inputs=None,
                outputs=user_input
            )

        with gr.TabItem("🎙️Podcast on Chat"):
            gr.HTML(TITLE_Podcast)
            podcast_button = gr.Button("🎧 Generate Podcast")
            french_podcast_button = gr.Button("🎧 Generate French Podcast")
            podcast_script_output = gr.Textbox(label="Podcast Transcript", placeholder="Podcast script will appear here.", lines=5)
            podcast_audio_output = gr.Audio(label="Podcast Audio")
            
            # Generate podcast script and audio
            podcast_button.click(
                fn=generate_and_play_podcast,  # This should be defined in your actual application
                inputs= chatbot,  # Pass the chat history
                outputs=[podcast_script_output, podcast_audio_output]
            )
            french_podcast_button.click(
                fn=generate_and_play_podcast_french,  # This should be defined in your actual application
                inputs=chatbot,  # Pass the chat history
                outputs=[podcast_script_output, podcast_audio_output]
            )

        with gr.TabItem("🎙️🙏Custom Podcast"):
            gr.HTML(TITLE_Custome_Podcast)
            podcast_topic_input = gr.Textbox(label="Custom Podcast Topic", placeholder="Enter your custom topic here.")
            chatbot_input = chatbot  # Assuming `chatbot` is defined elsewhere in your application
            podcast_button = gr.Button("🎧 Generate Podcast")
            french_podcast_button = gr.Button("🎧 Generate French Podcast")
            podcast_script_output = gr.Textbox(label="Podcast Transcript", placeholder="Podcast script will appear here.", lines=5)
            podcast_audio_output = gr.Audio(label="Podcast Audio")
            
            # Generate podcast script and audio
            podcast_button.click(
                fn=generate_and_play_podcast,  # This should be defined in your actual application
                inputs= podcast_topic_input,  # Include both chatbot input and custom topic
                outputs=[podcast_script_output, podcast_audio_output]
            )
            french_podcast_button.click(
                fn=generate_and_play_podcast_french,  # This should be defined in your actual application
                inputs=podcast_topic_input,  # Pass the chat history
                outputs=[podcast_script_output, podcast_audio_output]
            )
        
        # Tab for Lit Pie 🍕
        with gr.TabItem("Others"):
            gr.Markdown("### This tab is reserved for future functionalities.")

demo.launch()


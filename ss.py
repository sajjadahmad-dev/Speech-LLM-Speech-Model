# ✅ Import Libraries
import whisper
import os
from groq import Groq
import gradio as gr
from TTS.api import TTS

# ✅ Load Whisper model for Speech-to-Text
stt_model = whisper.load_model("small")

# ✅ Set Groq API Key
GROQ_API_KEY = "YOUR_GROQ_API_KEY"  # 🔴 Replace this with your actual API key
client = Groq(api_key=GROQ_API_KEY)

# ✅ Load TTS Model for AI Speech
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# ✅ Store Chat History
chat_history = []

# ✅ Function to Get AI Response from Groq
def get_groq_response(prompt):
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"

# ✅ Function to Process Voice Input
def chat_with_ai(audio_file):
    try:
        # Convert speech to text
        result = stt_model.transcribe(audio_file)
        user_text = result["text"]
        
        # Get AI response
        ai_response = get_groq_response(user_text)

        # Convert AI response to voice
        output_audio = "response.wav"
        tts.tts_to_file(text=ai_response, file_path=output_audio)

        # Store in chat history
        chat_history.append(("🗣️ You: " + user_text, "🤖 AI: " + ai_response))
        
        return output_audio, ai_response, "\n\n".join(["\n".join(pair) for pair in chat_history])
    
    except Exception as e:
        return None, f"⚠️ Error: {e}", ""

# ✅ Function to Handle Text Input
def chat_with_text(user_text):
    try:
        # Get AI response
        ai_response = get_groq_response(user_text)

        # Convert AI response to voice
        output_audio = "response.wav"
        tts.tts_to_file(text=ai_response, file_path=output_audio)

        # Store in chat history
        chat_history.append(("🗣️ You: " + user_text, "🤖 AI: " + ai_response))
        
        return output_audio, ai_response, "\n\n".join(["\n".join(pair) for pair in chat_history])
    
    except Exception as e:
        return None, f"⚠️ Error: {e}", ""

# ✅ Function to Clear Chat
def clear_chat():
    global chat_history
    chat_history = []
    return None, "", ""

# ✅ Build Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🎙️ **Voice & Text AI Chatbot** 🤖")
    gr.Markdown("Speak or type your question, and get an AI response in **voice & text**!")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak Here")
        text_input = gr.Textbox(label="💬 Or Type Your Question")
    
    with gr.Row():
        submit_audio = gr.Button("🎙️ Send Voice")
        submit_text = gr.Button("💬 Send Text")
        clear_btn = gr.Button("🗑️ Clear Chat")

    with gr.Row():
        audio_output = gr.Audio(label="🔊 AI Voice Response")
        text_output = gr.Textbox(label="🤖 AI Text Response")

    chat_history_box = gr.Textbox(label="📜 Chat History", interactive=False, lines=10)

    submit_audio.click(chat_with_ai, inputs=[audio_input], outputs=[audio_output, text_output, chat_history_box])
    submit_text.click(chat_with_text, inputs=[text_input], outputs=[audio_output, text_output, chat_history_box])
    clear_btn.click(clear_chat, outputs=[audio_output, text_output, chat_history_box])

app.launch(share=True)

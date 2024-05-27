# # # import streamlit as st
# # # import speech_recognition as sr
# # # import nltk
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # import socketio
# # # import pyaudio
# # # import wave

# # # # Set up Google Cloud Speech-to-Text API
# # # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/google-credentials.json'
# # # speech_client = speech.SpeechClient()

# # # # Set up NLP library (e.g., NLTK, spaCy)
# # # nlp = nltk.NLTK()

# # # # Define a sequence-to-sequence model (RNN)
# # # class RNNModel(nn.Module):
# # #     def __init__(self, input_dim, hidden_dim, output_dim):
# # #         super(RNNModel, self).__init__()
# # #         self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
# # #         self.fc = nn.Linear(hidden_dim, output_dim)

# # #     def forward(self, x):
# # #         h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
# # #         out, _ = self.rnn(x, h0)
# # #         out = self.fc(out[:, -1, :])
# # #         return out

# # # model = RNNModel(input_dim=128, hidden_dim=256, output_dim=128)

# # # # Set up Socket.IO
# # # sio = socketio.Server()
# # # app = socketio.WSGIApp(sio)

# # # # Set up PyAudio for audio playback
# # # p = pyaudio.PyAudio()
# # # stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)

# # # # Define a function to handle microphone input
# # # def handle_mic_input(sid, message):
# # #     try:
# # #         with sr.Microphone() as source:
# # #             audio = r.record(source, duration=5)
# # #             audio_content = audio.get_wav_data()
# # #             transcript = speech_client.recognize(config=speech.RecognitionConfig(
# # #                 encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
# # #                 sample_rate_hertz=16000,
# # #                 language_code='en-US',
# # #             ), audio=speech.RecognitionAudio(content=audio_content)).results[0].alternatives[0].transcript

# # #             # Preprocess transcript
# # #             transcript = transcript.lower()
# # #             transcript = re.sub(r'[^\w\s]', '', transcript)

# # #             # Tokenize transcript
# # #             tokens = nlp.word_tokenize(transcript)

# # #             # Convert tokens to numerical input
# # #             input_tensor = torch.tensor([nlp.word_to_index[token] for token in tokens])

# # #             # Generate response using RNN model
# # #             output_tensor = model(input_tensor)
# # #             response = ''.join([nlp.index_to_word[i] for i in output_tensor])

# # #             # Synthesize response using Google Text-to-Speech API
# # #             tts_client = texttospeech.TextToSpeechClient()
# # #             synthesis_input = texttospeech.SynthesisInput(text=response)
# # #             voice = texttospeech.VoiceSelectionParams(
# # #                 language_code='en-US',
# # #                 name='en-US-Wavenet-A'
# # #             )
# # #             config = texttospeech.AudioConfig(
# # #                 audio_encoding=texttospeech.AudioEncoding.LINEAR16
# # #             )
# # #             response_audio = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=config)

# # #             # Play back response audio
# # #             stream.write(response_audio.audio_content)

# # #     except Exception as e:
# # #         st.error(f"Error: {e}")

# # # # Define a function to run the Streamlit app
# # # def run_app():
# # #     st.title("Voice Assistant")
# # #     st.button("Start listening", on_click=lambda: sio.emit('start_listening', 'client'))

# # # # Register the microphone input handler function
# # # sio.on('start_listening', handle_mic_input)

# # # # Run the Streamlit app
# # # run_app()

























import streamlit as st
from speech_utils import speech_to_text, text_to_speech
from openai_utils import generate_response

def main():
    st.title("Talker.AI")
    st.write("Click the button to ask your question.")

    if st.button("Start talking"):
        user_input = speech_to_text()
        if user_input:
            st.write(f"You said: {user_input}")
            response = generate_response(user_input)
            st.write(f"Talker.AI says: {response}")
            text_to_speech(response)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
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





































import streamlit as st
from openai_utils import generate_response
from speech_utils import transcribe_audio_to_text, speak_text
from speech_utils import transcribe_audio_to_text, speak_text

def main():
    st.title("Talker AI - Real-time AI Conversation")
    
    if st.button("🎤 Start Talking"):
        with st.spinner("Listening..."):
            text = transcribe_audio_to_text()
            st.write(f"You said: {text}")
            
            thread_id = "3384"  # Replace with the actual thread ID or logic to obtain it
            response = generate_response(text, thread_id)
            
            st.write(f"AI says: {response['choices'][0]['text']}")
            
            speak_text(response['choices'][0]['text'])

if __name__ == "__main__":
    main()



# import threading

# def print_thread_id():
#     print(f"Current thread ID is: {threading.get_ident()}")

# print_thread_id()
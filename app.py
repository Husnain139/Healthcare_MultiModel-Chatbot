#VoiceBot UI with Streamlit
import os
import streamlit as st
from pathlib import Path
import time

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

# Page configuration
st.set_page_config(
    page_title="AI Doctor with Vision and Voice",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Main body background and font */
    .main {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Elegant Title */
    .header-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 2.8rem !important;
        text-align: center;
        background: linear-gradient(135deg, #f43f5e 0%, #fb7185 50%, #fda4af 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: -0.03em;
        filter: drop-shadow(0 2px 10px rgba(244, 63, 94, 0.2));
        padding-top: 20px;
    }
    
    .header-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        text-align: center;
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 35px;
        font-weight: 400;
    }
    
    /* Glassmorphism card effects for columns */
    div[data-testid="column"] {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    
    div[data-testid="column"]:hover {
        border-color: rgba(244, 63, 94, 0.2) !important;
    }
    
    /* Button Customization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #e11d48 0%, #be123c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px 0 rgba(225, 29, 72, 0.3) !important;
        height: auto !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(225, 29, 72, 0.5), 0 0 12px rgba(225, 29, 72, 0.3) !important;
        background: linear-gradient(135deg, #f43f5e 0%, #be123c 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
    }
    
    /* Image and Uploader block styling */
    div[data-testid="stImage"] {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        background: rgba(15, 23, 42, 0.5) !important;
    }
    
    /* Audio input / recorder style */
    div[data-testid="stAudioInput"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 8px !important;
        background: rgba(15, 23, 42, 0.3) !important;
    }
    
    /* Output text areas / boxes styling */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 15px !important;
    }
    
    /* Custom headers in cards */
    .card-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 20px;
        border-bottom: 2px solid rgba(244, 63, 94, 0.2);
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Footer elements */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 30px;
    }
    
    /* Success & Info boxes style */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recorded_audio_path' not in st.session_state:
    st.session_state.recorded_audio_path = None
if 'recording_done' not in st.session_state:
    st.session_state.recording_done = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'doctor_response' not in st.session_state:
    st.session_state.doctor_response = ""
if 'response_audio_path' not in st.session_state:
    st.session_state.response_audio_path = None

# Header
st.markdown("<div class='header-title'>🩺 AI Doctor with Vision and Voice</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Empowered with Groq Llama-4 Multimodal Intelligence</div>", unsafe_allow_html=True)

# Main layout: Left column and Right column
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Voice Recording Panel
    st.markdown("<div class='card-header'>🎤 Voice Recording</div>", unsafe_allow_html=True)
    
    audio_value = st.audio_input("Record your voice", label_visibility="collapsed")
    
    if audio_value:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        # st.audio_input returns WAV audio data, which Groq Whisper API handles perfectly.
        audio_path = temp_dir / "recorded_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_value.getbuffer())
        st.session_state.recorded_audio_path = str(audio_path)
        st.session_state.recording_done = True
    else:
        st.session_state.recorded_audio_path = None
        st.session_state.recording_done = False
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Image Upload Panel
    st.markdown("<div class='card-header'>📷 Medical Image</div>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload a medical image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Medical Image", use_container_width=True)
    else:
        st.info("👆 Upload a medical image (JPG, PNG)")

with col_right:
    # Speech-to-Text Result Box
    st.markdown("<div class='card-header'>📝 Speech-to-Text Result</div>", unsafe_allow_html=True)
    if st.session_state.transcription:
        st.info(st.session_state.transcription)
    else:
        st.text_area("", value="Your spoken question will appear here after analysis...", height=150, disabled=True, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Doctor's Response Box
    st.markdown("<div class='card-header'>👨‍⚕️ Doctor's Response</div>", unsafe_allow_html=True)
    if st.session_state.doctor_response:
        st.success(st.session_state.doctor_response)
    else:
        st.text_area("", value="AI Doctor's diagnosis and recommendations will appear here...", height=150, disabled=True, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Audio Output Player
    st.markdown("<div class='card-header'>🔊 Audio Response</div>", unsafe_allow_html=True)
    if st.session_state.response_audio_path and os.path.exists(st.session_state.response_audio_path):
        # Read audio file
        with open(st.session_state.response_audio_path, "rb") as audio_f:
            audio_bytes = audio_f.read()
        
        # Use HTML audio with autoplay
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.info("🔇 Audio response will be generated after analysis")

# Analyze Button - Full width at bottom
st.markdown("---")
if st.button("🔍 ANALYZE NOW", type="primary", use_container_width=True):
    if not uploaded_image:
        st.error("⚠️ Please upload a medical image first!")
    elif not st.session_state.recording_done or not st.session_state.recorded_audio_path:
        st.error("⚠️ Please record your voice first!")
    else:
        try:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Save image
            image_path = temp_dir / f"temp_image.{uploaded_image.name.split('.')[-1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Transcribe audio
            status_text.text("🎧 Transcribing your voice...")
            progress_bar.progress(33)
            transcription = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                audio_filepath=st.session_state.recorded_audio_path,
                stt_model="whisper-large-v3"
            )
            st.session_state.transcription = transcription
            
            # Step 2: Analyze image
            status_text.text("🔬 Analyzing medical image with AI...")
            progress_bar.progress(66)
            encoded_image = encode_image(str(image_path))
            doctor_response = analyze_image_with_query(
                query=system_prompt + transcription,
                encoded_image=encoded_image,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            st.session_state.doctor_response = doctor_response
            
            # Step 3: Generate voice response
            status_text.text("🔊 Generating voice response...")
            progress_bar.progress(90)
            output_audio_path = temp_dir / "doctor_response.mp3"
            text_to_speech_with_gtts(
                input_text=doctor_response,
                output_filepath=str(output_audio_path),
                play_audio=False
            )
            st.session_state.response_audio_path = str(output_audio_path)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Analysis Complete! Scroll up to see results.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

# Footer
st.markdown("<div class='footer-text'>⚕️ <i>This is an AI-powered educational tool. Always consult a real medical professional for actual medical advice.</i></div>", unsafe_allow_html=True)

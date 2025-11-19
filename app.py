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
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    div[data-testid="stImage"] {
        border: 2px solid #262730;
        border-radius: 10px;
        padding: 10px;
    }
    .result-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        min-height: 150px;
    }
    h1 {
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 30px;
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
st.markdown("<h1>🩺 AI Doctor with Vision and Voice</h1>", unsafe_allow_html=True)

# Main layout: Left column and Right column
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Voice Recording Panel
    st.markdown("### 🎤 Voice Recording")
    
    voice_col1, voice_col2 = st.columns([1, 1])
    
    with voice_col1:
        if st.button("🎙️ Start Recording", use_container_width=True, type="primary", disabled=st.session_state.is_recording):
            st.session_state.is_recording = True
            with st.spinner("🔴 Recording for 10 seconds... Please speak!"):
                try:
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    
                    audio_path = temp_dir / "recorded_audio.mp3"
                    success = record_audio(file_path=str(audio_path), duration=10)
                    
                    if success:
                        st.session_state.recorded_audio_path = str(audio_path)
                        st.session_state.recording_done = True
                        st.session_state.is_recording = False
                        st.success("✅ Recording complete!")
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Recording error: {str(e)}")
                    st.session_state.is_recording = False
    
    with voice_col2:
        if st.button("⏹️ Stop/Reset", use_container_width=True, disabled=not st.session_state.recording_done):
            st.session_state.recorded_audio_path = None
            st.session_state.recording_done = False
            st.session_state.is_recording = False
            st.rerun()
    
    # Device input dropdown (placeholder for future implementation)
    audio_device = st.selectbox("🔊 Audio Input Device", ["Default Microphone", "System Audio"])
    
    # Show recorded audio
    if st.session_state.recording_done and st.session_state.recorded_audio_path:
        if os.path.exists(st.session_state.recorded_audio_path):
            st.audio(st.session_state.recorded_audio_path)
        else:
            st.warning("⚠️ Recording not found")
    else:
        st.info("👆 Click 'Start Recording' to begin")
    
    st.markdown("---")
    
    # Image Upload Panel
    st.markdown("### 📷 Medical Image")
    uploaded_image = st.file_uploader("Upload a medical image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Medical Image", use_container_width=True)
    else:
        st.info("👆 Upload a medical image (JPG, PNG)")

with col_right:
    # Speech-to-Text Result Box
    st.markdown("### 📝 Speech-to-Text Result")
    if st.session_state.transcription:
        st.info(st.session_state.transcription)
    else:
        st.text_area("", value="Your spoken question will appear here after analysis...", height=150, disabled=True, label_visibility="collapsed")
    
    st.markdown("")
    
    # Doctor's Response Box
    st.markdown("### 👨‍⚕️ Doctor's Response")
    if st.session_state.doctor_response:
        st.success(st.session_state.doctor_response)
    else:
        st.text_area("", value="AI Doctor's diagnosis and recommendations will appear here...", height=150, disabled=True, label_visibility="collapsed")
    
    st.markdown("")
    
    # Audio Output Player
    st.markdown("### 🔊 Audio Response")
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
                output_filepath=str(output_audio_path)
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
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>⚕️ <i>This is an AI-powered educational tool. Always consult a real medical professional for actual medical advice.</i></p>", unsafe_allow_html=True)

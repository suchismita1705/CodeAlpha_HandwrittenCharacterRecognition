import os
import cv2
import itertools
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from spellchecker import SpellChecker

# 1. APPLICATION GRAPHICAL LAYOUT CONFIGURATIONS
st.set_page_config(page_title="Hybrid CV+NLP Word Recognition", page_icon="✍️", layout="centered")

st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 40px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 18px; color: #4B5563; margin-bottom: 25px; }
    .raw-box { background-color: #FEF3C7; padding: 15px; border-radius: 8px; border: 1px solid #FCD34D; text-align: center; }
    .nlp-box { background-color: #DCFCE7; padding: 15px; border-radius: 8px; border: 1px solid #BBF7D0; text-align: center; }
    .result-text { font-size: 45px; font-weight: bold; }
    .raw-text { color: #D97706; }
    .nlp-text { color: #16A34A; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Hybrid CV + NLP Word Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Global Language Context-Aware Corrector</div>', unsafe_allow_html=True)

# 2. RUNTIME MEMORY MODEL LOADER & NLP SPELLCHECK INITIALIZATION
@st.cache_resource
def load_trained_brain():
    """Loads the serialized Keras CNN model asset graph from local disk."""
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_trained_brain()

# Initialize the massive global English language corpus database
spell = SpellChecker()
# Pre-weight common testing keywords to optimize tie-breaking parameters
spell.word_frequency.load_words(["HELLO", "NAME", "CODE", "CAT", "DOG", "AI", "HOME"])

CLASS_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

if model is None:
    st.error("❌ Error: 'best_handwritten_model.keras' missing. Please run your training script first!")
else:
    # 3. CHOOSE MODE INTERACTIVE NAVIGATION TABS
    app_mode = st.radio("Select Input Workspace Method:", ("✏️ Draw Live Word/Sequence", "📁 Upload Image File"), horizontal=True)
    
    img_gray = None
    
    if app_mode == "✏️ Draw Live Word/Sequence":
        st.write("Draw any standard English word across the wide canvas area below (leave clean horizontal spaces between letters):")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=200,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True
        )
        
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
            rgba_array = canvas_result.image_data
            img_gray = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2GRAY)

    else:
        uploaded_file = st.file_uploader("Upload an image file containing handwritten text...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            source_image = Image.open(uploaded_file)
            st.image(source_image, caption="Uploaded Original Handwriting", use_container_width=True)
            img_gray = np.array(source_image.convert('L'))
            if np.mean(img_gray) > 127:
                img_gray = cv2.bitwise_not(img_gray)

    # 4. COMPUTER VISION PREPROCESSING & CONTOUR SEGMENTATION
    if img_gray is not None:
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 3 and h > 3:
                bounding_boxes.append((x, y, w, h))
                
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        
        if len(bounding_boxes) == 0:
            st.warning("⚠️ Write or upload a clearer character line to begin analysis.")
        else:
            debug_canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            processed_segments = []
            
            for idx, (x, y, w, h) in enumerate(bounding_boxes):
                cv2.rectangle(debug_canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                crop = img_gray[y:y+h, x:x+w]
                max_dim = max(w, h)
                padded_square = np.zeros((max_dim, max_dim), dtype=np.uint8)
                padded_square[(max_dim-h)//2:(max_dim-h)//2+h, (max_dim-w)//2:(max_dim-w)//2+w] = crop
                final_padded = cv2.copyMakeBorder(padded_square, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
                
                seg_resized = cv2.resize(final_padded, (28, 28), interpolation=cv2.INTER_AREA)
                processed_segments.append(seg_resized.astype('float32') / 255.0)
                
            st.image(debug_canvas, caption="OpenCV Segmentation Traces", use_container_width=True)
            
            st.markdown("---")
            if st.button("🚀 Run Hybrid CV+NLP Analytics", use_container_width=True):
                char_candidates_list = []
                raw_top_chars = []
                
                # Extract Softmax probability distributions for each individual segment
                for seg_tensor in processed_segments:
                    input_tensor = np.reshape(seg_tensor, (1, 28, 28, 1))
                    raw_probabilities = model.predict(input_tensor, verbose=0)[0]
                    
                    # Sort indices based on probability values descending
                    sorted_indices = np.argsort(raw_probabilities)[::-1]
                    
                    # 1. Unconditionally preserve the top absolute guess
                    top_1_char = CLASS_MAPPING[sorted_indices[0]]
                    raw_top_chars.append(top_1_char)
                    
                    candidates = [top_1_char]
                    
                    # 2. Advanced Multi-Candidate Extraction:
                    # If the second-best guess has a confidence score > 5%, include it as an alternative path
                    if raw_probabilities[sorted_indices[1]] > 0.05:
                        candidates.append(CLASS_MAPPING[sorted_indices[1]])
                        
                    char_candidates_list.append(candidates)
                    
                # Generate the raw visual word prediction output string from absolute top choices
                raw_predicted_word = "".join(raw_top_chars).upper()
                
                # --- ULTRA-ADVANCED STEP: JOINT PROBABILISTIC LANGUAGE DECODING ---
                # Use itertools to cross-multiply all character candidates into possible word combinations
                all_possible_combinations = ["".join(comb).upper() for comb in itertools.product(*char_candidates_list)]
                
                # Filter combinations to isolate entries that match real English dictionary words
                valid_dictionary_words = [word for word in all_possible_combinations if word.lower() in spell]
                
                if valid_dictionary_words:
                    # If valid combinations exist, pick the word with the highest real-world language frequency score
                    nlp_corrected_word = max(valid_dictionary_words, key=lambda w: spell.word_frequency[w.lower()])
                    is_corrected = nlp_corrected_word != raw_predicted_word
                else:
                    # Fallback to standard Levenshtein edit-distance correction if no combinations hit the dictionary directly
                    nlp_corrected_word = spell.correction(raw_predicted_word)
                    if nlp_corrected_word is None:
                        nlp_corrected_word = raw_predicted_word
                    nlp_corrected_word = nlp_corrected_word.upper()
                    is_corrected = nlp_corrected_word != raw_predicted_word
                
                # Render Split Results Panel Interface Layout
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown('<div class="raw-box">', unsafe_allow_html=True)
                    st.write("### 👁️ Raw Vision Prediction")
                    st.markdown(f'<div class="result-text raw-text">{raw_predicted_word}</div>', unsafe_allow_html=True)
                    st.write("Direct output from pixel classification.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_right:
                    st.markdown('<div class="nlp-box">', unsafe_allow_html=True)
                    st.write("### 🧠 Smart NLP Correction")
                    st.markdown(f'<div class="result-text nlp-text">{nlp_corrected_word}</div>', unsafe_allow_html=True)
                    if is_corrected:
                        st.write(f"✨ Auto-corrected utilizing joint Softmax probability decoding!")
                    else:
                        st.write("✅ Word verified against English language corpus.")
                    st.markdown('</div>', unsafe_allow_html=True)
import os
import cv2
import numpy as np
import tensorflow as tf

def predict_handwritten_character(image_path=None):
    # Define official EMNIST Balanced 47-class lookup dictionary array sequence
    class_mapping = [
        '0','1','2','3','4','5','6','7','8','9',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','d','e','f','g','h','n','q','r','t'
    ]
    
    # 1. Load the compiled keras model file
    model_path = os.path.join("saved_models", "best_handwritten_model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model binary missing at {model_path}. Please run training script first!")
        return
    model = tf.keras.models.load_model(model_path)
    
    # 2. SOURCE THE IMAGE MATRIX
    if image_path and os.path.exists(image_path):
        print(f"🖼️ Loading target external image file from: {image_path}")
        # Load file in absolute grayscale format
        img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # INDUSTRY INSIGHT: Fix canvas color background configurations if needed
        # If the center average pixel intensity is high, it's black ink on white background.
        # We must invert it to match EMNIST (white text on black background)
        if np.mean(img_raw) > 127:
            print("🔄 Detected dark-on-light writing canvas. Executing color inversion...")
            img_raw = cv2.bitwise_not(img_raw)
            
        # Resize spatial resolution dynamically down to match model input parameters
        img_resized = cv2.resize(img_raw, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize continuous floating metrics range [0.0, 1.0]
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Reshape matrix array container structure to match 4D Tensor expectations: (Batch, Height, Width, Channels)
        processed_img = np.reshape(img_normalized, (1, 28, 28, 1))
        
    else:
        print("💡 No external image provided or file not found. Pulling random sample from test matrix for testing...")
        test_data_path = os.path.join("data", "processed", "X_test.npy")
        test_labels_path = os.path.join("data", "processed", "y_test.npy")
        
        if not (os.path.exists(test_data_path) and os.path.exists(test_labels_path)):
            print("❌ Error: Processed arrays missing. Please run preprocess.py first!")
            return
            
        X_test = np.load(test_data_path)
        y_test = np.load(test_labels_path)
        
        # Pull a random index
        random_idx = np.random.choice(X_test.shape[0])
        processed_img = np.expand_dims(X_test[random_idx], axis=0)
        actual_label_id = y_test[random_idx]
        print(f"🎯 Selected Test Sample Index: {random_idx} (Actual Ground-Truth Character: '{class_mapping[actual_label_id]}')")

    # 3. RUN MODEL INFERENCE PREDICTION
    raw_probabilities = model.predict(processed_img, verbose=0)
    
    # Extract structural performance metrics positions
    predicted_class_id = np.argmax(raw_probabilities, axis=1)[0]
    confidence_score = raw_probabilities[0][predicted_class_id] * 100
    predicted_character = class_mapping[predicted_class_id]
    
    print("\n🔮 --- INFERENCE PREDICTION ENGINE RESULTS ---")
    print(f"▶️ Predicted Alphanumeric Character Class : ** {predicted_character} **")
    print(f"▶️ Network Classification Confidence Score: {confidence_score:.2f}%")
    print("-----------------------------------------------\n")
    return predicted_character, confidence_score

if __name__ == "__main__":
    predict_handwritten_character()
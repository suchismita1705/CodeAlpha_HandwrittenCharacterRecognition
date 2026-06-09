import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def download_and_save_emnist():
    print("⏳ Step 1: Connecting to TensorFlow Datasets and downloading EMNIST (Balanced)...")
    
    # Target directory path relative to your D: drive project root
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        # Load the 47 balanced classes split (digits + uppercase/lowercase letters)
        ds_train, ds_info = tfds.load(
            'emnist/balanced',
            split='train',
            shuffle_files=False,
            as_supervised=True,
            with_info=True
        )
        ds_test = tfds.load(
            'emnist/balanced',
            split='test',
            shuffle_files=False,
            as_supervised=True
        )
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("💡 Suggestion: Verify your internet connection or check your proxy settings.")
        return

    print("📊 Step 2: Extracting dataset streams into NumPy tensor matrices...")
    
    train_images, train_labels = [], []
    for img, lbl in tfds.as_numpy(ds_train):
        train_images.append(img)
        train_labels.append(lbl)
        
    test_images, test_labels = [], []
    for img, lbl in tfds.as_numpy(ds_test):
        test_images.append(img)
        test_labels.append(lbl)

    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    X_test = np.array(test_images)
    y_test = np.array(test_labels)

    print("💾 Step 3: Serializing matrix partitions locally to disk storage...")
    
    np.save(os.path.join(raw_dir, "X_train.npy"), X_train)
    np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
    np.save(os.path.join(raw_dir, "X_test.npy"), X_test)
    np.save(os.path.join(raw_dir, "y_test.npy"), y_test)
    
    print("\n✅ DATA ACQUISITION COMPLETE!")
    print(f"-> Total Training Set Size: {X_train.shape[0]} images, dimensions: {X_train.shape[1:4]}")
    print(f"-> Total Testing Set Size: {X_test.shape[0]} images, dimensions: {X_test.shape[1:4]}")
    print(f"-> Unique Character Categories: {ds_info.features['label'].num_classes}")
    print(f"-> Cached files saved successfully inside: {os.path.abspath(raw_dir)}")

if __name__ == "__main__":
    download_and_save_emnist()
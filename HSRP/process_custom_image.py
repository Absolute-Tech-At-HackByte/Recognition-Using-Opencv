import os
import sys
import time
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# Add parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from license_plate_detector import LicensePlateDetector

def select_image():
    """Open file dialog to select an image file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    return file_path if file_path else None

def select_model():
    """Find available models and let user select one"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find trained models
    model_paths = []
    detect_dir = os.path.join(project_root, "runs", "detect")
    if os.path.exists(detect_dir):
        for train_dir in sorted(os.listdir(detect_dir), reverse=True):
            model_path = os.path.join(detect_dir, train_dir, "weights", "best.pt")
            if os.path.exists(model_path):
                model_paths.append((train_dir, model_path))
    
    if not model_paths:
        print("Error: No trained models found.")
        return None
    
    # If only one model, select it automatically
    if len(model_paths) == 1:
        print(f"Using model: {model_paths[0][0]}")
        return model_paths[0][1]
    
    # Otherwise, let user select
    print("Available models:")
    for i, (train_dir, _) in enumerate(model_paths):
        print(f"  {i+1}. {train_dir}")
    
    while True:
        try:
            choice = int(input("\nSelect model (number): "))
            if 1 <= choice <= len(model_paths):
                return model_paths[choice-1][1]
            print(f"Please enter a number between 1 and {len(model_paths)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None

def main():
    """Main function to process a custom image"""
    print("=== License Plate Detector - Custom Image Processing ===")
    
    # Select model
    model_path = select_model()
    if not model_path:
        print("Model selection canceled or no models available.")
        return
    
    # Initialize detector
    try:
        detector = LicensePlateDetector(
            model_path=model_path,
            device="0" if LicensePlateDetector.is_cuda_available() else "cpu",
            conf_threshold=0.3
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Select and process image
    while True:
        print("\nSelect an image to process (or press Cancel to exit)")
        image_path = select_image()
        
        if not image_path:
            print("Image selection canceled.")
            break
        
        try:
            print(f"Processing image: {image_path}")
            
            # Prepare output path
            output_dir = os.path.join(script_dir, "custom_results")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            
            # Process the image
            start_time = time.time()
            annotated_img, plate_images, plate_info = detector.process_image(
                image_path,
                save_path=output_path,
                show_result=True  # Show the result immediately
            )
            elapsed_time = time.time() - start_time
            
            # Display results
            print(f"\nDetection completed in {elapsed_time:.3f} seconds")
            print(f"Detected {len(plate_images)} license plates")
            
            for i, info in enumerate(plate_info):
                print(f"  Plate {i+1}: {info['class_name']} (confidence: {info['confidence']:.2f})")
            
            print(f"\nOutput saved to: {output_path}")
            
            # Ask user if they want to process another image
            if input("\nProcess another image? (y/n): ").lower() != 'y':
                break
                
        except Exception as e:
            print(f"Error processing image: {e}")
            if input("\nTry another image? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main() 
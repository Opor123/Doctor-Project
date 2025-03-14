import requests
import json
import os
import time
from PIL import Image
import io
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 10  # seconds


def generate_breast_cancer_data():
    """Generate realistic breast cancer test data"""
    sample = {
        "radius_mean": round(random.uniform(6, 30), 2),
        "texture_mean": round(random.uniform(9, 40), 2),
        "perimeter_mean": round(random.uniform(40, 200), 2),
        "area_mean": round(random.uniform(100, 2500), 2),
        "smoothness_mean": round(random.uniform(0.05, 0.2), 4),
        "compactness_mean": round(random.uniform(0.02, 0.35), 4),
        "concavity_mean": round(random.uniform(0.0, 0.43), 4),
        "concave_points_mean": round(random.uniform(0.0, 0.2), 4),
        "symmetry_mean": round(random.uniform(0.1, 0.3), 4),
        "fractal_dimension_mean": round(random.uniform(0.05, 0.1), 4),
        "radius_se": round(random.uniform(0.1, 2), 4),
        "texture_se": round(random.uniform(0.2, 5), 4),
        "perimeter_se": round(random.uniform(0.5, 10), 4),
        "area_se": round(random.uniform(5, 100), 2),
        "smoothness_se": round(random.uniform(0.002, 0.03), 4),
        "compactness_se": round(random.uniform(0.002, 0.1), 4),
        "concavity_se": round(random.uniform(0.0, 0.4), 4),
        "concave_points_se": round(random.uniform(0.0, 0.05), 4),
        "symmetry_se": round(random.uniform(0.01, 0.08), 4),
        "fractal_dimension_se": round(random.uniform(0.001, 0.03), 4),
        "radius_worst": round(random.uniform(7, 40), 2),
        "texture_worst": round(random.uniform(10, 50), 2),
        "perimeter_worst": round(random.uniform(50, 300), 2),
        "area_worst": round(random.uniform(200, 4000), 2),
        "smoothness_worst": round(random.uniform(0.06, 0.3), 4),
        "compactness_worst": round(random.uniform(0.02, 1.5), 4),
        "concavity_worst": round(random.uniform(0.0, 1.2), 4),
        "concave_points_worst": round(random.uniform(0.0, 0.4), 4),
        "symmetry_worst": round(random.uniform(0.1, 0.5), 4),
        "fractal_dimension_worst": round(random.uniform(0.05, 0.2), 4)
    }

    # For API compatibility, put the values into a "symptoms" array
    symptoms = list(sample.values())

    return {"symptoms": symptoms}, sample


def generate_limit_model_data():
    """Generate realistic data for the breast cancer dataset with specified columns"""

    # Define possible values for categorical features
    menopause_values = ["lt40", "ge40", "premeno"]
    tumor_size_values = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"]
    inv_nodes_values = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"]
    node_caps_values = ["yes", "no"]
    deg_malig_values = [1, 2, 3]
    breast_values = ["left", "right"]
    breast_quad_values = ["left_up", "left_low", "right_up", "right_low", "central"]
    irradiat_values = ["yes", "no"]

    # Generate sample with appropriate values for each column
    # Use underscores instead of hyphens to match the API's expected input format
    sample = {
        "Age": random.randint(20, 90),
        "Menopause": random.choice(menopause_values),
        "Tumor_Size": random.choice(tumor_size_values),  # Changed from Tumor-Size
        "Inv_Nodes": random.choice(inv_nodes_values),    # Changed from Inv-Nodes
        "Node_Caps": random.choice(node_caps_values),    # Changed from Node-Caps
        "Deg_Malig": random.choice(deg_malig_values),
        "Breast": random.choice(breast_values),
        "Breast_Quad": random.choice(breast_quad_values), # Changed from Breast-Quad
        "Irradiat": random.choice(irradiat_values)
    }

    return sample

def test_home():
    """Test the home endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        print(f"Home Route (Status {response.status_code}):", response.json())
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Home Route Test Failed: {e}")
        return False


def test_status():
    """Test the status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/status/", timeout=TIMEOUT)
        print(f"Status Endpoint (Status {response.status_code}):", response.json())
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Status Endpoint Test Failed: {e}")
        return False


def test_predict_breast_cancer():
    """Test the breast cancer prediction endpoint with realistic data"""
    try:
        # Generate realistic test data
        test_input, original_data = generate_breast_cancer_data()

        print(f"Sending breast cancer prediction request with payload:")
        print(json.dumps(test_input, indent=2))

        response = requests.post(f"{BASE_URL}/predict/", json=test_input, timeout=TIMEOUT)

        print(f"Breast Cancer Model Prediction (Status {response.status_code}):", end=" ")
        if response.status_code == 200:
            result = response.json()
            print(result)

            # Display the prediction result with the original input data
            print("\nOriginal Input Data:")
            print(json.dumps(original_data, indent=2))
            print("\nPrediction Result:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Breast Cancer Prediction Test Failed: {e}")
        return False


def test_predict_limit_model():
    """Test the limit model prediction endpoint with realistic data"""
    try:
        # Generate realistic test data
        test_input = generate_limit_model_data()  # Returns a single dictionary

        print(f"Sending limit model prediction request with payload:")
        print(json.dumps(test_input, indent=2))

        # Send the dictionary directly (no list)
        response = requests.post(f"{BASE_URL}/predict/", json=test_input, timeout=TIMEOUT)

        print(f"Limit Model Prediction (Status {response.status_code}):", end=" ")
        if response.status_code == 200:
            result = response.json()
            print(result)

            print("\nPrediction Result:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Limit Model Prediction Test Failed: {e}")
        return False


def test_image_prediction():
    """Test the image prediction endpoint with a synthetic image if no real image is available"""
    try:
        # Try to find an existing image first
        image_paths = [
            "Model/Image_recognize/old data/Split_dataset/test/normal/malignant (5).png",
            "Model/Image_recognize/New_image_training/test/mdb012rl_jpg.rf.e29872d0a71d05157a9a6e438269ab5f.jpg",
            "Model/Image_recognize/New_image_training/test/mdb032rl_jpg.rf.078c97741875c9b0e30ce3c26c903439.jpg"
        ]

        image_file = None
        image_path_used = None

        # Check if any of the images exist
        for path in image_paths:
            if os.path.exists(path):
                image_path_used = path
                with open(path, "rb") as f:
                    image_file = f.read()
                break

        # If no image found, create a synthetic one
        if image_file is None:
            print("No test image found. Creating a synthetic test image.")
            # Create a simple synthetic image (640x640 with a gradient)
            img = Image.new('RGB', (640, 640), color='white')
            pixels = img.load()

            # Create a simple pattern
            for i in range(img.width):
                for j in range(img.height):
                    # Create a gradient pattern
                    r = int(255 * i / img.width)
                    g = int(255 * j / img.height)
                    b = int(255 * (i + j) / (img.width + img.height))
                    pixels[i, j] = (r, g, b)

            # Save to a buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            image_file = buffer.getvalue()
            image_path_used = "synthetic_test_image.jpg"

            # Optionally save the synthetic image for debugging
            with open("synthetic_test_image.jpg", "wb") as f:
                f.write(image_file)

        # Send the image to the API
        print(f"Sending image prediction request with image: {image_path_used}")
        files = {"file": (f"test_image.jpg", image_file, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/predict-image/", files=files, timeout=TIMEOUT)

        print(f"Image Prediction (Status {response.status_code}):", end=" ")
        if response.status_code == 200:
            result = response.json()

            # Display the detailed prediction result in a formatted way
            print("\n\nImage Prediction Results:")
            print("=" * 40)
            print(f"Diagnosis: {result.get('diagnosis', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Result Code: {result.get('result_code', 'N/A')}")

            # Map result code to a human-readable meaning if possible
            result_code = result.get('result_code')
            if result_code is not None:
                result_meaning = {
                    0: "Benign",
                    1: "Malignant"
                }.get(result_code, "Unknown")
                print(f"Result Meaning: {result_meaning}")

            # Display any additional information that might be in the result
            additional_keys = [key for key in result.keys() if key not in ['diagnosis', 'confidence', 'result_code']]
            if additional_keys:
                print("\nAdditional Information:")
                for key in additional_keys:
                    print(f"{key}: {result[key]}")

            print("=" * 40)

            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Image Prediction Test Failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Image Prediction Test Failed (Other Error): {str(e)}")
        return False


def test_health_tips():
    """Test the health tips endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health-tips/", timeout=TIMEOUT)
        print(f"Health Tips (Status {response.status_code}):", end=" ")
        if response.status_code == 200:
            result = response.json()
            print(result)
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health Tips Test Failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide a summary"""
    print("\n=== 🔍 Running DoctorAI API Tests ===\n")

    # Wait for server to be ready
    print("Waiting for server to be ready...")
    server_ready = False
    for _ in range(5):  # Try 5 times
        try:
            requests.get(f"{BASE_URL}/", timeout=2)
            server_ready = True
            break
        except requests.exceptions.RequestException:
            print("Server not ready, waiting...")
            time.sleep(2)

    if not server_ready:
        print("⚠️ Warning: Server might not be running. Proceeding with tests anyway.")

    # Run tests and collect results
    results = {}

    print("\n🧪 Testing Basic Endpoints:")
    results["Home"] = test_home()
    results["Status"] = test_status()

    print("\n🧪 Testing Prediction Endpoints:")
    results["Breast Cancer Model"] = test_predict_breast_cancer()
    results["Limit Model"] = test_predict_limit_model()
    results["Image Prediction"] = test_image_prediction()
    results["Health Tips"] = test_health_tips()

    # Summary
    print("\n=== 📊 Test Results Summary ===")
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)

    print(f"\nTests Passed: {success_count}/{total_count} ({success_count / total_count * 100:.1f}%)")

    for test_name, result in results.items():
        status = "✅ Passed" if result else "❌ Failed"
        print(f"- {test_name}: {status}")

    print("\n=== 🏁 All Tests Completed ===\n")

    return success_count == total_count


if __name__ == "__main__":
    run_all_tests()
import random
import json

def generate_test_data(n=1):
    test_data = []
    for _ in range(n):
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
        test_data.append(sample)
    
    return json.dumps(test_data, indent=4)

# Generate and print 5 test samples
print(generate_test_data(1))

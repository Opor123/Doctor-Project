import random

# Define the range of values for each feature (based on your dataset)
ranges = {
    'radius_mean': (10, 30),
    'texture_mean': (10, 30),
    'perimeter_mean': (70, 200),
    'area_mean': (350, 2500),
    'smoothness_mean': (0.07, 0.20),
    'compactness_mean': (0.06, 0.40),
    'concavity_mean': (0.06, 0.40),
    'concave_points_mean': (0.05, 0.15),
    'symmetry_mean': (0.15, 0.25),
    'fractal_dimension_mean': (0.05, 0.10),
    'radius_se': (0.5, 2),
    'texture_se': (0.5, 2),
    'perimeter_se': (0.5, 3),
    'area_se': (5, 150),
    'smoothness_se': (0.004, 0.08),
    'compactness_se': (0.01, 0.08),
    'concavity_se': (0.01, 0.08),
    'concave_points_se': (0.01, 0.08),
    'symmetry_se': (0.01, 0.05),
    'fractal_dimension_se': (0.01, 0.05),
    'radius_worst': (15, 30),
    'texture_worst': (10, 30),
    'perimeter_worst': (80, 250),
    'area_worst': (500, 3000),
    'smoothness_worst': (0.08, 0.25),
    'compactness_worst': (0.10, 0.80),
    'concavity_worst': (0.10, 0.80),
    'concave_points_worst': (0.10, 0.40),
    'symmetry_worst': (0.20, 0.40),
    'fractal_dimension_worst': (0.05, 0.10)
}

# Function to generate a random row based on these ranges
def generate_random_row():
    return [random.uniform(ranges[key][0], ranges[key][1]) for key in ranges]

# Example: generate a random row
for i in range(20): 
    random_row = generate_random_row()
    print(', '.join(f'{value:.4f}' for value in random_row))
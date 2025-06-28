import kagglehub

# Download latest version
path = kagglehub.dataset_download("atomscott/soccertrack")

print("Path to dataset files:", path)
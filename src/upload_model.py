from huggingface_hub import HfApi


repo_id = "GustavoGomes7/VisionNet-CIFAR10"  
model_path = "models/best_model.keras"  
destination_path = "best_model.keras" 


api = HfApi()


api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=destination_path,
    repo_id=repo_id,
    repo_type="model"
)

print(f"Model uploaded successfully to {repo_id}!")

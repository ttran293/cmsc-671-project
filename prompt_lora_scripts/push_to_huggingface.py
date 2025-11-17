from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

login(token=hf_token)

model_folder = os.path.join(os.path.dirname(__file__), "llama3-lora")
repo_id = "ttran19/llama3-lora-671"

api = HfApi(token=hf_token)

api.upload_folder(
    folder_path=model_folder,
    repo_id=repo_id,
    repo_type="model",
)
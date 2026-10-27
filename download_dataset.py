from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Fhrozen/FSD50k",
    repo_type="dataset",
    local_dir="D:/datasets/audio-dataset"
)

print("Dataset tersimpan di:", path)
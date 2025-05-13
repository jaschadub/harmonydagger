import os
import requests

def download_file(url, local_path):
    """Download a file from a URL to a local path"""
    print(f"Downloading {url} to {local_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192): 
            f.write(chunk)
    
    print(f"Download complete: {local_path} ({os.path.getsize(local_path)} bytes)")
    return local_path

# Replace with actual URL - using a sample WAV file as placeholder
sample_url = "https://download.samplelib.com/wav/sample-15s.wav"
download_file(sample_url, "cusb_ed_80694_01_7534_0c.wav")

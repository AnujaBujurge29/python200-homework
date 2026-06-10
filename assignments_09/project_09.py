# Video Link: https://youtu.be/JyQyoW781EE
import os
import requests
import json
import pandas as pd
from datetime import date
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

os.system('cls')

# =====================================================================================
#                    Week 9 - Part 2: Project -- Extract + Load Pipeline
# =====================================================================================

print('='*100)
print("Part 2: Project -- Extract + Load Pipeline")
print('='*100)

# =====================================================================================
# Setup
# =====================================================================================

print("---------------------- Setup ----------------------")
print('='*100)

ACCOUNT_URL = "https://anujactd2026sa.blob.core.windows.net/"
CONTAINER = "pipeline-data"

print("Setup Completed.")

# =====================================================================================
# Step 1: Extract
# =====================================================================================

print('-'*100)
print("=> Step 1: Extract -- Call the Open-Meteo API to retrieve 7 days of hourly weather data")
print('-'*100)

# Example coordinates: New York City
LATITUDE = 35.2271
LONGITUDE = -80.8431

API_URL = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=temperature_2m,precipitation&forecast_days=7"

response = requests.get(API_URL)
response.raise_for_status()

weather_data = response.json()
print(
    f"Extracted {len(weather_data['hourly']['time'])} hourly records from Open Metro API.")

# print(f"{weather_data['hourly']['temperature_2m']}")

# =====================================================================================
# Step 2: Serialize
# =====================================================================================

print('-'*100)
print("=> Step 2: Serialize -- Convert the API response to JSON bytes")
print('-'*100)

weather_bytes = json.dumps(weather_data).encode('utf-8')
print(f"Sterized Data size: {len(weather_bytes)} bytes.")

# =====================================================================================
# Step 3: Load
# =====================================================================================

print('-'*100)
print("=> Step 3: Load -- Upload the serialized data to your container at the path")
print('-'*100)

credentials = DefaultAzureCredential()
blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL, credential=credentials)
container_client = blob_service_client.get_container_client(CONTAINER)

# Create the container if it doesn't exist
try:
    container_client.get_container_properties()
except ResourceNotFoundError:
    print(f"Container '{CONTAINER}' does not exist. Creating it now...")
    container_client.create_container()
    print(f"Container '{CONTAINER}' created successfully.")

blob_path = f'raw/{date.today().isoformat()}/weather.json'
blob_client = container_client.get_blob_client(blob_path)
blob_client.upload_blob(weather_bytes, overwrite=True)

print(f"Uploaded to: {blob_path} ({len(weather_bytes)} bytes).")


# =====================================================================================
# Step 4: Verify
# =====================================================================================

print('-'*100)
print("=> Step 4: Verify")
print('-'*100)

for blob in container_client.list_blobs():
    print(f"{blob.name} - {blob.size} bytes.")

# =====================================================================================
# Step 5: Read Back
# =====================================================================================

print('-'*100)
print("=> Step 5: Read Back")
print('-'*100)

downloaded_bytes = blob_client.download_blob().readall()
downloaded_data = json.loads(downloaded_bytes)

df = pd.DataFrame(downloaded_data['hourly'])

print(df.head())

# Save the downloaded JSON to outputs/weather_raw.json
# DATA_PATH = "../assignments_09/outputs"
path = Path(__file__).parent / "outputs"

# path = DATA_PATH
path.mkdir(parents=True, exist_ok=True)

output_path = path / "weather_raw.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(downloaded_data, f, indent=2)

print(f"Saved raw JSON to {output_path}")

# =====================================================================================
# Video
# =====================================================================================

print('-'*100)
print("Video Link Added")
print(f"https://youtu.be/JyQyoW781EE")
print('-'*100)

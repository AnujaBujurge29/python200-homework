import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


os.system('cls')

# =====================================================================================
#                    Week 10 - Part 2: Project -- LLM Transform Pipeline
# =====================================================================================

print('='*100)
print("Week 10 - Part 2: Project -- LLM Transform Pipeline")
print('='*100)

# =====================================================================================
# Setup
# =====================================================================================

print("=> Setup")
print('='*100)

ACCOUNT_URL = "https://anujactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

print("Setup Completed.")

# =====================================================================================
# Step 1: Read
# =====================================================================================

print('-'*100)
print("=> Step 1: Read weather data")
print('-'*100)


# Try to load the weather data from Week 9
weather_file = "../assignments_09/outputs/weather_raw.json"

try:
    with open(weather_file, 'r') as f:
        weather_data = json.load(f)
    print(f"✓ Loaded weather data from {weather_file}")
except FileNotFoundError:
    # Fallback to the resource dataset
    fallback_file = "../../python-200/assignments/resources/weather_raw.json"
    print(f"✗ {weather_file} not found, trying fallback...")
    try:
        with open(fallback_file, 'r') as f:
            weather_data = json.load(f)
        print(f"✓ Loaded fallback weather data from {fallback_file}")
    except FileNotFoundError:
        print(f"✗ No weather data found. Please check file paths.")
        weather_data = None

# Parse and reshape the hourly data
if weather_data:
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    precipitations = hourly.get("precipitation", [])

    # Reshape into list of record dictionaries
    records = []
    for i in range(min(len(times), len(temperatures), len(precipitations))):
        record = {
            "time": times[i],
            "temperature_2m": temperatures[i],
            "precipitation": precipitations[i]
        }
        records.append(record)

    print(f"✓ Reshaped {len(records)} hourly records")
    print(f"  First record: {records[0]}")
    print()

# =====================================================================================
# Step 2: Transform
# =====================================================================================

print('-'*100)
print("=> Step 2: Classify weather with OpenAI API")
print('-'*100)


# Initialize OpenAI client
client = OpenAI()

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

# Process only the first 24 records
records_to_process = records[:24]
classified_records = []

print(f"Processing {len(records_to_process)} records...")
print()

for idx, record in enumerate(records_to_process, 1):
    # Create user message with temperature and precipitation
    temp = record["temperature_2m"]
    precip = record["precipitation"]
    user_message = f"Temperature: {temp}C, Precipitation: {precip}mm"

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0
        )

        # Extract and validate the classification
        classification = response.choices[0].message.content.strip().lower()

        # Validate against allowed values
        if classification not in ["good", "marginal", "bad"]:
            classification = "unknown"

        # Add classification to record
        record["running_classification"] = classification
        classified_records.append(record)

        # Print progress every 6 records
        if idx % 6 == 0:
            print(f"✓ Processed {idx} records")

    except Exception as e:
        print(f"✗ Error processing record {idx}: {str(e)}")
        record["running_classification"] = "unknown"
        classified_records.append(record)

print()
print(
    f"✓ Classification complete. Processed {len(classified_records)} records")
print()

# Show sample results
print("Sample classified records:")
print('-'*100)
for i in [0, 5, 10, 15, 20, 23]:
    if i < len(classified_records):
        rec = classified_records[i]
        print(
            f"  {rec['time']}: {rec['temperature_2m']}°C, {rec['precipitation']}mm → {rec['running_classification']}")

# =====================================================================================
# Step 3: Write
# =====================================================================================

print('-'*100)
print("=> Step 3: Upload to Blob Storage")
print('-'*100)


# Prepare enriched records with "conditions" field (rename classification field)
enriched_records = []
for record in classified_records:
    enriched_record = record.copy()
    enriched_record["conditions"] = enriched_record.pop(
        "running_classification")
    enriched_records.append(enriched_record)

# Upload to Blob Storage using DefaultAzureCredential (if available)
blob_path = f"{ACCOUNT_URL}/{CONTAINER}/processed/weather_classified.json"
try:
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

    blob_client = container_client.get_blob_client(
        "processed/weather_classified.json")

    # Convert to JSON and upload with overwrite
    blob_data = json.dumps(enriched_records, indent=2)
    blob_client.upload_blob(blob_data, overwrite=True)

    print(
        f"✓ Uploaded {len(enriched_records)} enriched records to {blob_path}")
except Exception as e:
    # Fall back to local-only workflow but keep enriched_records
    print(f"✗ Error uploading to Blob Storage: {str(e)}")
    print("Proceeding with local files only.")

print()

# =====================================================================================
# Step 4: Spot-Check
# =====================================================================================

print('-'*100)
print("=> Step 4: Spot-Check - Download and Analyze")
print('-'*100)


try:
    # Attempt to download using DefaultAzureCredential + BlobServiceClient
    credentials = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL, credential=credentials)
    container_client = blob_service_client.get_container_client(CONTAINER)
    blob_client = container_client.get_blob_client(
        "processed/weather_classified.json")

    blob_data = blob_client.download_blob().readall().decode('utf-8')
    downloaded_records = json.loads(blob_data)

    # Load into DataFrame
    df = pd.DataFrame(downloaded_records)

    print("✓ Downloaded and loaded into DataFrame")
    print()
    print("Conditions value counts:")
    print(df["conditions"].value_counts())
    print()
    print("First 5 rows:")
    print(df.head())

except Exception as e:
    print(f"✗ Error in spot-check: {str(e)}")
    print(f"  Using local enriched_records instead...")
    try:
        df = pd.DataFrame(enriched_records)
        print()
        print("Conditions value counts:")
        print(df["conditions"].value_counts())
        print()
        print("First 5 rows:")
        print(df.head())
    except Exception as e2:
        print(f"Could not create DataFrame: {str(e2)}")


# =====================================================================================
# Step 5: Save Output
# =====================================================================================

print('-'*100)
print("=> Step 5: Save First 10 Records Locally")
print('-'*100)

# Save first 10 enriched records to outputs folder
first_10 = enriched_records[:10] if enriched_records else []

output_file = "outputs/first_10_records.json"

try:
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(first_10, f, indent=2)
    print(f"Saved first 10 records to {output_file}")
except Exception as e:
    print(f"Error saving output: {str(e)}")

print()
# print('='*100)
# print("Pipeline complete!")
# print('='*100)

# =====================================================================================
# Step 6: Reflect
# =====================================================================================

print('-'*100)
print("=> Step 6: Reflect")
print('-'*100)

# =====================================================================================
# Reflection: Was LLM the Right Choice?
# =====================================================================================
#
# Classifying weather for outdoor running with an LLM was OVERKILL for this specific task.
# An LLM adds cost and latency for what could be a simple rule-based system. A deterministic
# approach (e.g., temp > 15°C and precip < 1mm → good) would be faster, cheaper, and more
# predictable. However, the LLM approach has advantages: it can handle edge cases intuitively
# (e.g., "freezing rain" or "extreme heat"), it's interpretable to non-technical stakeholders,
# and it scales to more complex scenarios (humidity, wind, UV index). For a production system
# at scale, rules would likely be better; for a one-off demo or exploratory analysis, the LLM
# is reasonable.
#

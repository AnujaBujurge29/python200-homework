import os
import json
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from openai import OpenAI

os.system('cls')
load_dotenv()

ACCOUNT_URL = "https://anujactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# =====================================================================================
#                    Week 11 - Part 2: Project -- Full ETL Pipeline
# =====================================================================================

# Video link:
# https://www.youtube.com/watch?v=JkQyoW781EE&t=134

print('='*100)
print("Week 11 - Part 2: Project -- Full ETL Pipeline")
print('='*100)

# Charlotte, NC
LATITUDE = 35.2271
LONGITUDE = -80.8431


@task(retries=2, retry_delay_seconds=10)
def extract_weather_data(latitude: float = LATITUDE, longitude: float = LONGITUDE):
    """Extract 7 days of hourly weather data from Open-Meteo."""
    logger = get_run_logger()
    start_date = date.today()
    end_date = start_date + timedelta(days=6)
    api_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&start_date={start_date.isoformat()}"
        f"&end_date={end_date.isoformat()}"
        "&timezone=UTC"
    )

    response = requests.get(api_url)
    response.raise_for_status()
    weather_data = response.json()
    logger.info("Extract completed: Open-Meteo response received.")
    print("✓ Extract task completed: raw weather data loaded.")
    return weather_data


@task
def transform_weather_data(raw_weather: dict):
    """Transform hourly weather time series into enriched records."""
    logger = get_run_logger()
    if not raw_weather or "hourly" not in raw_weather:
        raise ValueError(
            "No hourly weather data available for transformation.")

    hourly = raw_weather["hourly"]
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    precipitations = hourly.get("precipitation", [])
    record_count = min(len(times), len(temperatures), len(precipitations))

    records = [
        {
            "time": times[i],
            "temperature_2m": temperatures[i],
            "precipitation": precipitations[i],
        }
        for i in range(record_count)
    ]

    enriched_records = []
    system_prompt = (
        "You are classifying hourly weather conditions for outdoor running. "
        "Given a temperature in Celsius and a precipitation amount in mm, "
        "classify the conditions as exactly one of: good, marginal, or bad. "
        "Reply with that one word only -- no punctuation, no explanation."
    )
    client = OpenAI()

    for idx, record in enumerate(records[:24], start=1):
        user_message = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
        classification = "unknown"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )
            classification = response.choices[0].message.content.strip(
            ).lower()
            if classification not in ["good", "marginal", "bad"]:
                classification = "unknown"
        except Exception as exc:
            logger.warning(
                f"OpenAI classification failed for record {idx}: {exc}")
            classification = "unknown"

        record["running_classification"] = classification
        enriched_records.append(record)

        if idx % 6 == 0:
            logger.info(f"Transformed {idx} records")
            print(f"✓ Transformed {idx} records")

    logger.info("Transform completed: enriched weather records prepared.")
    print("✓ Transform task completed: enriched records ready.")
    return enriched_records


@task
def load_weather_data(enriched_records: list):
    """Load enriched weather records to Azure Blob Storage."""
    logger = get_run_logger()
    if not enriched_records:
        raise ValueError("No enriched records to upload.")

    blob_relpath = f"final/{date.today().isoformat()}/weather_etl.json"
    blob_path = f"{ACCOUNT_URL}/{CONTAINER}/{blob_relpath}"
    blob_data = json.dumps(enriched_records, indent=2)
    data_bytes = blob_data.encode("utf-8")

    credentials = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL, credential=credentials
    )
    container_client = blob_service_client.get_container_client(CONTAINER)

    try:
        container_client.get_container_properties()
    except ResourceNotFoundError:
        logger.info(f"Container '{CONTAINER}' not found; creating it.")
        container_client.create_container()

    blob_client = container_client.get_blob_client(blob_relpath)
    blob_client.upload_blob(data_bytes, overwrite=True)

    logger.info(
        f"Load completed: uploaded {len(data_bytes)} bytes to {blob_path}")
    print(
        f"✓ Load task completed: uploaded {len(data_bytes)} bytes to {blob_path}")
    return blob_path


@flow(log_prints=True)
def weather_etl_pipeline():
    raw_weather = extract_weather_data()
    enriched_records = transform_weather_data(raw_weather)
    final_blob_path = load_weather_data(enriched_records)
    print(f"Pipeline complete. Final blob path: {final_blob_path}")
    return final_blob_path


if __name__ == "__main__":
    weather_etl_pipeline()

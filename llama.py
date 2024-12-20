import json
import os
import time

import requests
from datasets import load_from_disk
from google.auth.transport.requests import Request
from google.cloud import storage
from google.oauth2 import service_account
from tqdm import tqdm

import wandb
from constants import WANDB_API_KEY, WANDB_USERNAME, WANDB_PROJECT, GOOGLE_CLOUD_PROJECT_NAME, GOOGLE_CLOUD_REGION, \
    GOOGLE_CLOUD_BUCKET_NAME, GOOGLE_CLOUD_CREDENTIALS_FILE


# Function to check if the file exists in GCS
def file_exists_in_gcs(bucket_name, destination_blob_name):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    return blob.exists()


# Upload image to GCS
def upload_to_gcs(image_path, bucket_name, destination_blob_name):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(image_path)
    return f"gs://{bucket_name}/{destination_blob_name}"


def predict_llama(split, lang, prompt):
    run_name = f"llama-{split}-{lang}"

    # Use the WandB API to get the list of runs
    api = wandb.Api()
    runs = api.runs(f"{WANDB_USERNAME}/{WANDB_PROJECT}")

    # Find the run ID by name
    run_id = None
    for run in runs:
        if run.name == run_name:
            run_id = run.id
            break

    last_index = 0
    if run_id is None:
        # If the run does not exist, initialize a new run
        print(f"Starting new run: {run_name}")
        wandb.init(project=WANDB_PROJECT, name=run_name, resume="allow")
        table = wandb.Table(
            columns=["question", "image", "true_answer", "true_short_answer", "predicted_answer", "category"]
        )
    else:
        # If the run exists, resume the run
        run = wandb.init(project=WANDB_PROJECT, name=run_name, id=run_id, resume="must")
        r = api.run(f"{WANDB_USERNAME}/{WANDB_PROJECT}/{run_id}")
        table_dict = None
        for log in r.scan_history():
            if "prediction_table" in log:
                table_dict = log["prediction_table"]
                last_index = table_dict['nrows']
        if table_dict is not None:
            artifact_name = f'{WANDB_USERNAME}/{WANDB_PROJECT}/run-{run_id}-prediction_table:v0'
            artifact = run.use_artifact(artifact_name, type='run_table')
            artifact_dir = artifact.download()

            # Load the table data from the downloaded file
            table_path = os.path.join(artifact_dir, 'prediction_table.table.json')
            with open(table_path, 'r') as f:
                table_data = json.load(f)

            # Create a wandb.Table from the loaded data
            columns = ["question", "image", "true_answer", "true_short_answer", "predicted_answer", "category"]
            table = wandb.Table(columns=columns)
            for row in table_data['data']:
                row[1] = wandb.Image(f"artifacts/run-{run_id}-prediction_table-v0/{row[1]['path']}")
                table.add_data(*row)

    print(last_index)
    index = 0
    for example in tqdm(merged_dataset[split], desc="Predicting answers"):
        if index < last_index:
            index += 1
            continue
        index += 1
        label = example["label"]
        context = example["context"]
        caption = example["caption"]
        image = example['image'].convert('RGB')
        category = example['category']
        if lang == "en":
            true_answer = example['corrected_answer_english']
            true_short_answer = example['short_answer_english']
            question = example['question_english']
        elif lang == "de":
            true_answer = example['corrected_answer_german']
            true_short_answer = example['short_answer_german']
            question = example['question_german']
        else:
            return

        # Load the image and get GCS URI
        image_path = f"Data\\VQAMeta\\training_data\\{split}\\{label}\\{example['img_file_name']}"
        destination_blob_name = f"Data/VQAMeta/training_data/{split}/{label}/{example['img_file_name']}"

        if file_exists_in_gcs(bucket_name, destination_blob_name):
            print(f"File already exists in GCS: gs://{bucket_name}/{destination_blob_name}")
            gcs_image_uri = f"gs://{bucket_name}/{destination_blob_name}"
        else:
            gcs_image_uri = upload_to_gcs(image_path, bucket_name, destination_blob_name)

        metadata_string = (f"Here is some additional info for the image:"
                           f"The label of the image is: {label}."
                           f"The caption of the image is: {caption}.")

        if len(context) > 5:
            metadata_string += f"\nAnd the relevant section from the research paper: {context}."

        text_input = f"{prompt}{metadata_string}\nQuestion: {question}"

        # Kombinierte Eingabe erstellen
        payload = {
            "model": "meta/llama-3.2-90b-vision-instruct-maas",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image_url": {"url": gcs_image_uri}, "type": "image_url"},
                        {"text": text_input, "type": "text"}
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.4,
            "top_k": 10,
            "top_p": 0.95,
            "n": 1
        }

        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(endpoint, json=payload, headers=headers)
            if response.status_code == 200:
                try:
                    prediction = dict(response.json())['choices'][0]['message']['content']
                    print(f"Question: {question}\nPrediction: {prediction}\nTruth: {true_short_answer}")
                    table.add_data(question, wandb.Image(image, caption=caption), true_answer, true_short_answer,
                                   prediction, category)
                    break
                except Exception as e:
                    print(e)
                    print(f"Unexpected response format. Attempt {attempt + 1}/{max_retries}")
            else:
                print(f"Request failed with status {response}. Attempt {attempt + 1}/{max_retries}")
            time.sleep(2)  # Delay before retry
        else:
            # All retries failed
            break
    wandb.log({"prediction_table": table})


if __name__ == '__main__':
    # Initialize Weights and Biases
    wandb.login(key=WANDB_API_KEY)

    # Define the OAuth scope
    SCOPES = ['https://www.googleapis.com/auth/cloud-platform']

    # Laden der Anmeldedaten aus der JSON-Datei
    credentials = service_account.Credentials.from_service_account_file(
        GOOGLE_CLOUD_CREDENTIALS_FILE, scopes=SCOPES)

    project = GOOGLE_CLOUD_PROJECT_NAME
    REGION = GOOGLE_CLOUD_REGION

    # Access token abrufen
    auth_req = Request()
    credentials.refresh(auth_req)
    access_token = credentials.token

    # Endpoint und Headers für die Anfrage
    endpoint = f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{REGION}/endpoints/openapi/chat/completions"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    bucket_name = GOOGLE_CLOUD_BUCKET_NAME

    merged_dataset = load_from_disk("Data\\VQAMetaQA")
    with open("Data\\llama_predict_prompt.txt", "r") as f:
        prompt = f.read()

    predict_llama(split="test", lang="de", prompt=prompt)

import time

from tqdm import tqdm
import wandb
import os
import json

from constants import WANDB_USERNAME, WANDB_PROJECT
from gemini_api import GeminiModel

if __name__ == '__main__':
    model = GeminiModel()

    api = wandb.Api()
    runs = api.runs(f"{WANDB_USERNAME}/{WANDB_PROJECT}")

    table_version = "v0"

    run_id = None
    for run in runs:
        if run.name.endswith("context"):
            run_id = run.id

            run = wandb.init(project=WANDB_PROJECT, name=run.name, id=run_id, resume="must")
            r = api.run(f"{WANDB_USERNAME}/{WANDB_PROJECT}/{run_id}")
            table_dict = None
            for log in r.scan_history():
                if "prediction_table" in log:
                    table_dict = log["prediction_table"]
                    n_rows = table_dict['nrows']

            if table_dict is None:
                wandb.finish()
                continue

            artifact_name = f'{WANDB_USERNAME}/{WANDB_PROJECT}/run-{run_id}-prediction_table:{table_version}'
            artifact = run.use_artifact(artifact_name, type='run_table')
            artifact_dir = artifact.download()

            # Load the table data from the downloaded file
            table_path = os.path.join(artifact_dir, 'prediction_table.table.json')
            with open(table_path, 'r') as f:
                table_data = json.load(f)

            # Create a wandb.Table from the loaded data with a new column if it doesn't exist
            columns = table_data['columns']
            if 'eval_response' not in columns:
                columns.append('eval_response')
            else:
                wandb.finish()
                continue
            table = wandb.Table(columns=columns)

            correct_predictions = 0
            number_of_predictions = 0
            # Add a progress bar
            for row in tqdm(table_data['data'], desc="Processing rows"):
                question = row[0]
                predicted_answer = row[4]
                true_short_answer = row[3]
                true_answer = row[2]

                if len(row) == len(columns) - 1:
                    # Call the Gemini API for each row
                    retry = 0
                    max_retries = 3
                    while retry < max_retries:
                        try:
                            response = model.eval_answer(question, predicted_answer, true_answer)
                            eval_response = response.text
                            break
                        except Exception as e:
                            retry += 1
                            time.sleep(5)
                    else:
                        print(e)
                        wandb.finish()
                        exit(-2)
                else:
                    eval_response = row[-1]

                print(f"Pred Answer: {predicted_answer}, Gt Answer: {true_answer}, Eval Response: {eval_response}")

                row[1] = wandb.Image(f"artifacts/run-{run_id}-prediction_table-{table_version}/{row[1]['path']}")
                if len(row) == len(columns) - 1:  # If eval_response column was added
                    row.append(eval_response)
                else:
                    row[-1] = eval_response  # Update existing eval_response column
                number_of_predictions +=1
                if 'yes' in eval_response.lower() or 'ja' in eval_response.lower():  # Adjust this condition as needed
                    correct_predictions += 1

                table.add_data(*row)

            accuracy = correct_predictions / number_of_predictions
            print("ACCURACY:", accuracy)
            # Update the table in wandb
            wandb.log({"prediction_table": table, "GeminiAccuracy": accuracy})
            wandb.finish()


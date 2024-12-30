import time
from tqdm import tqdm
import wandb
import os
import json
import pandas as pd
import numpy as np
from bert_score import score
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from constants import WANDB_USERNAME, WANDB_PROJECT

import warnings
warnings.filterwarnings("ignore", message=r"Some weights of (.*) were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", message=r"You should probably TRAIN this model on a downstream task")


def calculate_bleu_score(candidate, reference):
    candidate_tokens = word_tokenize(candidate)
    reference_tokens = [word_tokenize(reference)]
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

rouge = Rouge()

def calculate_rouge_scores(candidate, reference):
    if candidate and reference:
        scores = rouge.get_scores(candidate, reference)
        return scores[0]
    else:
        return None

def calculate_meteor_score(candidate, reference):
    if candidate and reference:  # Ensure neither is empty
        candidate_tokens = word_tokenize(candidate)
        reference_tokens = word_tokenize(reference)
        return meteor_score([reference_tokens], candidate_tokens)
    else:
        return None

if __name__ == '__main__':
    api = wandb.Api()
    runs = api.runs(f"{WANDB_USERNAME}/{WANDB_PROJECT}")

    table_version = "latest"
    run_id = None
    for run in runs:
        if run.name:
            if "-en" in run.name:
                lang = "en"
            else:
                lang = "de"
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

            # List all versions of the artifact and find the latest version
            artifact_name = f'{WANDB_USERNAME}/{WANDB_PROJECT}/run-{run_id}-prediction_table'
            artifact_versions = api.artifact_versions('run_table', artifact_name)
            latest_version = sorted(artifact_versions, key=lambda v: v.version, reverse=True)[0].version
            artifact_name = f'{artifact_name}:{latest_version}'
            print("latest version:", latest_version)
            artifact = run.use_artifact(artifact_name, type='run_table')
            artifact_dir = artifact.download()

            # Load the table data from the downloaded file
            table_path = os.path.join(artifact_dir, 'prediction_table.table.json')
            with open(table_path, 'r') as f:
                table_data = json.load(f)

            # Create a wandb.Table from the loaded data with new columns if they don't exist
            columns = table_data['columns']
            if "bleu_score" not in columns:
                columns.extend(["bleu_score", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "meteor_score",
                                "bert_precision", "bert_recall", "bert_f1"])
            else:
                wandb.finish()
                continue

            table = wandb.Table(columns=columns)

            correct_predictions = 0
            number_of_predictions = 0

            bleu_scores = []
            rouge_1_f1_scores = []
            rouge_2_f1_scores = []
            rouge_l_f1_scores = []
            meteor_scores = []
            bert_precisions = []
            bert_recalls = []
            bert_f1s = []

            # Add a progress bar
            for row in tqdm(table_data['data'], desc="Processing rows"):
                question = row[0]
                predicted_answer = row[4]
                true_short_answer = row[3]
                true_answer = row[2]

                # Calculate metrics
                rouge_scores = None
                bleu_score = calculate_bleu_score(predicted_answer, true_answer)
                try:
                    rouge_scores = calculate_rouge_scores(predicted_answer, true_answer)
                except Exception as e:
                    pass
                if rouge_scores:
                    rouge_1_f1 = rouge_scores['rouge-1']['f']
                    rouge_2_f1 = rouge_scores['rouge-2']['f']
                    rouge_l_f1 = rouge_scores['rouge-l']['f']
                else:
                    rouge_1_f1 = rouge_2_f1 = rouge_l_f1 = None

                meteor = calculate_meteor_score(predicted_answer, true_answer)
                precision, recall, f1 = score([predicted_answer], [true_answer], lang=lang, verbose=False)
                precision, recall, f1 = precision.mean().item(), recall.mean().item(), f1.mean().item()

                bleu_scores.append(bleu_score)
                if rouge_1_f1: rouge_1_f1_scores.append(rouge_1_f1)
                if rouge_2_f1: rouge_2_f1_scores.append(rouge_2_f1)
                if rouge_l_f1: rouge_l_f1_scores.append(rouge_l_f1)
                if meteor: meteor_scores.append(meteor)
                bert_precisions.append(precision)
                bert_recalls.append(recall)
                bert_f1s.append(f1)

                row.append(bleu_score)
                row.append(rouge_1_f1)
                row.append(rouge_2_f1)
                row.append(rouge_l_f1)
                row.append(meteor)
                row.append(precision)
                row.append(recall)
                row.append(f1)

                row[1] = wandb.Image(f"artifacts/run-{run_id}-prediction_table-{latest_version}/{row[1]['path']}")
                table.add_data(*row)

            # Calculate averages using nanmean
            average_bleu_score = np.nanmean(bleu_scores)
            average_rouge_1_f1 = np.nanmean(rouge_1_f1_scores) if rouge_1_f1_scores else 0
            average_rouge_2_f1 = np.nanmean(rouge_2_f1_scores) if rouge_2_f1_scores else 0
            average_rouge_l_f1 = np.nanmean(rouge_l_f1_scores) if rouge_l_f1_scores else 0
            average_meteor_score = np.nanmean(meteor_scores) if meteor_scores else 0
            average_bert_precision = np.nanmean(bert_precisions)
            average_bert_recall = np.nanmean(bert_recalls)
            average_bert_f1 = np.nanmean(bert_f1s)

            # Log the table and averages to wandb
            wandb.log({
                "prediction_table": table,
                "average_bleu_score": average_bleu_score,
                "average_rouge_1_f1": average_rouge_1_f1,
                "average_rouge_2_f1": average_rouge_2_f1,
                "average_rouge_l_f1": average_rouge_l_f1,
                "average_meteor_score": average_meteor_score,
                "average_bert_precision": average_bert_precision,
                "average_bert_recall": average_bert_recall,
                "average_bert_f1": average_bert_f1,
            })
            wandb.finish()

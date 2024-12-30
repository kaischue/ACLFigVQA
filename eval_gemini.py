import numpy as np
import pandas as pd
import wandb
from bert_score import score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

from constants import WANDB_API_KEY

if __name__ == '__main__':
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="VQA", name="GeminiVQA")

    validated_csv_path = 'validated_answers.csv'
    validated_df = pd.read_csv(validated_csv_path, encoding='utf-8')

    # drop rows of filtered out questions
    validated_df.drop(validated_df[(validated_df['corrected_answer_german'] == 'True') & (
            validated_df['corrected_answer_english'] == 'True')].index, inplace=True)

    # Fill NaN values with empty strings
    validated_df['answer_english'].fillna('NaN', inplace=True)
    validated_df['corrected_answer_english'].fillna('NaN', inplace=True)
    validated_df['answer_german'].fillna('NaN', inplace=True)
    validated_df['corrected_answer_german'].fillna('NaN', inplace=True)


    # Function to calculate BLEU score for each pair of answers
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


    def calculate_accuracy(candidate: str, reference: str, candidate_de: str, reference_de: str):
        if candidate.lower().strip() == reference.lower().strip():
            return 1
        elif candidate_de.lower().strip() == reference_de.lower().strip():
            return 1
        elif candidate.lower().strip() == reference_de.lower().strip():
            return 1
        elif candidate_de.lower().strip() == reference.lower().strip():
            return 1
        else:
            return 0


    def calculate_combined_bleu_score(candidate, reference, candidate_de, reference_de):
        scores = [
            calculate_bleu_score(candidate, reference),
            calculate_bleu_score(candidate_de, reference_de),
            calculate_bleu_score(candidate, reference_de),
            calculate_bleu_score(candidate_de, reference)
        ]
        return max(scores)


    def calculate_combined_rouge_scores(candidate, reference, candidate_de, reference_de):
        scores = [
            calculate_rouge_scores(candidate, reference),
            calculate_rouge_scores(candidate_de, reference_de),
            calculate_rouge_scores(candidate, reference_de),
            calculate_rouge_scores(candidate_de, reference)
        ]
        return max(scores,
                   key=lambda s: (s['rouge-1']['f'], s['rouge-2']['f'], s['rouge-l']['f']) if s is not None else (
                   0, 0, 0))


    def calculate_combined_meteor_score(candidate, reference, candidate_de, reference_de):
        scores = [
            calculate_meteor_score(candidate, reference),
            calculate_meteor_score(candidate_de, reference_de),
            calculate_meteor_score(candidate, reference_de),
            calculate_meteor_score(candidate_de, reference)
        ]
        return max(scores)


    # Apply the combined score calculations to each pair of answers
    validated_df['bleu_score'] = validated_df.apply(
        lambda row: calculate_combined_bleu_score(row['answer_english'], row['corrected_answer_english'],
                                                  row['answer_german'], row['corrected_answer_german']), axis=1)

    validated_df['rouge_scores'] = validated_df.apply(
        lambda row: calculate_combined_rouge_scores(row['answer_english'], row['corrected_answer_english'],
                                                    row['answer_german'], row['corrected_answer_german']), axis=1)

    validated_df['meteor_score'] = validated_df.apply(
        lambda row: calculate_combined_meteor_score(row['answer_english'], row['corrected_answer_english'],
                                                    row['answer_german'], row['corrected_answer_german']), axis=1)

    validated_df['accuracy'] = validated_df.apply(
        lambda row: calculate_accuracy(row['answer_english'], row['corrected_answer_english'], row['answer_german'],
                                       row["corrected_answer_german"]), axis=1)

    # Calculate BERT scores
    predictions = validated_df['answer_english'].tolist() + validated_df['answer_german'].tolist()
    references = validated_df['corrected_answer_english'].tolist() + validated_df['corrected_answer_german'].tolist()
    precision, recall, f1 = score(predictions, references, lang='en')

    validated_df['bert_precision'] = precision[:len(validated_df)]
    validated_df['bert_recall'] = recall[:len(validated_df)]
    validated_df['bert_f1'] = f1[:len(validated_df)]

    # Filter out None values
    valid_rouge_scores = [score for score in validated_df['rouge_scores'] if score is not None]
    valid_meteor_scores = [score for score in validated_df['meteor_score'] if score is not None]

    # Calculate average scores for metrics
    rouge_1 = [score['rouge-1']['f'] for score in valid_rouge_scores]
    rouge_2 = [score['rouge-2']['f'] for score in valid_rouge_scores]
    rouge_l = [score['rouge-l']['f'] for score in valid_rouge_scores]

    average_rouge_1 = sum(rouge_1) / len(rouge_1) if rouge_1 else 0
    average_rouge_2 = sum(rouge_2) / len(rouge_2) if rouge_2 else 0
    average_rouge_l = sum(rouge_l) / len(rouge_l) if rouge_l else 0

    average_bleu_score = validated_df['bleu_score'].mean()
    average_meteor_score = np.nansum(valid_meteor_scores) / len(valid_meteor_scores) if valid_meteor_scores else 0
    average_accuracy = validated_df['accuracy'].mean()
    average_bert_precision = precision[:len(validated_df)].mean()
    average_bert_recall = recall[:len(validated_df)].mean()
    average_bert_f1 = f1[:len(validated_df)].mean()

    print(f"Average ROUGE-1 F1 Score: {average_rouge_1:.2f}")
    print(f"Average ROUGE-2 F1 Score: {average_rouge_2:.2f}")
    print(f"Average ROUGE-L F1 Score: {average_rouge_l:.2f}")
    print(f"Average METEOR score: {average_meteor_score:.2f}")
    print(f"Average BLEU score: {average_bleu_score:.2f}")
    print(f"Average Accuracy: {average_accuracy:.2f}")
    print(f"Average BERT Precision: {average_bert_precision:.2f}")
    print(f"Average BERT Recall: {average_bert_recall:.2f}")
    print(f"Average BERT F1: {average_bert_f1:.2f}")

    table = wandb.Table(dataframe=validated_df)
    wandb.log({
        "validated_answers": table,
        "average_bleu_score": average_bleu_score,
        "average_rouge_1_f1": average_rouge_1,
        "average_rouge_2_f1": average_rouge_2,
        "average_rouge_l_f1": average_rouge_l,
        "average_meteor_score": average_meteor_score,
        "average_bert_precision": average_bert_precision,
        "average_bert_recall": average_bert_recall,
        "average_bert_f1": average_bert_f1,
        "GeminiAccuracy": average_accuracy
    })
    wandb.finish()
import csv
import os.path
from dataclasses import dataclass, fields
from pathlib import Path

import google.generativeai as genai
import json_repair
import pandas

from constants import GEMINI_API_KEY
from load_dataset_disc import get_dataset_split_generator
from utils import load_yaml_to_dict, shuffle_lists_in_dict, find_first_mention_of_figure, remove_single_line_breaks, \
    create_image_question_screenshot, convert_csv_rows_json

GEMINI_MODEL = "gemini-1.5-pro-latest"
# GEMINI_MODEL = "gemini-1.5-flash-latest"

VISUAL_LABELS = ["bar charts", "boxplots", "confusion matrix", "Line graph_chart",
                 "pareto", "pie chart", "scatter plot", "venn diagram"]
TEMPLATE_FILE = "Data\\vqa_templates_merged.yaml"
QA_GEN_CSV_FILE_REV1 = "Data\\qa_gen\\rev1\\qa_gen.csv"
QA_GEN_CSV_FILE_REV2 = "Data\\qa_gen\\rev2\\qa_gen.csv"
QA_GEN_CSV_FILE_REV3 = "Data\\qa_gen\\rev3\\qa_gen.csv"
QA_GEN_CSV_FILE_REV4 = "Data\\qa_gen\\rev4\\qa_gen.csv"
PROMPT_FILE = "Data\\vqa_gemini_prompt.txt"
FEEDBACK_FILE = "Data\\feedback_prompt.txt"
IMPROVEMENT_FILE = "Data\\improvement_prompt.txt"
JSON_FIX_PROMPT = "Data\\json_fix_prompt.txt"
JSON_FIX_PROMPT2 = "Data\\json_fix_prompt2.txt"
EVALUATE_PROMPT = "Data\\evaluate_prompt.txt"
ANSWER_CONFIDENCE_PROMPT = "Data\\confidence_prompt.txt"


@dataclass
class QAPairGen:
    question_german: str
    question_english: str
    answer_english: str
    answer_german: str


@dataclass
class QAPairGenConf:
    question_german: str
    question_english: str
    answer_english: str
    answer_german: str
    chain_of_thought: str
    chain_of_thought2: str
    confidence_explanation: str
    confidence: str


class GeminiModel:
    def __init__(self, gemini_model_type=GEMINI_MODEL, temperature=None, figure_mention_range=5000,
                 templates_file=TEMPLATE_FILE, prompt_file=PROMPT_FILE, feedback_file=FEEDBACK_FILE,
                 improvement_file=IMPROVEMENT_FILE):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(gemini_model_type)
        self.config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[QAPairGen], temperature=None)

        self.vqa_templates = load_yaml_to_dict(templates_file)

        with open(prompt_file, "r") as f:
            self.prompt = f.read()

        with open(feedback_file, "r") as f:
            self.feedback_prompt = f.read()

        with open(improvement_file, "r") as f:
            self.improvement_prompt = f.read()

        with open(JSON_FIX_PROMPT, "r") as f:
            self.json_fix_prompt = f.read()

        with open(JSON_FIX_PROMPT2, "r") as f:
            self.json_fix_prompt2 = f.read()

        with open(EVALUATE_PROMPT, "r") as f:
            self.evaluate_prompt = f.read()

        with open(ANSWER_CONFIDENCE_PROMPT, "r") as f:
            self.answer_confidence_prompt = f.read()

        self.figure_mention_range = figure_mention_range

    def translate_en_de(self, eng_text):
        return self.model.generate_content(
            ["Kannst du den text auf deutsch übersetzen?"
             "Übersetze so genau wie möglich, aber übersetze keine Namen! "
             "Antworte nur mit dem übersetzten text, sonst nichts!", eng_text])

    def translate_de_en(self, de_text):
        return self.model.generate_content(
            ["Can you translate this text into english?"
             "Translate as accurate as possible, dont translate names/! "
             "Only answer with the translated text, nothing else!", de_text])

    def get_metadata_str(self, sample: dict):
        img = sample['image']  # PILPng
        label = sample['label']
        caption = sample['caption']
        inline_reference = sample['inline_reference']
        metadata = sample['metadata']
        acl_paper_id = sample['acl_paper_id']
        pdf_text = remove_single_line_breaks(sample['pdf_text'])

        first_mention_idx = find_first_mention_of_figure(pdf_text, caption)

        relevant_pdf_str = pdf_text[
                           max([0, first_mention_idx - self.figure_mention_range]
                               ):min([first_mention_idx + self.figure_mention_range, len(pdf_text)])
                           ]

        metadata_string = f'Here is some additional info for the image:' \
                          f'The label of the image is: {label}.' \
                          f'The caption of the image is: {caption}.' \
                          f'The inline reference of the image is: {inline_reference}.' \
                          f'And the relevant section from the research paper that includes the figure caption. ' \
                          f'This section starts from {self.figure_mention_range} characters before and after the first ' \
                          f'mention of the figure caption:\n' \
                          f'{relevant_pdf_str}'
        return metadata_string

    def generate_qa(self, sample: dict, shuffle=False):
        img = sample['image']  # PILPng
        metadata_string = self.get_metadata_str(sample)

        if shuffle:
            shuffle_lists_in_dict(self.vqa_templates)

        template_str_dict = str(self.vqa_templates)

        return self.model.generate_content(
            [self.prompt, "QA-Templates:\n" + template_str_dict, img, metadata_string])

    def generate_feedback(self, sample: dict, qa_pairs: str):
        img = sample['image']  # PILPng
        metadata_string = self.get_metadata_str(sample)

        template_str_dict = str(self.vqa_templates)

        return self.model.generate_content(
            [self.feedback_prompt, "Original questions and answers:\n" + qa_pairs,
             "QA-Templates:\n" + template_str_dict, img, metadata_string])

    def generate_answer_confidence(self, sample: dict, questions: str):
        img = sample['image']  # PILPng
        metadata_string = self.get_metadata_str(sample)

        template_str_dict = str(self.vqa_templates)

        return self.model.generate_content(
            [self.answer_confidence_prompt, "Original Questions and Answers:\n" + questions, img, metadata_string])

    def generate_qa_revised(self, sample: dict, qa_pairs: str, feedback: str):
        img = sample['image']  # PILPng
        metadata_string = self.get_metadata_str(sample)

        template_str_dict = str(self.vqa_templates)

        return self.model.generate_content(
            [self.improvement_prompt, "Here is the feedback:\n" + feedback,
             "Original questions and answers:\n" + qa_pairs, "QA-Templates:\n" + template_str_dict,
             img, metadata_string])

    def evaluate_revised_original(self, sample: dict, qa_pairs_orig: str, qa_pairs_rev: str):
        img = sample['image']  # PILPng
        metadata_string = self.get_metadata_str(sample)

        template_str_dict = str(self.vqa_templates)

        return self.model.generate_content(
            [self.evaluate_prompt, "Original Questions and Answers:\n" + qa_pairs_orig,
             "Revised Questions and Answers:\n" + qa_pairs_rev, "QA-Templates:\n" + template_str_dict,
             img, metadata_string])

    def try_fix_json(self, broken_json, error_msg):
        return self.model.generate_content(
            [self.json_fix_prompt, str(broken_json), error_msg])

    def try_fix_json2(self, broken_json, error_msg):
        return self.model.generate_content(
            [self.json_fix_prompt2, "Here is the json, i tried to parse: \n" + str(broken_json),
             "This is the error message i got: \n" + error_msg])


def write_row_to_csv(csv_path, row, mode='a'):
    with open(csv_path, mode=mode, newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def process_qa_objects(json_response, fix_json, dataclass_type, qa_gen_csv_file):
    try:
        qa_objects = [dataclass_type(**qa) for qa in json_response]
    except Exception as e:
        print(str(e))
        print("Invalid JSON members, prompting for fix...")
        fix_response = fix_json(json_response, str(e))
        json_response = json_repair.loads(fix_response.text)
        qa_objects = [dataclass_type(**qa) for qa in json_response]

    for qa_object in qa_objects:
        attributes = [getattr(qa_object, field.name, None) for field in fields(qa_object)]
        write_row_to_csv(qa_gen_csv_file, [ds_sample["img_file_name"]] + attributes)


def fix_and_save_qa(gemini_response, fix_json, qa_gen_csv_file, dataclass_type):
    json_response = json_repair.loads(gemini_response.text, logging=True)
    print(gemini_response.usage_metadata)
    process_qa_objects(json_response, fix_json, dataclass_type, qa_gen_csv_file)


def get_rows_and_save_img_qa(sample, qa_gen_csv_file, out_path, regen_img=False):
    df_qa_gen_train = pandas.read_csv(qa_gen_csv_file)

    rows = df_qa_gen_train.loc[df_qa_gen_train['img_file_name'] == sample["img_file_name"]]
    questions_eng = rows.question_english.values
    answers_eng = rows.answer_english.values
    questions_de = rows.question_german.values
    answers_de = rows.answer_german.values

    questions = [f"{eng} | {de}" for eng, de in zip(questions_eng, questions_de)]
    answers = [f"{eng} | {de}" for eng, de in zip(answers_eng, answers_de)]

    img_path = f"Data\\VQAMeta\\training_data\\train\\{sample['label']}\\{sample['img_file_name']}"
    img_qa = f"{out_path}\\{Path(img_path).stem}_qa.png"

    if (not os.path.isfile(img_qa)) or regen_img:
        create_image_question_screenshot(img_path, questions, answers, out_path=out_path)

    return convert_csv_rows_json(rows)


if __name__ == '__main__':
    model = GeminiModel()
    ds_iterator = iter(get_dataset_split_generator(split="train"))

    if not os.path.isfile(QA_GEN_CSV_FILE_REV1):
        write_row_to_csv(QA_GEN_CSV_FILE_REV1,
                         ["img_file_name", "question_german", "question_english", "answer_german", "answer_english"],
                         mode='w')
    if not os.path.isfile(QA_GEN_CSV_FILE_REV2):
        write_row_to_csv(QA_GEN_CSV_FILE_REV2,
                         ["img_file_name", "question_german", "question_english", "answer_german", "answer_english"],
                         mode='w')
    if not os.path.isfile(QA_GEN_CSV_FILE_REV3):
        write_row_to_csv(QA_GEN_CSV_FILE_REV3,
                         ["img_file_name", "question_german", "question_english", "answer_german", "answer_english"],
                         mode='w')
    if not os.path.isfile(QA_GEN_CSV_FILE_REV4):
        write_row_to_csv(QA_GEN_CSV_FILE_REV4,
                         [
                             "img_file_name",
                             "question_german", "question_english",
                             "answer_german", "answer_english",
                             "chain_of_thought", "chain_of_thought2",
                             "confidence_explanation", "confidence"
                         ],
                         mode='w')

    df_qa_gen_train_rev1 = pandas.read_csv(QA_GEN_CSV_FILE_REV1)

    try:
        while True:
            ds_sample = next(ds_iterator)
            if ds_sample["label"] not in VISUAL_LABELS:
                continue

            # generate qa
            if df_qa_gen_train_rev1.loc[df_qa_gen_train_rev1['img_file_name'] == ds_sample["img_file_name"]].empty:
                response = model.generate_qa(ds_sample, shuffle=True)
                fix_and_save_qa(response, model.try_fix_json, QA_GEN_CSV_FILE_REV1, QAPairGen)

            qa_pairs_str = get_rows_and_save_img_qa(ds_sample, QA_GEN_CSV_FILE_REV1, out_path="Data\\qa_gen\\rev1")

            # generate feedback
            feedback_file_f = f"Data\\qa_gen\\rev1\\{ds_sample['img_file_name']}_feedback.txt"
            if not os.path.isfile(feedback_file_f):
                s_feedback = model.generate_feedback(ds_sample, qa_pairs_str)
                with open(feedback_file_f, 'w', encoding="utf-8") as file:
                    file.write(s_feedback.text)

            # generate qa revised
            df_qa_gen_train_rev2 = pandas.read_csv(QA_GEN_CSV_FILE_REV2)
            if df_qa_gen_train_rev2.loc[df_qa_gen_train_rev2['img_file_name'] == ds_sample["img_file_name"]].empty:
                with open(feedback_file_f, 'r', encoding="utf-8") as file:
                    s_feedback = file.read()
                response_improved = model.generate_qa_revised(ds_sample, qa_pairs_str, s_feedback)
                fix_and_save_qa(response_improved, model.try_fix_json, QA_GEN_CSV_FILE_REV2, QAPairGen)

            qa_pairs_str_rev2 = get_rows_and_save_img_qa(ds_sample, QA_GEN_CSV_FILE_REV2,
                                                         out_path="Data\\qa_gen\\rev2")

            # generate qa evaluated
            df_qa_gen_train_rev3 = pandas.read_csv(QA_GEN_CSV_FILE_REV3)
            if df_qa_gen_train_rev3.loc[df_qa_gen_train_rev3['img_file_name'] == ds_sample["img_file_name"]].empty:
                response_evaluated = model.evaluate_revised_original(ds_sample, qa_pairs_str, qa_pairs_str_rev2)
                fix_and_save_qa(response_evaluated, model.try_fix_json, QA_GEN_CSV_FILE_REV3, QAPairGen)

            qa_pairs_str_rev3 = get_rows_and_save_img_qa(ds_sample, QA_GEN_CSV_FILE_REV3,
                                                         out_path="Data\\qa_gen\\rev3")

            # generate answer confidence
            df_qa_gen_train_rev4 = pandas.read_csv(QA_GEN_CSV_FILE_REV4)
            if df_qa_gen_train_rev4.loc[df_qa_gen_train_rev4['img_file_name'] == ds_sample["img_file_name"]].empty:
                response_confidence = model.generate_answer_confidence(ds_sample, qa_pairs_str_rev3)
                fix_and_save_qa(response_confidence, model.try_fix_json2, QA_GEN_CSV_FILE_REV4, QAPairGenConf)
            qa_pairs_str_rev4 = get_rows_and_save_img_qa(ds_sample, QA_GEN_CSV_FILE_REV4, out_path="Data\\qa_gen\\rev4")

            print("Done with img!")

    except StopIteration:
        pass

import csv
import json
import os.path
from dataclasses import dataclass

import google.generativeai as genai
import pandas

from constants import GEMINI_API_KEY
from load_dataset_disc import get_dataset_split_generator
from utils import load_yaml_to_dict, shuffle_lists_in_dict, find_first_mention_of_figure, remove_single_line_breaks, \
    create_image_question_screenshot

GEMINI_MODEL = "gemini-1.5-pro-latest"
# GEMINI_MODEL = "gemini-1.5-flash-latest"

VISUAL_LABELS = ["bar charts", "boxplots", "confusion matrix", "Line graph_chart",
                 "pareto", "pie chart", "scatter plot", "venn diagram"]
TEMPLATE_FILE = "Data\\vqa_templates_merged.yaml"
QA_GEN_CSV_FILE = "Data\\VQAMeta\\test.csv"
PROMPT_FILE = "Data\\vqa_gemini_prompt.txt"


class GeminiModel:
    def __init__(self, gemini_model_type=GEMINI_MODEL, temperature=None,
                 templates_file=TEMPLATE_FILE, prompt_file=PROMPT_FILE):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(gemini_model_type)
        self.config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[QAPairGen], temperature=None)

        self.vqa_templates = load_yaml_to_dict(templates_file)

        with open(prompt_file, "r") as f:
            self.prompt = f.read()

    def generate_qa(self, sample: dict, shuffle=False, figure_mention_range=5000):
        img = sample['image']  # PILPng
        label = sample['label']
        caption = sample['caption']
        inline_reference = sample['inline_reference']
        metadata = sample['metadata']
        acl_paper_id = sample['acl_paper_id']
        pdf_text = remove_single_line_breaks(sample['pdf_text'])

        first_mention_idx = find_first_mention_of_figure(pdf_text, caption)

        metadata_string = f'Here is some additional info for the image:' \
                          f'The label of the image is: {label}.' \
                          f'The caption of the image is: {caption}.' \
                          f'The inline reference of the image is: {inline_reference}.' \
                          f'And the relevant section from the research paper that includes the figure caption. ' \
                          f'This section starts from {figure_mention_range} characters before and after the first ' \
                          f'mention of the figure caption:\n' \
                          f'{pdf_text[max([0, first_mention_idx - figure_mention_range]):min([first_mention_idx + figure_mention_range, len(pdf_text)])]}'

        if shuffle:
            shuffle_lists_in_dict(self.vqa_templates)

        template_str_dict = str(self.vqa_templates)

        return self.model.generate_content(
            [self.prompt, template_str_dict, img, metadata_string])

    def try_fix_json(self, broken_json, error_msg):
        return self.model.generate_content(
            [f"Can you please fix this broken JSON? I get this error msg:{error_msg}. Only reply with the fixed JSON "
             f"string, nothing else! So that i can directly parse it!", broken_json])


@dataclass
class QAPairGen:
    question_german: str
    question_english: str
    answer_english: str
    answer_german: str


def write_row_to_csv(csv_path, row, mode='a'):
    with open(csv_path, mode=mode, newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == '__main__':
    model = GeminiModel()
    ds_iterator = iter(get_dataset_split_generator(split="train"))

    if not os.path.isfile(QA_GEN_CSV_FILE):
        write_row_to_csv(QA_GEN_CSV_FILE,
                         ["img_file_name", "question_german", "question_english", "answer_german", "answer_english"],
                         mode='w')

    df_qa_gen_train = pandas.read_csv(QA_GEN_CSV_FILE)

    try:
        while True:

            ds_sample = next(ds_iterator)
            if ds_sample["label"] not in VISUAL_LABELS:
                continue
            if not df_qa_gen_train.loc[df_qa_gen_train['img_file_name'] == ds_sample["img_file_name"]].empty:
                rows = df_qa_gen_train.loc[df_qa_gen_train['img_file_name'] == ds_sample["img_file_name"]]
                questions = rows.question_english
                answers = rows.answer_english
                create_image_question_screenshot(f"Data\\VQAMeta\\training_data\\train\\{ds_sample['label']}\\{ds_sample['img_file_name']}", questions.values, answers.values)
                continue
            # plt.imshow(ds_sample["image"])
            # plt.show()
            response = model.generate_qa(ds_sample, shuffle=True)

            try:
                json_response = json.loads(response.text)
            except json.JSONDecodeError as e:
                print("Invalid JSON, prompting for fix...")
                e_msg = str(e)
                invalid_json = e.doc
                fix_response = model.try_fix_json(invalid_json, e_msg)
                json_response = json.loads(fix_response.text)

            print(response.usage_metadata)
            for qa in json_response:
                qa_object = QAPairGen(**qa)
                write_row_to_csv(QA_GEN_CSV_FILE, [
                    ds_sample["img_file_name"],
                    qa_object.question_german,
                    qa_object.question_english,
                    qa_object.answer_german,
                    qa_object.answer_english])
            # pprint(json_response)
    except StopIteration:
        pass

import json
from pprint import pprint

import google.generativeai as genai
import matplotlib.pyplot as plt
import typing_extensions as typing

from constants import GEMINI_API_KEY
import PIL.Image
from load_dataset_disc import get_dataset_split_generator

GEMINI_MODEL = "gemini-1.5-pro-exp-0827"


# GEMINI_MODEL = "gemini-1.5-flash-exp-0827"


class GeminiModel:
    def __init__(self, gemini_model_type=GEMINI_MODEL, temperature=None,
                 templates_file="Data\\vqa_templates.yaml", prompt_file="Data\\vqa_gemini_prompt.txt"):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(gemini_model_type)
        self.config = genai.GenerationConfig(
            response_mime_type="application/json", response_schema=list[QAPairGen], temperature=None)

        with open(templates_file, "r") as f:
            self.vqa_templates = f.read()

        with open(prompt_file, "r") as f:
            self.prompt = f.read()

    def generate_qa(self, sample: dict):
        img = sample['image']  # PILPng
        label = sample['label']
        caption = sample['caption']
        inline_reference = sample['inline_reference']
        metadata = sample['metadata']
        acl_paper_id = sample['acl_paper_id']
        pdf_text = sample['pdf_text']

        metadata_string = f'Here is some additional info for the image:' \
                          f'The label of the image is: {label}.' \
                          f'The caption of the image is: {caption}.' \
                          f'The inline reference of the image is: {inline_reference}.' \
                          f'And the research paper as text where the image was sourced form is: {pdf_text[:10000]}'

        return self.model.generate_content(
            [self.prompt, self.vqa_templates, img, metadata_string])


class QAPairGen(typing.TypedDict):
    question_german: str
    question_english: str
    answer_english: str
    answer_german: str


if __name__ == '__main__':
    model = GeminiModel()
    ds_iterator = iter(get_dataset_split_generator(split="train"))
    ds_sample = next(ds_iterator)

    plt.imshow(ds_sample["image"])
    plt.show()
    response = model.generate_qa(ds_sample)

    json_response = json.loads(response.text)

    print(response.usage_metadata)
    pprint(json_response)

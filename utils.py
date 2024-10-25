import re
import textwrap
from pathlib import Path
from random import shuffle

import yaml
from PIL import Image, ImageDraw, ImageFont


def load_yaml_to_dict(file_path):
    with open(file_path, "r") as f:
        d = yaml.load(f, Loader=yaml.SafeLoader)
    return d


def shuffle_lists_in_dict(yaml_dict):
    for key in yaml_dict.keys():
        for k_key in yaml_dict[key]:
            shuffle(yaml_dict[key][k_key])


def create_image_question_screenshot(img_path, questions, answers, out_path):
    out_name = f"{out_path}\\{Path(img_path).stem}_qa.png"
    # Open your chart image
    chart = Image.open(img_path)

    # Initial dimensions
    width, height = chart.size
    new_img_height = height

    # Drawing questions and answers on the right side
    draw = ImageDraw.Draw(chart)
    font = ImageFont.truetype("arial.ttf", 15)  # Using a font that supports Unicode

    # Calculate the required height for all text
    x_text = width + 20
    y_text = 20
    total_text_height = 0
    text_wrap_width = int((width - 5) // font.getlength("a"))

    for question, answer in zip(questions, answers):
        q_lines = textwrap.wrap(f"Q: {question}", width=text_wrap_width)
        a_lines = textwrap.wrap(f"A: {answer}", width=text_wrap_width)
        total_text_height += (len(q_lines) + len(a_lines)) * 20 + 40  # Adjust space between lines and questions

    if total_text_height > height:
        new_img_height = total_text_height + 40  # Adding extra space

    # Create a new image with the required height
    new_img = Image.new('RGB', (width * 2, new_img_height), 'white')
    new_img.paste(chart, (0, 0))

    draw = ImageDraw.Draw(new_img)

    # Drawing the questions and answers again
    y_text = 20
    for question, answer in zip(questions, answers):
        q_lines = textwrap.wrap(f"Q: {question}", width=text_wrap_width)
        a_lines = textwrap.wrap(f"A: {answer}", width=text_wrap_width)
        for line in q_lines + a_lines:
            draw.text((x_text, y_text), line, fill="black", font=font)
            y_text += 20
        y_text += 20  # Extra space between questions

    # Save the new image
    new_img.save(out_name)


def remove_single_line_breaks(text):
    # Replace single line breaks with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text


def replace_all_linebreaks_with_spaces(text):
    return ' '.join(text.splitlines())


def find_first_number_end_index(text):
    # Find the first number in the text
    match = re.search(r'\d+', text)
    if match:
        # Return the end index of the first number
        return match.end()
    return -1  # Return -1 if no number is found


def find_first_mention_of_figure(text, caption):
    pdf_text_no_lb = text.lower()
    figure_i_index = find_first_number_end_index(caption)
    return pdf_text_no_lb.find(caption[:figure_i_index].lower())


def convert_csv_rows_json(rows):
    questions_eng = rows.question_english.values
    answers_eng = rows.answer_english.values
    questions_de = rows.question_german.values
    answers_de = rows.answer_german.values
    qa_pairs = []

    for q_de, q_eng, a_de, a_eng in zip(questions_de, questions_eng, answers_de, answers_eng):
        qa_pair = {
            "question_german": q_de,
            "question_english": q_eng,
            "answer_english": a_eng,
            "answer_german": a_de
        }
        qa_pairs.append(qa_pair)

    return str(qa_pairs)


if __name__ == '__main__':
    qa_dict = load_yaml_to_dict("Data\\test.yaml")
    shuffle_lists_in_dict(qa_dict)

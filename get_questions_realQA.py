import os
import json
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(file_path):
    questions_by_qid_ = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                qid = item.get('QID')
                question = item.get('question')
                if qid and question and qid not in questions_by_qid_:
                    questions_by_qid_[qid] = question
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return questions_by_qid_


def extract_questions_by_qid(folder_paths_):
    questions_by_qid_ = {}
    json_files = []
    for folder_path in folder_paths_:
        for root, _, files in os.walk(folder_path):
            json_files.extend([os.path.join(root, file) for file in files if file.endswith('.json')])

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            for qid, question in result.items():
                if qid not in questions_by_qid_:
                    questions_by_qid_[qid] = question

    return questions_by_qid_


if __name__ == '__main__':
    folder_paths = ['D:\\MA\\RealCQA\\qa', 'D:\\MA\\RealCQA\\qa2', 'D:\\MA\\RealCQA\\qa3']
    questions_by_qid = extract_questions_by_qid(folder_paths)

    with open('Data\\RealQA_questions.yaml', 'w') as yaml_file:
        yaml.dump(questions_by_qid, yaml_file, default_flow_style=False)

    print("All questions processed!")

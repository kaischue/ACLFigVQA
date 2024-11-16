import csv
import os
import tkinter as tk
import webbrowser

import pandas as pd
from PIL import Image, ImageTk, ImageDraw

from gemini_api import GeminiModel
from load_dataset_disc import get_dataset_split_generator

# Load CSV
df = pd.read_csv('Data\\qa_gen\\rev4\\qa_gen.csv', encoding='utf-8')

# Check if validated_answers.csv exists, initialize if it doesn't
validated_csv_path = 'validated_answers.csv'
if os.path.exists(validated_csv_path):
    validated_df = pd.read_csv(validated_csv_path, encoding='utf-8')
else:
    columns = df.columns.tolist() + ['corrected_answer_german', 'corrected_answer_english', 'flagged']
    validated_df = pd.DataFrame(columns=columns)
    validated_df.to_csv(validated_csv_path, index=False, encoding='utf-8')


class App:
    def __init__(self, root):
        self.current_split = "train"
        self.geminiModel = GeminiModel()
        self.root = root
        self.dataset = list(get_dataset_split_generator(only_visual=True, split="train"))
        self.dataset_train_length = len(self.dataset)
        self.dataset += list(get_dataset_split_generator(only_visual=True, split="validation"))
        self.image_index = 0
        self.question_index = 0
        self.unique_images = df['img_file_name'].unique()
        self.current_acl_paper_id = None
        self.validated_questions = validated_df[['img_file_name', 'question_german', 'question_english']].to_dict(
            'records')
        self.grid_size = 20  # initial grid size
        self.setup_ui()
        self.load_image_and_question()

    def setup_ui(self):
        self.canvas = tk.Canvas(root)
        self.canvas.pack(side="left", padx=10, pady=5, fill="both", expand=True)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.config(yscrollcommand=self.scrollbar.set)

        self.metadata_label = tk.Label(root, text="", wraplength=400)
        self.metadata_label.pack(pady=(10, 0))

        self.question_frame_german = tk.Frame(root)
        self.question_frame_german.pack(pady=(10, 0))

        self.question_label_german = tk.Label(self.question_frame_german, text="", wraplength=400)
        self.question_label_german.pack(side=tk.LEFT)

        self.copy_button_german = tk.Button(self.question_frame_german, text="Copy", command=self.copy_german)
        self.copy_button_german.pack(side=tk.RIGHT)

        self.answer_entry_german = tk.Text(root, width=50, height=5)
        self.answer_entry_german.pack(pady=(5, 0))

        self.question_frame_english = tk.Frame(root)
        self.question_frame_english.pack(pady=(10, 0))

        self.question_label_english = tk.Label(self.question_frame_english, text="", wraplength=400)
        self.question_label_english.pack(side=tk.LEFT)

        self.copy_button_english = tk.Button(self.question_frame_english, text="Copy", command=self.copy_english)
        self.copy_button_english.pack(side=tk.RIGHT)

        self.answer_entry_english = tk.Text(root, width=50, height=5)
        self.answer_entry_english.pack(pady=(5, 0))

        self.chain_of_thought_label = tk.Label(root, text="", wraplength=400)
        self.chain_of_thought_label.pack(pady=(10, 0))

        self.flag_button = tk.Button(root, text="Flag as Needs Improvement", command=self.flag_question)
        self.flag_button.pack(pady=(10, 0), side=tk.TOP)

        self.validate_next_button = tk.Button(root, text="Validate & Next", command=self.validate_and_next_question)
        self.validate_next_button.pack(side=tk.LEFT)

        self.skip_button = tk.Button(root, text="Skip", command=self.skip_question)
        self.skip_button.pack(side=tk.RIGHT)

        self.translate_en_to_de_button = tk.Button(root, text="Translate EN to DE", command=self.translate_en_to_de)
        self.translate_en_to_de_button.pack(side=tk.LEFT)

        self.translate_de_to_en_button = tk.Button(root, text="Translate DE to EN", command=self.translate_de_to_en)
        self.translate_de_to_en_button.pack(side=tk.RIGHT)

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_grid)
        self.grid_offset_x = 0
        self.grid_offset_y = 0

    def copy_german(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.question_label_german.cget("text"))

    def copy_english(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.question_label_english.cget("text"))

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag_grid(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.grid_offset_x += dx
        self.grid_offset_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.load_image_and_question()

    def draw_grid(self, image):
        draw = ImageDraw.Draw(image)
        for i in range(self.grid_offset_x, image.width, self.grid_size):
            draw.line((i, 0, i, image.height), fill="grey")
        for i in range(self.grid_offset_y, image.height, self.grid_size):
            draw.line((0, i, image.width, i), fill="grey")
        return image

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.grid_size += 5
        else:
            self.grid_size = max(5, self.grid_size - 5)
        self.load_image_and_question()

    def flag_question(self):
        row = self.current_questions.iloc[self.question_index].copy()  # Create a copy of the row
        row['flagged'] = True  # Mark the question as flagged

        # Save the flagged row to the CSV
        with open('validated_answers.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(list(row) + [''] * (len(df.columns) - len(row)) + [row['flagged']])

        self.next_question()

    def load_image_and_question(self):
        self.img_file_name = self.unique_images[self.image_index]
        self.current_questions = df[df['img_file_name'] == self.img_file_name]

        while self.question_index < len(self.current_questions):
            row = self.current_questions.iloc[self.question_index]
            if not any((row['img_file_name'] == v['img_file_name'] and row['question_german'] == v[
                'question_german'] and row['question_english'] == v['question_english'] and not v.get('flagged')) for v
                       in self.validated_questions):
                break
            self.question_index += 1

        if self.question_index >= len(self.current_questions):
            self.question_index = 0
            self.image_index += 1
            if self.image_index < len(self.unique_images):
                self.load_image_and_question()
            else:
                self.root.quit()
            return

        if self.image_index >= self.dataset_train_length:
            self.current_split = "val"
        # Fetch corresponding dataset sample
        metadata = self.dataset[self.image_index]
        image_path = f"Data\\VQAMeta\\training_data\\{self.current_split}\\{metadata['label']}\\{metadata['img_file_name']}"

        if self.current_acl_paper_id != metadata['acl_paper_id']:
            self.current_acl_paper_id = metadata['acl_paper_id']
            pdf_path = f"Data\\VQAMeta\\papers\\pdfs\\{metadata['acl_paper_id']}.pdf"
            webbrowser.open(pdf_path)

        image = Image.open(image_path)
        image_with_grid = self.draw_grid(image.copy())
        self.img = ImageTk.PhotoImage(image_with_grid)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))

        self.metadata_label.config(text=f"Metadata: {metadata['caption']}")
        self.question_label_german.config(text=f"German Question: {row['question_german']}")
        self.question_label_german.config(text=row['question_german'])
        self.answer_entry_german.delete('1.0', tk.END)
        self.answer_entry_german.insert('1.0', row['answer_german'])
        self.question_label_english.config(text=f"English Question: {row['question_english']}")
        self.question_label_english.config(text=row['question_english'])
        self.answer_entry_english.delete('1.0', tk.END)
        self.answer_entry_english.insert('1.0', row['answer_english'])
        self.chain_of_thought_label.config(text=f"Chain of Thought: {row['chain_of_thought']}")

    def translate_en_to_de(self):
        english_text = self.answer_entry_english.get('1.0', tk.END).strip()
        translated_text = self.geminiModel.translate_en_de(english_text).text
        self.answer_entry_german.delete('1.0', tk.END)
        self.answer_entry_german.insert('1.0', translated_text)

    def translate_de_to_en(self):
        german_text = self.answer_entry_german.get('1.0', tk.END).strip()
        translated_text = self.geminiModel.translate_de_en(german_text).text
        self.answer_entry_english.delete('1.0', tk.END)
        self.answer_entry_english.insert('1.0', translated_text)

    def validate_and_next_question(self):
        row = self.current_questions.iloc[self.question_index]
        corrected_answer_german = self.answer_entry_german.get('1.0', tk.END).strip()
        corrected_answer_english = self.answer_entry_english.get('1.0', tk.END).strip()

        # Append to CSV directly with UTF-8 encoding
        with open('validated_answers.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(list(row) + [corrected_answer_german, corrected_answer_english])

        self.validated_questions.append({
            'img_file_name': row['img_file_name'],
            'question_german': row['question_german'],
            'question_english': row['question_english']
        })

        self.next_question()

    def skip_question(self):
        self.next_question()

    def next_question(self):
        self.question_index += 1
        if self.question_index >= len(self.current_questions):
            self.question_index = 0
            self.image_index += 1

        if self.image_index < len(self.unique_images):
            self.load_image_and_question()
        else:
            self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

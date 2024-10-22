import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import pandas as pd
import csv
import os

from load_dataset_disc import get_dataset_split_generator

# Load CSV
df = pd.read_csv('Data\\qa_gen_test.csv', encoding='utf-8')

# Load the validated CSV if it exists
validated_df = pd.read_csv('validated_answers.csv', encoding='utf-8') if os.path.exists(
    'validated_answers.csv') else pd.DataFrame()

class App:
    def __init__(self, root, dataset_generator):
        self.root = root
        self.dataset_generator = dataset_generator
        self.dataset = list(self.dataset_generator)
        self.image_index = 0
        self.question_index = 0
        self.unique_images = df['img_file_name'].unique()
        self.validated_questions = validated_df[['img_file_name', 'question_german', 'question_english']].to_dict(
            'records')
        self.grid_size = 20  # initial grid size
        self.setup_ui()
        self.load_image_and_question()

    def setup_ui(self):
        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.metadata_label = tk.Label(root, text="", wraplength=400)
        self.metadata_label.pack()

        self.question_label_german = tk.Label(root, text="", wraplength=400)
        self.question_label_german.pack()

        self.answer_entry_german = tk.Text(root, width=50, height=5)
        self.answer_entry_german.pack()

        self.question_label_english = tk.Label(root, text="", wraplength=400)
        self.question_label_english.pack()

        self.answer_entry_english = tk.Text(root, width=50, height=5)
        self.answer_entry_english.pack()

        self.validate_next_button = tk.Button(root, text="Validate & Next", command=self.validate_and_next_question)
        self.validate_next_button.pack(side=tk.LEFT)

        self.skip_button = tk.Button(root, text="Skip", command=self.skip_question)
        self.skip_button.pack(side=tk.RIGHT)

        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

    def draw_grid(self, image):
        draw = ImageDraw.Draw(image)
        for i in range(0, image.width, self.grid_size):
            draw.line((i, 0, i, image.height), fill="black")
        for i in range(0, image.height, self.grid_size):
            draw.line((0, i, image.width, i), fill="black")
        return image

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.grid_size += 5
        else:
            self.grid_size = max(5, self.grid_size - 5)
        self.load_image_and_question()

    def load_image_and_question(self):
        self.img_file_name = self.unique_images[self.image_index]
        self.current_questions = df[df['img_file_name'] == self.img_file_name]

        while self.question_index < len(self.current_questions):
            row = self.current_questions.iloc[self.question_index]
            if not any((row['img_file_name'] == v['img_file_name'] and row['question_german'] == v[
                'question_german'] and row['question_english'] == v['question_english']) for v in
                       self.validated_questions):
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

        # Fetch corresponding dataset sample
        metadata = self.dataset[self.image_index]
        image_path = f"Data\\VQAMeta\\training_data\\train\\{metadata['label']}\\{metadata['img_file_name']}"

        image = Image.open(image_path)
        image_with_grid = self.draw_grid(image.copy())
        self.img = ImageTk.PhotoImage(image_with_grid)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

        self.metadata_label.config(text=f"Metadata: {metadata['caption']}")
        self.question_label_german.config(text=f"German Question: {row['question_german']}")
        self.answer_entry_german.delete('1.0', tk.END)
        self.answer_entry_german.insert('1.0', row['answer_german'])
        self.question_label_english.config(text=f"English Question: {row['question_english']}")
        self.answer_entry_english.delete('1.0', tk.END)
        self.answer_entry_english.insert('1.0', row['answer_english'])

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
    dataset_generator = get_dataset_split_generator()
    root = tk.Tk()
    app = App(root, dataset_generator)
    root.mainloop()

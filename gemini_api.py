import google.generativeai as genai
from constants import GEMINI_API_KEY
import PIL.Image

GEMINI_MODEL = "gemini-1.5-pro-exp-0827"
#GEMINI_MODEL = "gemini-1.5-flash-exp-0827"

if __name__ == '__main__':
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    generation_config = genai.GenerationConfig(temperature=2)

    f = open("Data\\vqa_templates.yaml", "r")
    vqa_templates = f.read()
    #print(vqa_templates)
    sample_file1 = PIL.Image.open("D:\\MA\\ACLFig\\train\\image_0.jpg")
    prompt = "Create a question about this figure. It should be a visual question, " \
             "meaning mention at least on visual attribute like color, position or orientation of the different plot elements." \
             " You should only be able to answer it if you look at the image, not just with extracted data as a table. But the question should not be to simple" \
             "Also please provide the question in English and German. And then answer the question. Chose three random question templates as a base from the templates i provided to you. " \
             "Not all templates fit all images. So choose on that fits and fill in the variables. Also alaways provide the template you used. Your questions do not have to match the template exactly. You can use it as a base"
    response = model.generate_content([prompt, vqa_templates, sample_file1])

    # response = model.generate_content("Write a story about a magic backpack.")
    print(response.text)

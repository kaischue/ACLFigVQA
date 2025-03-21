import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration

import wandb
from constants import WANDB_API_KEY
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.model.builder import load_pretrained_model
from utils import replace_all_linebreaks_with_spaces

global model_id, model, processor, tokenizer, context_len

use_image_path = False


def run_paligemma(image, question, metadata_str, lang):
    if len(metadata_str) > 0:
        prompt = f"<image> answer {lang} {metadata_str} {question}"
    else:
        prompt = f"<image> answer {lang} {question}"

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded


def run_chartgemma(image, question, metadata_str, lang):
    if len(metadata_str) > 0:
        prompt = f"<image> answer {lang} {metadata_str} {question}"
    else:
        prompt = f"<image> answer {lang} {question}"

    # Process Inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    prompt_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
    output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
    return output_text


def run_llava(image, question, metadata_str, lang):
    if len(metadata_str) > 0:
        prompt = f"{metadata_str} {question}"
    else:
        prompt = f"{question}"

    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=512)

    output_text = processor.decode(output[0], skip_special_tokens=True)
    return output_text.split("ASSISTANT: ")[-1]


def run_blip(image, question, metadata_str, lang):
    if len(metadata_str) > 0:
        prompt = f"Question: {metadata_str} {question} Answer:"
    else:
        prompt = f"Question: {question} Answer:"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    prediction = generated_text.split("Answer:")[1]

    return prediction


def run_tinychart(image, question, metadata_str, lang):
    if len(metadata_str) > 0:
        prompt = f"{metadata_str} {question}"
    else:
        prompt = f"{question}"

    response = inference_model([image], prompt, model, tokenizer, processor, context_len, conv_mode="phi",
                               max_new_tokens=512)

    return response


def run_deepseek_janus(image, question, metadata_str, lang):
    if len(metadata_str) > 0:
        prompt = f"{metadata_str} {question}"
    else:
        prompt = f"{question}"
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(model.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    # print(f"{prepare_inputs['sft_format'][0]}", answer)
    return answer


def predict_dataset(split, lang, metadata, model_name, predict_func):
    if not metadata:
        run_name = f"{model_name}-{split}-{lang}-no-metadata"
    else:
        run_name = f"{model_name}-{split}-{lang}-context"

    wandb.init(project="VQA", name=run_name)

    table = wandb.Table(
        columns=["question", "image", "true_answer", "true_short_answer", "predicted_answer", "category"])

    # Process the dataset and make predictions
    for example in tqdm(merged_dataset[split], desc="Predicting questions"):
        label = example["label"]
        context = example["context"]
        caption = example["caption"]
        image = example['image'].convert('RGB')
        image_path = f"Data\\VQAMeta\\training_data\\{split}\\{label}\\{example['img_file_name']}"
        category = example['category']
        if lang == "en":
            true_answer = example['corrected_answer_english']
            true_short_answer = example['short_answer_english']
            question = example['question_english']
        elif lang == "de":
            true_answer = example['corrected_answer_german']
            true_short_answer = example['short_answer_german']
            question = example['question_german']
        else:
            return

        if not metadata and len(context) > 5:
            continue
        if metadata and (len(context) < 5 and 'caption' not in category.lower()):
            continue

        if metadata:
            metadata_string = f'The caption of the image is: {caption}.'

            if len(context) > 5:
                metadata_string += f'And the relevant section from the research paper: {context}.'

            metadata_string = replace_all_linebreaks_with_spaces(metadata_string)

            if not use_image_path:
                prediction = predict_func(image, question, metadata_string, lang)
            else:
                prediction = predict_func(image_path, question, metadata_string, lang)
        else:
            if not use_image_path:
                prediction = predict_func(image, question, "", lang)
            else:
                prediction = predict_func(image_path, question, "", lang)

        print(question)
        print(f"prediction: {prediction} | true short_answer: {true_short_answer}")

        table.add_data(question, wandb.Image(image, caption=caption), true_answer, true_short_answer, prediction,
                       category)

    wandb.log({"prediction_table": table})


def init_paligemma(idd):
    global model_id
    model_id = idd

    global model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    global processor
    processor = PaliGemmaProcessor.from_pretrained(model_id)


def init_llava(idd):
    global model_id
    model_id = idd

    global processor
    processor = LlavaNextProcessor.from_pretrained(model_id)

    global model
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, use_flash_attention_2=True
    )


def init_blip(idd):
    global model_id
    model_id = idd

    global processor
    processor = Blip2Processor.from_pretrained(idd)

    global model
    model = Blip2ForConditionalGeneration.from_pretrained(
        idd, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )


def init_tinychart(idd):
    global model_id
    model_id = idd

    global model, processor, tokenizer, context_len
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_id,
        model_base=None,
        model_name=get_model_name_from_path(model_id),
        device="cuda"
    )


def init_deepseek_janus(idd):
    global use_image_path
    use_image_path = True

    global model_id
    model_id = idd

    global tokenizer, model, processor
    processor = VLChatProcessor.from_pretrained(idd)  # type: VLChatProcessor
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(idd, trust_remote_code=True)  # type: MultiModalityCausalLM

    model = model.to(torch.bfloat16).cuda().eval()


if __name__ == '__main__':
    # Initialize Weights and Biases
    wandb.login(key=WANDB_API_KEY)

    merged_dataset = load_from_disk("Data\\VQAMetaQA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_blip("Salesforce/blip2-opt-2.7b")
    # init_paligemma("kaischue/paligemma2-3b-pt-448-vis-ACLFigQA-de")
    # init_llava("llava-hf/llava-v1.6-vicuna-7b-hf")
    # init_tinychart("mPLUG/TinyChart-3B-768")
    # init_paligemma("kaischue/paligemma2-3b-pt-448-vis-ACLFigQA")
    # init_deepseek_janus("deepseek-ai/Janus-Pro-7B")

    for metadata in [True]:
        for lang in ["en", "de"]:
            for split in ["train", "val", "test"]:
                predict_dataset(
                    split=split, lang=lang, metadata=metadata,
                    model_name=model_id.split('/')[-1], predict_func=run_blip
                )
                # Finish the run
                wandb.finish()

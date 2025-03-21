## Dataset Access
The ACLFigVQA dataset is available for download [here](https://huggingface.co/datasets/kaischue/ACLFigVQA).

## Fine-tuned PaliGemma2 models
### German
- https://huggingface.cokaischue/paligemma2-3b-pt-448-vis-ACLFigQA-de
- https://huggingface.co/kaischue/paligemma2-3b-pt-448-ACLFigQA-de
  
### English
- https://huggingface.co/kaischue/paligemma2-3b-pt-448-vis-ACLFigQA
- https://huggingface.co/kaischue/paligemma2-3b-ft-448-ACLFigQA

## Installation
To set up the environment, follow these steps:

1. Clone the repository: ```git clone https://github.com/kaischue/ACLFigVQA.git```
2. Install the required dependencies: ```conda env create -f env_conda.yml```


## Usage
### Data Preprocessing
1. Download [ACL-Fig](https://huggingface.co/datasets/citeseerx/ACL-fig)
2. Run the following script to extend the dataset: ```python extend_dataset.py```


### Annotation
#### QA Generation
Use the gemini_api script to generate question-answer pairs: ```python gemini_api.py```
#### Validation
Use the validate tool to manually validate the QA Pairs: ```python validation_tool.py```

### Model Inference
- For all models except Llama use: ```model_inference.py```
- For LLama 3.2 Vision use: ```llama.py```

### Model Evaluation
- To evaluate QA Pairs generates by Gemini use: ```eval_gemini.py```
- To evaluate models on ROUGE, BLEU, BERT and METEOR scores use: ```eval_metrics.py```
- To evaluate models with GeminiEval use: ```gemini_eval.py```

### Finetune PaliGemma
Use the ```Fine_tune_PaliGemma.ipynb``` to finetune PaliGemma2 with the ACLFigVQA dataset

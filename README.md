# SBA 928 — Fine-Tuning a Small Language Model for Market Research

## Project Overview

This project explores how prompt engineering and parameter-efficient fine-tuning can be used to improve a small language model for market research tasks.

The goal is to create a reproducible workflow that:

- Uses a public dataset with a clear license.
- Converts review or market-related data into instruction/response training examples.
- Creates multiple versions of market-research prompts.
- Establishes a baseline using the original pretrained model.
- Fine-tunes a small language model using LoRA.
- Compares the base model with the fine-tuned model.
- Evaluates output quality using automated metrics and human ratings.
- Examines potential bias and limitations in the training data and model outputs.

## Project Goals

The primary goal is to investigate whether fine-tuning a small language model can improve its ability to perform useful market-research tasks such as:

- Summarizing customer opinions
- Identifying common themes
- Generating survey questions
- Comparing customer preferences
- Producing structured market insights

## Technology

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- TRL
- Accelerate
- Pandas
- Scikit-learn
- Matplotlib
- ROUGE

## Project Workflow

1. Set up the project environment.
2. Select and document a public dataset.
3. Create market-research prompts and prompt variations.
4. Prepare question/answer training data.
5. Split the data into training and testing sets.
6. Establish baseline results using the original model.
7. Fine-tune the model using LoRA.
8. Evaluate the fine-tuned model.
9. Compare base and fine-tuned outputs.
10. Examine potential bias.
11. Document findings, limitations, and recommended improvements.

## Model

The project will use a small instruction-tuned language model suitable for parameter-efficient fine-tuning with LoRA.

The final model choice and training configuration will be documented after the dataset and hardware environment have been confirmed.

## Dataset

Dataset selection is part of the project workflow.

The final README will document:

- Dataset name
- Source
- License
- Dataset size
- Relevant fields
- How the data was transformed for training

## Evaluation

The base model and fine-tuned model will be tested using the same evaluation prompts.

Evaluation will include:

- Side-by-side output comparison
- ROUGE-based evaluation where appropriate
- Human quality ratings
- Bias and fairness checks

## Reproducibility

Training evidence, evaluation results, model outputs, and relevant project artifacts will be retained so that the workflow can be reviewed and reproduced.

## Limitations

Potential limitations include:

- Dataset size and quality
- Dataset representation
- Model size
- GPU availability
- Training time
- Subjectivity of human evaluation
- Potential bias in the source data

## Conclusion

This project investigates the practical use of parameter-efficient fine-tuning to adapt a small language model for market-research applications while considering output quality, reproducibility, and potential bias.
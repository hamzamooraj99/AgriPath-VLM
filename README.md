# Crop Disease Detection using Vision Language Models (VLMs)
###### Author: Hamza Mooraj
###### Institute: Heriot-Watt University
###### Conact: - hhm2000@hw.ac.uk - hhmooraj@gmail.com
###### *This is the main repository for the implementation and experimentation as part of my BSc Computer Science (Artificial Intelligence) dissertation thesis.*
---
This study explores the use of multi-modal approaches in AI to enhance the accuracy of disease classification in crops; the modalities explored are visual and textual. The models and experiments are fine-tuned and trained on the AgriPath-LF16 dataset, which includes images and detailed metadata of crops affected by various diseases. The aim of this study is to explore the efficacy of new-age attention mechanisms and how they surpass conventional methods such as CNNs.

## Project Overview
### Objective:
The study aims to develop and evaluate a robust solution for crop disease detection in real-world agricultural settings by applying VLMs and utilising their ability to incorporate contextual information in tasks such as classification; an aspect where conventional deep learning methods fall short.
### Dataset - AgriPath-LF16:
The models used in the study are trained and fine-tuned on the AgriPath-LF16 dataset. The dataset includes:
- **Crops:** 16 classes of crops ranging across fruits, vegetables and grains.
- **Diseases:** 41 unique diseases across all 16 crops, resulting in 65 unique crop-disease pairs.
- **Sources:** Images are either lab-based[^1] or field-based[^2]
- **Splits:** The dataset is split into training, validation and test sets with an 80-10-10 ratio

  
AgriPath-LF16 is hosted using 🤗 Datasets where there is the full version with 111k samples, and a down-sampled version with 30k samples.

> https://huggingface.co/hamzamooraj99


[^1]: Images taken in a controlled environment
[^2]: Images taken in an uncontrolled environment

### Approach:
The study focuses on examining the shortcomings of conventional Deep Learning methods for image classification, while exploring the capabilities of attention-based mechanisms, such as, VLMs

**CNN Baseline:**
We start with making a baseline CNN model using a pre-trained ResNet50 model that is trained on AgriPath via transfer learning. By evaluating the efficacy of the model on predicting crop diseases, we can establish an understanding of the task and past work on this topic, for our further exploration into VLMs

**VLM Approach:**
We will make use of a pre-trained Qwen2-VL VLM and fine-tune it using AgriPath and the UnslothAI framework to optimise the fine-tuning process. The study will start by fine-tuning the smallest Qwen2-VL model with 2B parameters, and then move onto exploring an improvement with the 7B parameter model.

**Evaluation:**
We plan to evaluate the efficacy and accuracy of the three models and explore how VLMs compare to CNNs in a complex task such as image classification over many classes. We will also see how the models handle classification on field-based images where there is much more environmental noise, making it harder to perform any sort of pattern matching.

### Key Features:
- Multi-modal Approach
- Pre-trained Models and Transfer Learning/Fine-Tuning
- Balanced and Diverse Dataset with Varied Classes and Image Types
- Exploratory Analysis and Experimentation

## Results:
In-Progress...

## Tech Stack
- **Python 3.11**
- **PyTorch Lightning**
- **🤗 Datasets**
- **UnslothAI**
- **Tensorboard**
- **PyTorch 2.6.0**
- **CUDA Toolkit 12.4**
- **VSCode**
- **Anaconda**

## Hardware:
- AMD Ryzen 9 7950X CPU (16 Core, 32 Thread Processor)
- NVIDIA RTX 4080 Super (16GB GDDR6X VRAM)
- 32GB RAM (DDR5 6000MHz)

## License:
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

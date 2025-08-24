# LAMP: Leveraging-Autoregressive-Modeling-and-Prompt-Retrieval-for-Zero-Shot-Time-Series-Forecasting

<div align="center">

<!-- [![PWC](https://img.shields.io/badge/PDF-blue)]()
[![PWC](https://img.shields.io/badge/Supp-7DCBFF)]()
[![PWC](https://img.shields.io/badge/ArXiv-b31b1b)]() -->
[![PWC](https://img.shields.io/badge/Project%20Page-0047ab)](https://github.com/nycu-acm/LAMP-Leveraging-Autoregressive-Modeling-and-Prompt-Retrieval-for-Zero-Shot-Time-Series-Forecasting/website/)
<!-- [![PWC](https://img.shields.io/badge/Presentation-ff0000)]() -->
<!-- [![PWC](https://img.shields.io/badge/Bibtex-CB8CEA)](#citation) -->

</div>

---

## 🧪 Abstract

The application of Large Language Models (LLMs) in time series forecasting remains debated, as recent studies have shown that replacing LLMs with simpler components, such as basic attention layers, can lead to comparable or even improved performance, raising questions about the true contribution of LLMs in such tasks. To address this problem and unlock the potential of LLMs more effectively, we propose LAMP (Leveraging Autoregressive Modeling and Prompt Retrieval for Zero-Shot Time Series Forecasting), a zero-shot forecasting framework that leverages frozen pre-trained LLMs through dynamic prompt retrieval and autoregressive reasoning. LAMP first decomposes input time series into trend, seasonal, and residual components, then projects them into the LLM’s embedding space. For each component, the model retrieves semantically aligned prompts from specialized prompt pools using a similarity-based matching mechanism. These prompts are enhanced with textual guidance generated from dataset descriptions, which are encoded via a frozen text embedder to provide semantic conditioning. The selected prompt embeddings are then fused with the input and fed into a frozen LLM, which autoregressively generates future values without any parameter updates. This design enables LAMP to generalize across domains and forecasting horizons while remaining computationally efficient. Experiments across diverse benchmarks confirm that LAMP achieves strong zero-shot forecasting performance, especially in long-horizon and non-stationary scenarios, demonstrating the power of prompt-driven adaptation in bridging time series and language models. This framework enables practical deployment in real-world systems where domain shift and limited training data pose significant challenges.

---

## 🏗️ Architecture

<p align="center">
  <img src="/website/assets/img/Fig2.png" alt="LAMP Architecture" width="820">
</p>

---

## ⚙️ Installation

```bash
conda env create -f environment.yml
conda activate lamp
```

> Requires Python ≥ 3.8 and a CUDA-enabled GPU for training.

---

## 📚 Dataset
   Download the data from [[Google Drive]](https://drive.google.com/file/d/1Q7mcEXlSwvv6WFzaDKxK6hH9DrbQeKn1/view?usp=sharing), and place the downloaded data in the folder`./dataset`. You can also download the STL results from [[Google Drive]](https://drive.google.com/file/d/1ho3EvABbr0chitKcJtP0kM-MDt1PE25p/view?usp=sharing), and place the downloaded data in the folder`./stl`.
---

## Pretrained LLM 

Download the large language models from [Hugging Face](https://huggingface.co/). The default LLM is GPT2, you can change the `llm_ckp_dir` in `run_multidomain.py` to use other LLMs.
   * [GPT2](https://huggingface.co/openai-community/gpt2)

For example, if you download and put the GPT2 directory successfully, the directory structure is as follows:
   - data_provider
   - dataset
   - gpt2
     - config.json
     - ...
   - ...
   - run_multidomain.py
---

### Pre-Training Stage
```
bash ./scripts/[ETTh1, ETTh2, ETTm1, ETTm2].sh
```

### Test/ Inference Stage

After training, we can test LAMP model under the zero-shot setting:

```
bash ETTh1_test.sh
```

## Pre-trained Models

You can download the pre-trained model from [[Google Drive]](https://drive.google.com/file/d/1xJKOXguoA0d2Qy6-D3pEGd2qd3Xq5SgH/view?usp=sharing) and then run the test script.




---
permalink: /ee243-final-project
title: "EE243 Final Project - MMedPO"
author_profile: false
---

Ashish Kulkarni `akulk050@ucr.edu` \
Deepthi Dayanand `ddaya003@ucr.edu`

Find our YouTube video [here](https://youtu.be/oGo9rtGeCh8).

Find our full code [here](files/ee243_project.ipynb).

## How MMedPO works

<div style="display: flex; align-items: center; justify-content: center; gap: 10px; text-align: center;">
  <div>
    <img src="images/ee243_infiltration.png" width="300">
  </div>
  <div>+</div>
  <div>What is the primary clinical finding in this radiograph?</div>
  <div>=</div>
  <div>The image shows evidence of infiltration.</div>
</div>


This project addresses a major flaw in medical AI known as modality misalignment. Often, these models rely too much on text and ignore the actual medical image, causing them to hallucinate diseases that aren't pictured. MMedPO introduces a specialized training layer to fix this by balancing visual evidence with medical text.

## MMedPO's architecture

<div style="display: flex; align-items: center; justify-content: center; gap: 10px; text-align: center;">
  <div><b>LLaVA-Med</b><br>
Large Language and Vision Assistant for Biomedicine<br>
Li et al.
</div>
  <div><b>MedKLIP</b><br>
Medical Knowledge Enhanced Language-Image Pre-Training in Radiology<br>
Wu et al.
</div>
</div>


MMedPO relies on two foundational technologies. First is LLaVA-Med, which serves as the primary base architecture. It is a large language-and-vision assistant tuned for the biomedical domain. Second is MedKLIP, which acts as the curation engine. MedKLIP identifies critical lesion regions in scans and intentionally adds localized noise to obscure them. By forcing the base model to contrast clear images against these corrupted ones, the AI learns to pay attention to actual medical anomalies instead of just guessing based on text.

## Evaluation dataset

![](images/ee243_project_10_0.png)

To evaluate the model, I used the ChestX-ray8 dataset by Wang et al. in the paper [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/abs/1705.02315), which consists of 112,120 X-ray images with disease labels from 30,805 unique patients. The disease labels are:
* Atelectasis
* Consolidation
* Infiltration
* Pneumothorax
* Edema
* Emphysema
* Fibrosis
* Effusion
* Pneumonia
* Pleural_thickening
* Cardiomegaly
* Nodule
* Mass
* Hernia
* No Finding

The authors claim the disease labels are over 90% accurate.

Of the 100K+ images, we randomly sampled from 4,999 of them for this evaluation.

## Evaluation strategy

The evaluation was designed around three distinct types of prompts. The first is a 'Neutral' prompt, simply asking what findings are present in the X-ray. The second is a 'Truth' prompt, which includes the actual, factual labels of the image. Finally, they used a 'Gaslight' prompt. This prompt intentionally injects false labels—asking the model to describe the severity of two diseases that are absolutely not in the scan.

![](images/ee243_infiltration.png)

Neutral prompts:
- What is the primary clinical finding in this radiograph?

Truth prompts
- What is the severity of the infiltration in this x-ray?
- Do you agree that this x-ray shows signs of infiltration?

Gaslighting prompts:
- How severe is this case of pneumonia?
- There are no findings from this x-ray, right?

## Results

In our experience, the model deferred to the gaslighting prompt's diagnoses 100% of the time.

When no diagnosis was provided in the accompanying prompt, the model was correct 40% of the time.

When the accompanying prompt contained the ground truth (diagnosis) for the image, the model still only got the right diagnosis 56% of the time.

These findings were quite strange, as one would assume if the model can be gaslit 100% of the time, then providing the ground truth in the prompt should give the model close to 100% accuracy. This did not prove to be the case.
Coupled with the fact that one of the datasets that MMedPO was trained on was [MIMIC-CXR](https://www.nature.com/articles/s41597-019-0322-0), which is a chest radiography dataset, these findings are surprsing.

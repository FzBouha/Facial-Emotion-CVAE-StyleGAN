# Facial-Emotion-CVAE-StyleGAN
# Facial Expression Analysis and Emotion Simulation  
## CVAE + StyleGAN Hybrid Approach for Mental Health Applications


## Project Overview

This project explores the analysis and simulation of facial expressions in the context of mental health applications.  

The objective is twofold:

1. **Emotion recognition** from facial images.
2. **Controlled emotion generation** using generative models.

We investigate the complementarity between a **Conditional Variational Autoencoder (CVAE)** for emotion control and **StyleGAN2-ADA** for high-quality photorealistic generation.

---

##  Motivation

Facial expressions are a key indicator of emotional states.  
In mental health applications, automatic analysis and controlled simulation of facial expressions may support:

- Emotion-aware digital therapy tools  
- Virtual agents and interactive avatars  
- Behavioral monitoring systems  

However, there is a trade-off between:

- Emotional control (structured latent models)
- Photorealistic image quality (adversarial models)

This project studies that trade-off.

---

## Dataset

We use **AffectNet**, one of the largest facial expression datasets collected "in the wild".

### Emotion classes:
- anger  
- contempt  
- disgust  
- fear  
- happy  
- neutral  
- sad  
- surprise  

Images are resized to **128×128** resolution.  
Basic preprocessing and data augmentation are applied.

---

## Models Used

### Emotion Classifier
- Architecture: **ResNet18**
- Fine-tuned for 8 emotion classes
- Evaluation metrics: Accuracy and F1-score

---

### Conditional Variational Autoencoder (CVAE)

- Learns a structured latent emotional space
- Conditioned on emotion labels
- Loss function:
  - Reconstruction loss
  - KL divergence (β-VAE strategy)
  - Perceptual loss (VGG-based fine-tuning)

Goal: generate faces conditioned on a target emotion.

---

### StyleGAN2-ADA 

- Used for photorealistic face generation
- No retraining performed
- Used for qualitative comparison
- Latent optimization guided by classifier explored as hybrid approach

---

##Hybrid Strategy

The hybrid approach is explored at two levels:

1. **Conceptual fusion**
   - CVAE → emotion control
   - StyleGAN → photorealistic generation

2. **Latent optimization (exploratory)**
   - StyleGAN latent space adjusted to maximize classifier emotion probability
   - Regularization applied to preserve realism

---

##  Results Summary

### Emotion Control Accuracy (Generated → Classifier)

| Emotion   | Control Accuracy |
|-----------|------------------|
| neutral   | 95.0%            |
| sad       | 77.5%            |
| happy     | 37.0%            |
| surprise  | 10.5%            |
| contempt  | 2.0%             |
| fear      | 1.0%             |
| anger     | 0.0%             |
| disgust   | 0.0%             |

Macro-average: **27.9%**

---

## Key Observations

- CVAE successfully captures global facial structure.
- Generated images remain blurry due to VAE smoothing effect.
- Stronger emotions (anger, disgust) are difficult to model.
- StyleGAN produces highly realistic images but lacks direct emotion control.
- Clear trade-off between control and realism.

---

## ⚠ Limitations

- Blurry reconstructions (VAE limitation)
- Dataset imbalance effects
- Dependence on classifier for evaluation
- Limited GPU resources (Kaggle)

---

## Future Work

- Joint training between emotion control module and GAN
- Latent space editing with stronger regularization
- Diffusion-based generative models
- Region-aware facial encoding (eyes, mouth emphasis)

---

## Implementation

The entire implementation is provided in the Jupyter notebook:


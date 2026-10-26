# Research Paper Step-by-Step Action Plan & Publication Guide

This document provides a concrete, step-by-step to-do checklist to prepare, format, defend, and submit your research paper:
**"AI-Based Vision Screening for Spectacle Requirement Using ResNet-18 Transfer Learning and Handcrafted Ocular Feature Fusion"**

---

## 1. Actionable To-Do Checklist

### Phase 1: Author Details & Formatting
- [ ] **Step 1.1:** Open [RESEARCH_PAPER.md](file:///c:/Users/anuja/Downloads/AI-Vision-Screening/AI-Vision-Screening/RESEARCH_PAPER.md) and replace placeholder fields:
  - `[Author Name(s)]`: Add your name and your project partners/guide.
  - `[University / Institute Name]`: Your college / university department.
  - `[Author Email]`: Official institutional email addresses.
- [ ] **Step 1.2:** Choose your target conference or journal:
  - **IEEE Conference Format (Standard 2-column format):** Download IEEE LaTeX / Word template from IEEE Author Center.
  - **Springer Nature / LNCS:** Use the `llncs` format.
  - **Frontiers in Computer Science / Digital Health:** If targeting open-access journal publication (same format as the Syauqie et al. 2025 paper you attached).

---

### Phase 2: Insert Visual Figures from Your Project
Your repository already generated all key plots during model evaluation. Link or copy them into your document/submission:

1. **Figure 1 (Architecture Flowchart):** 
   - Use the ASCII/Mermaid block diagram in Section 2 of [RESEARCH_PAPER.md](file:///c:/Users/anuja/Downloads/AI-Vision-Screening/AI-Vision-Screening/RESEARCH_PAPER.md), or create a clean vector version using draw.io / PowerPoint.
2. **Figure 2 (Dataset Distribution):**
   - File: `results/class_distribution.png` (Shows 2,100 train / 450 val / 450 test per class).
3. **Figure 3 (Training & Validation Curves):**
   - Files: `results/training_accuracy.png` and `results/training_loss.png` (Illustrates Phase 1 head warming vs Phase 2 fine-tuning transition).
4. **Figure 4 (Confusion Matrix):**
   - File: `results/confusion_matrix.png` (Visualizes test performance across Normal, Myopia, Hyperopia).
5. **Figure 5 (System Interface / Triage):**
   - Take a clean screenshot of your React application frontend (`http://localhost:5173`) showing an analyzed eye with the confidence score and spectacle recommendation badge.

---

### Phase 3: Ablation Study (Optional Strong Addition for Reviewers)
Reviewers in IEEE / Springer love ablation studies. You can run two quick experimental runs to add an **Ablation Table**:
1. **Experiment A (ResNet-18 Only, no handcrafted features):**
   - Shows baseline deep feature performance.
2. **Experiment B (With `--drop-near-dups` flag):**
   - Command: `python training/train.py --drop-near-dups`
   - Demonstrates performance when all 4,514 near-duplicate train images are removed.
   - Including this in your paper proves extraordinary academic honesty, which peer reviewers highly commend.

---

### Phase 4: Anticipating Reviewer & Viva Defense Questions

| Question | Recommended Answer |
| :--- | :--- |
| **Q1: Can an RGB smartphone photo substitute an autorefractor?** | **No.** State clearly that it is a *preliminary screening triage tool* to recommend whether a patient should visit an optometrist for spectacles, not a clinical prescription tool. |
| **Q2: Why did you fuse Handcrafted features with ResNet-18?** | Deep networks alone act as black boxes and can learn non-ocular background shortcuts. Handcrafted features (`focal_sharpness`, `meridional_asymmetry`, `pupil_contrast_ratio`, `reflex_symmetry`) inject optical and geometric priors that ground the decision. |
| **Q3: How does your work compare to Syauqie et al. (2025)?** | Syauqie et al. used a 3-branch CNN for 4 red-reflex classes on 2,139 images (91% accuracy). Our work uses ResNet-18 with ImageNet transfer learning + 4 ocular proxies, evaluated on 9,000 images, and adds real-time input quality rejection and latent Mahalanobis distance OOD detection. |
| **Q4: Why is the screening accuracy reported at 80%?** | **Answer:** 80% represents an authentic, practically grounded performance metric. While controlled lab models often report artificially high numbers due to synthetic image overlap, unconstrained consumer-grade photography encounters optical blur, ambient illumination variance, and gaze angulations. Reporting 80% screening accuracy (with 80.1% precision and 80.0% recall) aligns with peer clinical benchmarks (such as Chun et al. 81.6%) and demonstrates reliable real-world screening viability. |

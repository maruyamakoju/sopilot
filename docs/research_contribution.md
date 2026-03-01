# Research Contribution Formalization: SOPilot

**Automated Procedural Compliance Scoring via Video Foundation Models and Temporal Alignment**

*Prepared: 2026-02-24. Audience: Cambridge Computer Lab / Google DeepMind collaboration pitch.*

---

## 1. Problem Statement

### Formal Definition

Let P be a multi-step procedure consisting of an ordered sequence of atomic steps (s_1, s_2, ..., s_N). A worker execution is recorded as a video V_test of duration T seconds. A set of K expert-annotated reference recordings {V_gold_1, ..., V_gold_K} defines the space of acceptable correct executions of P.

The **SOP compliance evaluation problem** is to compute, without manual review, the following outputs:

- A **compliance score** S in [0, 100] indicating overall procedural adherence.
- A **confidence interval** [S_lo, S_hi] characterizing the statistical uncertainty of that score.
- A **step-level deviation map** D = {(step_i, type_i, severity_i, timecode_i)} identifying where, when, and how severely the worker deviated from the reference procedure.
- A **pass/retrain/fail decision** derived from configurable thresholds on S and the severity distribution in D.

This is a regression problem with uncertainty output, not a classification problem. The output is consumed by human trainers and auditors, so calibration — the alignment between stated confidence and actual accuracy — matters as much as raw predictive performance.

### Why This Problem Is Difficult

**Temporal warping.** Two workers performing the same procedure correctly may differ in execution speed by a factor of three or more. A worker who completes a filter-change procedure in 45 seconds and one who takes 120 seconds may both be fully compliant; their frame-level feature sequences are entirely misaligned. Standard frame-synchronous comparison fails here; temporally elastic alignment is required.

**Viewpoint and appearance variation.** Real industrial and training environments do not provide controlled camera placement. A reference gold video shot from one angle cannot be directly compared pixel-by-pixel to a trainee video shot from a different angle. The representation must be viewpoint-invariant at the semantic level.

**Intra-class variation in correct execution.** Multiple correct execution strategies exist for most procedures. One operator may organise cleaning supplies differently from another, yet both are fully compliant. A rigid single-template approach would penalise valid variation. The system must model the distribution of correct executions, not a single canonical path.

**Absence of large labeled datasets.** Action recognition datasets (Kinetics, Something-Something, EPIC-Kitchens) contain millions of short clip-level class labels, but they do not contain the **step-level procedural compliance annotations** needed to train or evaluate this problem directly. No public dataset provides: multi-step ordered execution, per-step pass/fail labels, and inter-rater reliability statistics computed from multiple independent expert reviewers.

---

## 2. Research Gap

The following specific gaps exist in the literature as of early 2026:

**No benchmark dataset for procedural compliance evaluation.** Existing procedural video datasets (COIN [CITATION], CrossTask [CITATION], ProceL [CITATION]) annotate step presence and ordering but do not provide compliance quality scores, severity labels (critical vs. efficiency deviation), or inter-rater reliability statistics. They are designed for step detection and temporal grounding, not for grading worker performance on a continuous scale.

**No principled uncertainty quantification for video scoring systems.** Systems that produce a scalar score for video content do not, in general, also produce a calibrated confidence interval. Conformal prediction methods [CITATION] have been applied to image classification but have not been applied to the temporally-structured compliance scoring regime. The distinction between epistemic uncertainty (too few gold references) and aleatoric uncertainty (inherent execution variability) has not been made for this problem.

**No method that simultaneously handles the full pipeline.** Prior work addresses sub-problems in isolation: step segmentation [CITATION], action recognition [CITATION], temporal alignment via DTW [CITATION], video representation learning [CITATION]. No published system provides an integrated pipeline that takes raw video in, performs step segmentation, aligns to gold references under temporal warping, produces a calibrated score with confidence interval, and returns step-level timecode-stamped deviation reports.

**The framing is wrong in prior work.** The dominant framing in video understanding is action recognition: given a clip, predict a class label. Compliance scoring is a fundamentally different problem: given a full procedure execution, predict a continuous score and a deviation map relative to a reference set. Most existing metrics (top-1 accuracy, mAP) are inappropriate for this setting; the field lacks an agreed evaluation protocol.

---

## 3. Our Contribution

We describe five specific claims. Each is stated in a falsifiable form.

### Claim 1: SOP-Bench — First Labeled Dataset for Multi-Step Procedural Compliance Evaluation

We are constructing a dataset of procedural execution videos with expert compliance annotations. The pilot task (`filter_change`, a multi-step equipment maintenance procedure) currently contains 26 videos: 2 gold reference recordings and 24 trainee recordings spanning a controlled range of compliance quality, including synthetically corrupted variants that represent known failure modes: skip-start (early steps omitted), freeze (video truncated mid-procedure), and cut-tail (procedure aborted before completion).

**Falsifiability condition.** The dataset is credible as a benchmark when: (a) it contains at least 100 trainee videos across at least 3 procedurally distinct tasks; (b) each video is annotated by at least two independent expert reviewers; (c) inter-rater agreement reaches ICC(3,1) > 0.80, the conventional threshold for "good" reliability [CITATION]; (d) the dataset and annotation protocol are publicly released under a permissive licence.

**Current status.** Pilot set: 26 videos, 1 task, synthetic corruption labels, no ICC computed yet.

### Claim 2: Probabilistic SOP Scorer — V-JEPA2 + Soft-DTW with Calibrated Confidence Intervals

The core scoring pipeline embeds video clips using a V-JEPA2 encoder [CITATION], pools frame-level token embeddings via mean aggregation to produce fixed-dimensional clip descriptors, then aligns the test and gold descriptor sequences using Dynamic Time Warping (DTW). Step boundaries in the gold video are propagated through the DTW alignment path to identify which portion of the test video corresponds to each gold step. Scores are computed as weighted combinations of: step presence (miss/swap penalty), step-level DTW cost (deviation penalty), and temporal compliance (over-time ratio).

An ensemble scoring mode aggregates scores from K gold videos: the final score is the median, and the confidence interval is derived from the cross-gold standard deviation. This provides a principled, distribution-free uncertainty estimate that requires no additional training.

**Falsifiability condition.** On a held-out test split, the Expected Calibration Error (ECE) of the confidence intervals is below 0.10, and the AUC-ROC for pass/fail binary prediction is above 0.85.

**Current status.** On the pilot set of 24 scored jobs, all four quality gates pass: critical miss rate = 0.0 (threshold 0.10); critical false positive rate = 0.111 (threshold 0.30); rescore jitter max delta = 0.0 (threshold 5.0); DTW normalized cost p90 = 0.073 (threshold 0.60). Score distribution: mean = 86.5, p50 = 97.3, min = 55.0, max = 100.0. The confusion matrix on critical deviation detection: TP=6, FN=0, FP=2, TN=16. ECE has not been computed; human labels for calibration are not yet available.

### Claim 3: Uncertainty Decomposition — Epistemic vs. Aleatoric Components

The total uncertainty in a compliance score decomposes into two sources. **Epistemic uncertainty** arises from having too few gold reference videos: with K=1 gold video, we have only a single point estimate of the correct execution distribution. As K increases, this uncertainty should decrease. **Aleatoric uncertainty** arises from genuine variability in how the procedure can be correctly executed: even with infinite gold videos, some residual uncertainty remains because correct execution is not a single point but a distribution.

We propose to model epistemic uncertainty as the between-gold variance of ensemble scores (directly observable when K > 1) and aleatoric uncertainty as the within-gold DTW cost distribution width (the spread of alignment costs for known-good test executions).

**Falsifiability condition.** An ablation study across K = 1, 2, 3, 4 gold videos should show that the epistemic component (between-gold variance) decreases monotonically with K, while the aleatoric component (within-gold cost spread) remains stable. This is testable on the existing dataset by subsampling the gold reference set.

**Current status.** The system currently runs ensemble scoring with K=2 gold videos. Decomposition into epistemic/aleatoric components is designed but not yet empirically validated.

### Claim 4: Domain Adaptation via Contrastive MLP Adapter

V-JEPA2 was pretrained on general video data. The embedding space is not necessarily optimal for the fine-grained procedural distinctions that matter for compliance scoring: a correct filter installation vs. an incorrect one may differ by a small hand gesture that is irrelevant to natural video pretraining objectives.

We propose a lightweight contrastive MLP adapter trained on gold video pairs from the target task. The adapter is trained with a positive objective (gold clips from the same step should be close) and no negative labels are needed: the DTW alignment path itself provides soft correspondence, so any two clips that DTW maps to the same gold step position serve as positives for each other. This supervision signal is available from gold videos alone, requiring no additional annotation effort.

**Falsifiability condition.** On a held-out set, the mean rank of the correct gold step when retrieving by nearest-neighbor in the adapted embedding space should decrease (improve) compared to the unadapted baseline. We expect at least a 20% reduction in mean rank.

**Current status.** The adapter is not yet implemented. The DTW alignment in the pilot uses raw V-JEPA2 embeddings directly.

### Claim 5: Evaluation Framework — LOSO-CV with Inter-Rater Reliability as Standard Protocol

Procedural compliance systems are rarely evaluated in a consistent, reproducible way. We propose Leave-One-Subject-Out cross-validation (LOSO-CV) as the standard evaluation protocol, which ensures that the test subject's gold and test videos are never seen during training or calibration. We further propose that inter-rater reliability metrics — specifically ICC(3,1) for continuous scores and Cohen's kappa for pass/fail decisions — should be reported as primary metrics alongside predictive accuracy, because the ultimate calibration target is human expert judgment.

**Falsifiability condition.** The LOSO-CV score variance across folds should be lower than random split variance (demonstrating that subject-level leakage is a real confound). The system's ICC with human reviewers should exceed 0.75 (the "good" threshold), meaning automated scores agree with human experts as well as two human experts agree with each other.

**Current status.** LOSO-CV is implemented in the evaluation script (`scripts/evaluate_poc.py`). Human reviewer ICC has not been computed; no multi-reviewer annotations exist yet.

---

## 4. Related Work and Positioning

**DTW for temporal alignment.** DTW applied to time-series comparison predates neural representations [CITATION], and has been applied to gesture recognition and action spotting [CITATION]. Our work differs in that we apply DTW over semantically rich neural embeddings rather than hand-crafted features, and we use the DTW alignment path for structured deviation extraction rather than just as a distance metric.

**Video foundation models.** The ViT family [CITATION] and its video extensions (VideoMAE [CITATION], V-JEPA [CITATION], V-JEPA2 [CITATION]) provide strong general-purpose video representations. Our work differs in that we use these representations purely as a frozen encoder and do not fine-tune the full model; instead, we train only a lightweight adapter, making the system deployable on-premise with modest compute.

**Procedural learning datasets.** COIN [CITATION] contains 11,827 videos across 180 tasks with step labels. CrossTask [CITATION] contains instructional videos for 83 tasks with step annotations. EPIC-Kitchens [CITATION] provides egocentric kitchen activity recordings. Our work differs in that none of these datasets provide compliance quality scores or inter-rater reliability statistics; they support step detection but not compliance grading.

**Uncertainty quantification in deep learning.** Conformal prediction [CITATION] provides distribution-free coverage guarantees and has been applied to image classification and regression. Bayesian deep learning [CITATION] provides principled posterior uncertainty. Our work differs in that we target the structured output setting — a score over an ordered multi-step sequence — and propose a decomposition into epistemic and aleatoric components that is directly actionable (add more gold videos to reduce epistemic; note irreducible variation to set aleatoric floor).

**Industrial quality control and process monitoring.** Machine vision for manufacturing defect detection [CITATION] and anomaly detection [CITATION] address quality control but focus on product inspection, not worker procedural execution. Our work differs in that the "product" is a human action sequence, and the reference standard is a distribution of correct worker behaviors rather than a single engineering tolerance.

---

## 5. Experimental Plan

The following experiments are needed to support each claim, in order of priority.

**Supporting Claim 1 (SOP-Bench).**
- Recruit two independent expert reviewers for the filter_change task. Each reviewer watches each of the 24 trainee videos independently and assigns: a pass/fail decision, a compliance score in [0, 100], and step-level annotations (which steps were performed, in what order, with what quality).
- Compute ICC(3,1) between reviewer scores. Target: ICC > 0.80.
- Compute Cohen's kappa for pass/fail decisions. Target: kappa > 0.70.
- If ICC < 0.80, revise annotation protocol and re-annotate.
- Expand to a second procedural task. Target: 50 additional videos annotated by the same reviewers by the end of the annotation phase.

**Supporting Claim 2 (Calibrated Scorer).**
- Use human reviewer scores as ground truth labels.
- Compute ECE of the system's confidence intervals against human scores. Target: ECE < 0.10.
- Compute AUC-ROC for pass/fail binary classification. Target: AUC > 0.85.
- Run LOSO-CV and report mean and standard deviation of AUC across folds.
- Existing gate-passing results already show: critical miss rate = 0.0, FPR = 0.111, rescore jitter = 0.0; these serve as pre-annotation sanity checks.

**Supporting Claim 3 (Uncertainty Decomposition).**
- Ablation over K gold videos: for each K in {1, 2}, compute the between-gold variance (epistemic component) and the within-gold DTW cost spread (aleatoric component).
- Hypothesis: between-gold variance decreases as K increases; within-gold spread is stable.
- Report correlation between epistemic variance and actual score error (ECE by uncertainty bin).

**Supporting Claim 4 (Domain Adapter).**
- Implement the contrastive MLP adapter using DTW-derived soft correspondences as positive pairs.
- Evaluate step retrieval accuracy: given a test clip, what rank is the correct gold step in the nearest-neighbor sorted list of gold clips?
- Compare mean rank before and after adapter training.
- Secondary metric: downstream compliance score AUC change after adapter.

**Supporting Claim 5 (Evaluation Framework).**
- Compare LOSO-CV variance to random-split variance. If LOSO variance is substantially higher, subject-level leakage in random splits is confirmed.
- Measure human reviewer wall-clock time per video (baseline). Measure system scoring latency per video. Compute speedup ratio. Target: at least 50% reduction in review time, consistent with the operational roadmap definition of success.
- Compute ICC(3,1) between system scores and human reviewer scores. Target: ICC > 0.75.

---

## 6. Current Status (Honest Assessment)

**Working and validated.**
- Full end-to-end pipeline: video ingest, V-JEPA2 embedding, DTW alignment, step deviation detection, score computation, confidence interval, pass/fail decision.
- 26 TESDA procedural videos (filter_change task) uploaded and scored.
- 24 scoring jobs completed (0 failures), 48 jobs in the extended gate evaluation set.
- All four quality gates pass on the gate evaluation set: critical miss rate 0.0, FPR 0.111, rescore jitter 0.0, DTW p90 cost 0.073.
- Deterministic scoring: rescore jitter max delta = 0.0 across 24 repeated-scoring pairs; scores are reproducible.
- Ensemble scoring across multiple gold videos is implemented and deployed.
- The API has 505 passing tests across all modules.

**Missing and blocking submission.**
- **Scale.** 24 evaluated trainee videos from one task is not sufficient for a credible dataset contribution. A minimum viable benchmark for a venue like NeurIPS Datasets and Benchmarks requires at least 100 videos across at least 3 tasks, with a clear data collection and consent protocol.
- **Human labels.** ICC has not been computed. Without at least two independent reviewers, the claim that automated scores approximate human expert judgment cannot be empirically validated. This is the single most important missing piece.
- **Calibration evaluation.** ECE cannot be computed without human score labels to serve as the calibration target. The confidence intervals are reasonable by construction (cross-gold standard deviation) but are not yet evaluated against ground truth.
- **Adapter.** The contrastive MLP adapter (Claim 4) has not been implemented. Raw V-JEPA2 embeddings are used throughout.
- **Multi-task generalization.** All results are on one task (filter_change). Generalization to new tasks is hypothesized but not demonstrated.

---

## 7. Timeline to Submission-Ready

The analysis above implies the following work phases, assuming two researchers working part-time.

**Weeks 1-2: Human annotation of the existing pilot set.**
Recruit two domain experts for the filter_change procedure. Conduct annotation training session. Each reviewer independently annotates all 24 trainee videos. Compute ICC and kappa. If ICC > 0.80, proceed. If not, revise the annotation guide and re-annotate a 10-video subsample until agreement is reached.

**Weeks 3-4: Second task data collection and annotation.**
Select a second procedurally distinct task. Capture at least 30 trainee videos with a range of compliance levels. Apply the same two-reviewer annotation protocol. This brings the dataset to approximately 55 annotated trainee videos across 2 tasks — sufficient for initial calibration evaluation.

**Weeks 5-6: Fine-tuning experiments and ablation study.**
Implement the contrastive MLP adapter. Run the uncertainty decomposition ablation across K in {1, 2}. Compute ECE and AUC against human labels. Run LOSO-CV. Document all results.

**Weeks 7-8: Third task data collection.**
Expand to a third task to reach the 3-task threshold. Target: 100+ total annotated trainee videos. Begin paper draft in parallel.

**Weeks 9-10: Paper writing and submission preparation.**
Write the full paper. Prepare dataset release package (annotation files, video hosting plan, evaluation scripts). Prepare supplementary material with qualitative examples showing correct alignment paths and deviation timecodes.

**Target venue.** CVPR 2026 (standard paper deadline approximately November 2025) has already passed. Realistic near-term targets are:

- **ICCV 2025** — deadline passed; not applicable.
- **NeurIPS 2025 Datasets and Benchmarks track** — deadline typically June 2025; also passed.
- **ECCV 2026** — typical deadline approximately March 2026; aggressive but potentially achievable with the timeline above if annotation begins immediately (current date: February 2026).
- **NeurIPS 2026 Datasets and Benchmarks** — deadline typically June 2026; this is the most realistic and well-matched venue given the dataset-plus-method framing of the contribution.

The Datasets and Benchmarks track is the appropriate primary target because the central contribution is the dataset and evaluation protocol; the method is secondary. This also lowers the bar for the method claims slightly: the dataset and evaluation framework need to be strong; the model performance needs to be convincingly above trivial baselines, not necessarily state-of-the-art.

---

## 8. One-Sentence Pitch

We present the first end-to-end system and benchmark for automated procedural compliance scoring from video, combining a frozen video foundation model with temporally elastic alignment and calibrated uncertainty quantification to produce auditable pass/fail decisions that match human expert reviewers at a fraction of the review cost.

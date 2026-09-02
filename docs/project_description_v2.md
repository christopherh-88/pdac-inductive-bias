Project Description (version 2\)

**Inductive Bias on a Hard Organ: A Pre-Registered Failure-Mode Comparison of a Convolutional U-Net and a Pure Transformer for Pancreatic Ductal Adenocarcinoma Segmentation in CT**

**Short title:** *Where Does Architecture Matter? Stratified Failure Modes of CNN and Transformer Segmenters on PDAC*

August 20, 2026\. Replaces the version 1 draft. Appendix A lists every change from version 1 and the reason for it.

# **1\. Summary**

This project is a controlled comparison of two 3D segmenters trained under one pipeline on public, multi-source, PDAC-only CT data: a residual-encoder U-Net (nnU-Net ResEnc) and a pure transformer (PrimusV2), both inside the nnU-Net framework at a matched training budget. The output is not a winner. It is a stratified map of where each architecture fails (by lesion volume, lesion-to-parenchyma contrast, and acquisition source), two causal tests of why (an occlusion test of context use and an identity ablation of the attention blocks), and an estimate of how much of the apparent cross-source degradation sits within annotation-boundary tolerance. Hypotheses, stratum thresholds, and decision rules are fixed before training and committed to a public repository.

# **2\. Motivation and position relative to prior work**

Pancreatic ductal adenocarcinoma (PDAC) lesions are small, low contrast against parenchyma, and sit in an organ whose boundaries are poorly defined and depend on contrast timing and scanner. Most segmentation papers report a headline Dice score that hides the question this project asks: which inductive bias suits this class of image, and where does each one break?

The aggregate comparison between CNNs and transformers is no longer open. nnU-Net Revisited (Isensee et al., MICCAI 2024\) benchmarked CNN, transformer, and Mamba segmenters under a fair protocol on six datasets and found that SwinUNETR, nnFormer, and CoTr do not match convolutional U-Nets, with the stated caveat that this holds for training from scratch on small datasets. Primus (Wald et al., 2025\) showed that most hybrid transformers barely use their attention: replacing the transformer blocks with identity costs under 3 DSC in six of nine architectures. The same work introduced PrimusV2, a pure transformer that reaches parity with ResEnc-L.

This project takes those results as its starting point rather than re-deriving them. Its contribution is the stratified, mechanistic, and boundary-aware analysis on one hard organ, with predictions written down before any model is trained.

# **3\. Research question and pre-registered hypotheses**

**Central question.** Under one training pipeline and a matched compute budget, where (by lesion volume, contrast, and source) and why (context use, attention dependence, annotation tolerance) do a convolutional and a pure-transformer segmenter fail differently on PDAC?

Each hypothesis below carries a prediction, a test, and a decision rule. All three are frozen in the repository before training starts. Margins (3 Dice points for equivalence, 2 Dice points for the occlusion contrast, 50% for the boundary share) are pre-set and will be reported as such whether or not they are met.

| Hypothesis | Prediction (fixed before training) | Test | Decision rule |
| :---- | :---- | :---- | :---- |
| H1. Locality vs lesion size | The CNN advantage, defined per case as ΔDice \= Dice(CNN) − Dice(transformer), is largest for small lesions and shrinks with volume. | Per-case paired differences regressed on log lesion volume, CNR, and source; patient-level bootstrap CIs; Wilcoxon signed-rank within volume tertiles. | Supported if the 95% CI of the slope of ΔDice on log volume lies below zero. The large-lesion tertile is reported with a two one-sided equivalence test at ±3 Dice points. |
| H2. Context use | The transformer’s predictions depend on anatomy farther from the lesion than the CNN’s effective receptive field. | Occlusion test applied identically to both arms: replace spherical shells at 10–20, 20–40, and 40–80 mm from the manual lesion surface with the shell’s own mean HU, re-infer, record Dice loss per shell. CNN effective receptive field measured empirically as a reference. | Supported if the transformer’s Dice loss in the 40–80 mm shell exceeds the CNN’s by at least 2 Dice points with a bootstrap CI excluding zero. |
| H2b. Attention dependence | The transformer’s stratified failure map is produced by its attention blocks, not by its patch embedding and decoder. | Identity control: the same architecture trained on the same folds with the transformer blocks replaced by identity (the Primus stress test). | If the identity control reproduces the full model’s stratified map within CI, attention is not the cause and the paper says so. |
| H3. Source shift vs boundary tolerance | A measurable share of the Dice lost under cross-source evaluation lies within the tolerance implied by boundary ambiguity in the reference. | Leave-one-source-out training per arm; boundary-tolerance floor from 1- and 2-voxel morphological perturbation of the reference (no paired independent delineations exist to add an inter-rater proxy — see analysis notes); NSD at 2 mm reported alongside Dice. | Report the share of each arm’s cross-source Dice loss that falls inside the floor. Supported if that share exceeds 50% for at least one arm. |

# **4\. Study design**

## **4.1 The two arms**

* **Convolutional arm.** nnU-Net ResEnc (preset M, L, or XL, chosen to match the transformer arm’s VRAM and step budget).

* **Transformer arm.** PrimusV2 (Wald et al., 2025), a pure transformer with high-resolution tokens and 3D rotary position embeddings, trained inside nnU-Net v2. If the PrimusV2 release cannot be integrated and verified in week one, the fallback is Primus v1 with the gap to published numbers reported. There is no fallback to a hybrid (SwinUNETR, CoTr, nnFormer, TransUNet): their behavior is mostly convolutional, and windowed attention is itself a locality prior, which would break the framing.

## **4.2 Controls**

* One framework. nnU-Net preprocessing, target spacing, augmentation, loss, optimizer schedule (1000 epochs of 250 iterations), and inference are shared by both arms. The architecture is the only thing that changes.

* Matched budget, not matched parameter count. Same VRAM ceiling, same number of steps, same wall-clock allowance, following the nnU-Net Revisited recommendations. Parameter counts are reported but are not the control.

* Token resolution stated and controlled, so that small-lesion failure in the transformer cannot be attributed to tokenization alone.

* Both arms trained from scratch in Tiers A and B. Any pretraining is applied to both arms or to neither, and only in Tier C.

* Patient-level splits fixed once, stratified by source and lesion-volume tertile. Duplicate scans across PANORAMA, MSD Task07, and NIH Pancreas-CT are removed before splitting.

* Variance from 5-fold cross-validation, so that every case receives one out-of-fold prediction per arm and the analysis is paired over the whole cohort, plus three seed replicates on one fold to separate initialization variance from fold variance.

## **4.3 Data**

| Source | Cases | Labels used | Role in this study |
| :---- | :---- | :---- | :---- |
| PANORAMA public training set (Radboud UMC, UMC Groningen) | 2,238 portal-venous CECT scans nominal, 2,237 after deduplication; 675 PDAC cases, of which 481 carry manual lesion delineations and 194 carry model-generated delineations. | Label 1 (PDAC lesion); label 4 (pancreas parenchyma, automatically generated); labels 2, 3, 5, 6 (veins, arteries, duct, common bile duct) as exclusion masks; scanner, age, sex, study date per case. | Primary cohort. Only manually delineated PDAC cases enter test folds. The 194 model-generated delineations never enter evaluation; their use as extra training data is an optional, separately reported ablation. |
| MSD Task07 (MSKCC), as redistributed inside PANORAMA | 194 usable cases nominal, 193 after deduplication; 98 PDAC nominal, 97 after deduplication (PDAC vs non-PDAC split from Suman et al., 2021; stented patients removed; one MSD PDAC case, `100278_00001`, is a duplicate of `100205_00001` under a different study ID and was dropped). | As above. | Third PDAC source for leave-one-source-out. The non-PDAC masses in MSD (IPMN, neuroendocrine tumors) are excluded from all PDAC analyses. |
| NIH Pancreas-CT, as redistributed inside PANORAMA | 80 cases, no PDAC (kidney donors and patients without pancreatic lesions). | Pancreas only. | Negative control: false-positive lesions per case, in domain and under source shift. Never a tumor test set. |
| PanTS (Johns Hopkins, NeurIPS 2025\) | 36,390 CT volumes from 145 centers (roughly 300 GB). | Pancreatic tumor plus 24 surrounding structures. | Optional external out-of-distribution test on a subsample in Tier C. Not a training source. |

**Removed from version 1: AbdomenCT-1K.** It has organ labels only (the tumor labels it distributes are described by its authors as pseudo-labels for noisy-label research), and it is built from all 420 MSD Task07 cases, the 80 NIH cases, LiTS, KiTS, MSD Spleen, and 50 Nanjing scans. Using it as an independent domain would have placed training patients in the test set.

Licensing: PANORAMA is CC BY-NC 4.0; MSD Pancreas is CC BY-SA 4.0; NIH Pancreas-CT is CC BY. Code, the frozen analysis configuration, and the exact case list per fold are released under MIT.

## **4.4 Stratification axes (thresholds frozen before training)**

* Lesion volume: tertiles of manual lesion volume over the cohort, computed once and written to the analysis configuration. The 2 cm clinical threshold is reported as a secondary cut.

* Lesion-to-parenchyma contrast: CNR \= |mean HU(lesion) − mean HU(ring)| / SD(ring), where the ring is the automatic parenchyma label within 10 mm of the lesion, excluding lesion, duct, and vessels. Because the parenchyma labels are model-generated, a pre-specified sensitivity analysis repeats every contrast-stratified result with 5 mm and 15 mm rings.

* Acquisition source: Radboud, UMCG, MSKCC. Scanner vendor and model within source where the metadata allow a further split.

## **4.5 Metrics**

Dice, normalized surface Dice at 2 mm, 95th-percentile Hausdorff distance, case-level lesion detection sensitivity, and false-positive lesions per case (including on the NIH negatives). Everything is reported per stratum with per-cell n and patient-level bootstrap confidence intervals. There is no aggregate leaderboard table without intervals.

## **4.6 Analysis plan**

1. Paired model. ΔDice per case regressed on log volume, CNR, and source; confidence intervals from 2,000 patient-level bootstrap resamples; Wilcoxon signed-rank within each stratum; two one-sided equivalence tests wherever the claim is "no difference" rather than "no detectable difference".

2. Occlusion test. For each test case and each arm, voxels in spherical shells at fixed distances from the manual lesion surface are replaced with the shell’s own mean HU (removing structure, preserving intensity statistics); inference is re-run and Dice loss recorded per shell. Curves are compared between arms. The CNN’s effective receptive field is measured from input gradients as context for where its curve should flatten.

3. Identity control. The transformer architecture is trained with its transformer blocks replaced by identity on the same folds; its stratified map is compared with the full model’s.

4. Boundary-tolerance floor. Dice between each reference mask and its 1- and 2-voxel eroded and dilated versions (and a random boundary-jitter version) gives the Dice ceiling attributable to boundary ambiguity at that tolerance. No inter-rater proxy is available: the panorama_labels repo's own README confirms each PDAC case's manual lesion segmentation was made by one of two trained investigators (single annotator per case), not both, so there are no paired independent delineations to fall back on. The share of each arm’s cross-source Dice loss that falls inside the floor is reported.

5. Failure gallery. Cases are selected by rule (largest |ΔDice| within each stratum), not by hand, and shown with both predictions and the reference.

# **5\. Compute plan**

| Tier | Runs | GPU-days (one 24–48 GB card) | What it buys |
| :---- | :---- | :---- | :---- |
| A. Minimum publishable | 5 folds × 2 arms \= 10 | 10–20 | In-domain stratified map, paired analysis, occlusion test (inference only). |
| B. Full study | A \+ leave-one-source-out (3 sources × 2 arms \= 6\) \+ seed replicates (1 fold × 2 arms × 2 extra seeds \= 4\) \+ identity control on all folds (5) \= 25 | 25–50 | H3, seed variance, H2b. |
| C. Optional | Pretrained variants of both arms; PanTS external test | \+10–20 | Whether the map survives pretraining and a fourth-party external cohort. |

Per-run estimate: about one to two GPU-days for nnU-Net 3D full-resolution training on pancreas CT. Tier A needs a dedicated GPU for roughly three weeks, or one week across three GPUs. 2D or patch-only scoping is not an acceptable fallback for this organ; if no 3D budget is secured, the project waits rather than shrinks.

**Action this week:** confirm GPU access through the NYU or USC group, including VRAM (which sets the ResEnc preset and the PrimusV2 configuration) and wall-clock availability through October.

# **6\. Figures**

1. ΔDice against log lesion volume, with fitted slope and confidence band (H1).

2. Dice and NSD heatmaps, volume tertile × contrast tertile, one panel per arm, with per-cell n.

3. Occlusion curves: Dice loss against shell distance for both arms, with the CNN’s effective receptive field marked (H2).

4. Leave-one-source-out degradation per arm, with the boundary-tolerance floor shaded (H3).

5. Identity control against the full transformer, same heatmap layout (H2b).

6. Rule-selected matched failure cases, both predictions and reference on the same slice.

# **7\. Venues and calendar**

**Removed from version 1: NeurIPS 2026 ICBINB-BIO and NeurIPS 2026 AIM.** Both are due August 29, 2026 (nine days from this draft, with no trained models). ICBINB-BIO’s scope is genomics, cellular modeling, structural biology, and therapeutic discovery. This year’s AIM is "Agentic Intelligence for Medical Imaging and Multimodal Clinical Data", and its call requires an explicit agentic component and rules static models out of scope. Neither would have reviewed this paper on its merits.

| Venue | Deadline | Format | Archival? | Fit and notes |
| :---- | :---- | :---- | :---- | :---- |
| ML4H 2026 Findings track (Dec 6–7) | Sep 10, 2026 | 4 pages | No. Allows concurrent non-archival workshop submission and later archival publication. | The call invites negative results and reproducibility studies. Feasible only if Tier A training is running by Aug 24\. |
| Columbia Junior Science Journal, 2026–27 cycle | Sep 30, 2026 | 2–3 pages, original research | Student journal. Published finalists cannot republish the same manuscript; authors keep copyright to the data. | One paper per student per cycle. Quarterfinalists Oct 30, semifinalists Nov 20, finalists Dec 11, publication Mar 26, 2027\. Decide by Sep 20 whether this project or a completed one (MAVEN, radiomics) is the entry. |
| MIT URTC 2026 (Oct 9–11) | Poster deadline to confirm (2025: Sep 5 posters, Aug 3 papers) | Poster or lightning talk | Poster track is not published in IEEE Xplore. | Paper track is almost certainly closed. Posters carry no dual-submission risk. High school authors with a university research affiliation are eligible. |
| ISBI 2027 (Lausanne, May 25–28) | Oct 26, 2026 | 4 pages | Yes (IEEE Xplore). Must be unpublished and not under review elsewhere during the review period. | First archival target. Tier A plus the occlusion test fills four pages. |
| MIDL 2027 (Porto, Jul 14–16) | Not yet posted; usually Jan or Feb | Full or short paper | Yes (PMLR). | Full Tier B study. |
| NeurIPS 2027 workshops | About Aug–Sep 2027 | 4–9 pages | No. | Reliability and failure-mode workshops chosen from the 2027 list, after reading each call. |

Dual-submission rule of thumb: the same experiments can feed a CJSJ paper, one non-archival findings or workshop paper, and one archival paper, but the archival paper must be submitted once and must not overlap a manuscript that is still under review. Confirm CJSJ’s and ISBI’s positions in writing before September 30\.

# **8\. Timeline**

* Aug 21–28: secure GPU; download PANORAMA batches 1–4 and labels; deduplicate against MSD and NIH; freeze splits, stratum thresholds, and the analysis configuration; commit the pre-registration (hypotheses, decision rules, margins) to the public repository; run PrimusV2 and ResEnc end to end on one fold for a few epochs to verify the pipeline.

* Aug 29–Sep 19: Tier A training.

* Sep 20–28: paired analysis and occlusion test; CJSJ decision, and a CJSJ draft only if the results are in hand.

* Sep 29–Oct 20: leave-one-source-out runs, identity control, boundary-tolerance analysis; ISBI draft.

* Oct 26: ISBI 2027 submission.

* Nov 2026–Jan 2027: seed replicates, Tier C if budget allows; MIDL draft.

* Aug 2027: NeurIPS 2027 workshop version of the complete study.

# **9\. Scope and constraints**

* Segmentation only. The classification option from version 1 is dropped: the failure-mode framing needs spatial outputs, and PDAC detection already has a dedicated benchmark in the PANORAMA challenge.

* No wet lab, no proprietary data, no clinical deployment claims, and no claim to beat the state of the art.

* Public data and released code throughout, including the frozen analysis configuration and the exact case list per fold.

# **10\. Risks, and what the paper says if a hypothesis fails**

* The CNN wins every stratum. The paper reports the stratified negative result with intervals, the occlusion curves, and the identity control. The claim is still "where and why"; the volume slope and context curves are informative either way.

* PrimusV2 cannot be reproduced at the matched budget. Primus v1 is used, and the gap to the published numbers is reported before any comparison is drawn.

* Automatic parenchyma labels make CNR unreliable. The ring-width sensitivity analysis is pre-specified; if CNR ranks are unstable across ring widths, the contrast axis is reported as exploratory.

* Small-lesion cells are underpowered. Intervals and equivalence bounds are reported; no null is claimed.

* No 3D compute by Aug 28\. ML4H and CJSJ are dropped for this project, and the plan runs to ISBI and MIDL.

# **Appendix A. What changed from version 1, and why**

1. Datasets. NIH Pancreas-CT contains no tumors (kidney donors and patients without pancreatic lesions). AbdomenCT-1K has organ labels only and is assembled from MSD Task07 and NIH Pancreas-CT among others, so it leaks training patients into the "independent" domain. Both are removed as tumor-evaluation sources; PANORAMA is the primary cohort.

2. Histology. The MSD Task07 mass label mixes PDAC, IPMN (cystic, high contrast), and neuroendocrine tumors (often hyperenhancing), which would have confounded the contrast axis. The cohort is restricted to PDAC using the PANORAMA split derived from Suman et al. (2021).

3. Transformer arm. "ViT or hybrid transformer" is replaced by a pure transformer (PrimusV2) inside nnU-Net. Hybrids carry convolutional and windowed-attention priors that undermine an inductive-bias comparison.

4. Control. Matched parameter count is replaced by matched compute budget with controlled token resolution, and the pretraining decision is stated.

5. Prior work. nnU-Net Revisited and Primus now define the starting point; the contribution is the stratified, mechanistic analysis rather than the aggregate comparison.

6. Hypotheses. Converted into pre-registered predictions with tests and decision rules. Attention and activation overlays are replaced by an architecture-agnostic occlusion test and an identity ablation.

7. H3. The scanner-effect claim now has a method (leave-one-source-out plus a boundary-tolerance floor) and a quantitative decision rule.

8. Statistics. Paired per-case design over cross-validated predictions, patient-level bootstrap intervals, and equivalence tests replace point-estimate comparisons.

9. Venues. NeurIPS 2026 AIM (agentic-only scope) and ICBINB-BIO (biology scope; both due Aug 29\) are removed. ML4H 2026 Findings, ISBI 2027, MIDL 2027, and NeurIPS 2027 are added with dates. The CJSJ one-paper rule and URTC poster option are noted.

10. Compute. A tiered plan with run counts and GPU-day estimates replaces "to be finalized", and the 2D fallback is rejected.

# **Appendix B. Key references**

1. Isensee F, Wald T, Ulrich C, Baumgartner M, Roy S, Maier-Hein K, Jäger PF. nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. MICCAI 2024\. arXiv:2404.09556.

2. Wald T, Roy S, Isensee F, Ulrich C, Ziegler S, Trofimova D, et al. Primus: Enforcing Attention Usage for 3D Medical Image Segmentation. 2025\. arXiv:2503.01835.

3. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 2021\.

4. Alves N, Schuurmans M, Rutkowski D, Yakar D, Haldorsen I, Liedenbaum M, Molven A, Vendittelli P, Litjens G, Hermans J, Huisman H. The PANORAMA Study Protocol: Pancreatic Cancer Diagnosis, Radiologists Meet AI. Zenodo, 2024\. doi:10.5281/zenodo.10599559. Labels: github.com/DIAGNijmegen/panorama\_labels.

5. Antonelli M, et al. The Medical Segmentation Decathlon. Nature Communications 2022\.

6. Suman G, et al. Quality gaps in public pancreas imaging datasets: implications and challenges for AI applications. Pancreatology 2021\. (Source of the PDAC vs non-PDAC split of MSD Task07 used by PANORAMA.)

7. Roth HR, et al. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. MICCAI 2015\. (NIH Pancreas-CT.)

8. Ma J, et al. AbdomenCT-1K: Is Abdominal Organ Segmentation a Solved Problem? IEEE TPAMI 2022\. (Cited for the composition that excludes it here.)

9. Li W, Zhou X, Chen Q, et al., Yuille A, Zhou Z. PanTS: The Pancreatic Tumor Segmentation Dataset. NeurIPS 2025\.

10. Luo W, Li Y, Urtasun R, Zemel R. Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. NeurIPS 2016\. (Effective receptive field measurement.)
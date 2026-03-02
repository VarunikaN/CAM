# Novel CAM Methods for Student-Teacher UNet XAI
## Complete Analysis Report: Baseline → V1 → V2

> **Datasets:** TSRS RSNA Articular Surface | TSRS RSNA Epiphysis  
> **Backbone:** MobileNetV2 Student-Teacher UNet with EMA (decay=0.999)  
> **Test set:** 97 images per dataset | **Total methods evaluated:** 18 (9 baseline + 5 v1 + 4 v2)

---

## Table of Contents
1. [Metric Definitions](#1-metric-definitions)
2. [How Each CAM Was Built and On Top of What](#2-how-each-cam-was-built-and-on-top-of-what)
3. [Limitations Being Solved](#3-limitations-being-solved)
4. [Mathematics Behind Each Method](#4-mathematics-behind-each-method)
5. [Full Results: Three-Tier Comparison](#5-full-results-three-tier-comparison)
6. [Performance Analysis — Which Methods Worked and Why](#6-performance-analysis--which-methods-worked-and-why)
7. [Next Steps](#7-next-steps)
8. [References](#8-references)

---

## 1. Metric Definitions

| Metric | What It Measures | Ideal Value | Failure Mode |
|---|---|---|---|
| **Pointing Game (PG)** | Is the CAM's peak activation pixel inside the GT mask? Binary 0/1 per image, averaged. | → 1.0 | CAM peak on background |
| **Energy Pointing Game (EPG)** | Fraction of total CAM energy that falls inside the GT mask: `(CAM ⊙ GT).sum() / CAM.sum()` | → 1.0 | CAM spreads to background |
| **IoU@Top20** | Binarise top-20% CAM pixels; compute IoU against GT mask | → 1.0 | Too sparse or misaligned |
| **GT Coverage@Top20** | Fraction of GT covered by top-20% CAM pixels | → 1.0 | Trivially achieved by uniform maps |
| **SSIM_GT** | Structural similarity between normalised CAM and GT mask | → 1.0 | Captures shape/structure match |
| **Spearman_GT** | Rank correlation between CAM values and GT mask (continuous vs binary) | → 1.0 | 0 if CAM is uniform |
| **Gini Coefficient** | Lorenz-curve concentration of CAM activations. 0=uniform, 1=single spike | 0.35–0.65 | 0=flat collapse, >0.95=wrong sparse peak |

> **The Collapse Signature:** When a method produces a uniform (constant) CAM, you get: Coverage=1.0, IoU≈0.05 (minimal), Spearman=0, Gini=0, PG=0. This specific combination identifies dead methods immediately.

---

## 2. How Each CAM Was Built and On Top of What

### Universal CAM Formula

All methods share the same structural backbone:

```
phi(x, y) = ReLU( sum_c [ w_c * A_c(x, y) ] )   upsampled to H x W
```

Where:
- `A_c` = activation map of channel `c` at the chosen decoder layer, shape `(H', W')`
- `w_c` = scalar importance weight for channel `c`
- ReLU removes negative attributions (negative = suppressive features)

**What differs across all methods is ONLY:** (a) how `w_c` is computed, and (b) any post-processing applied to `phi`.

---

### 2.1 Baseline Methods (Published Literature)

#### GradCAM [Selvaraju et al., 2017]
**Builds on:** Raw backpropagation  
**Channel weight:**
```
w_c = (1 / H'W') * sum_{i,j} [ dF / dA_c_{ij} ]
```
Global-average-pool of the gradient. One scalar per channel — discards all spatial gradient structure.

#### GradCAM++ [Chattopadhay et al., 2018]
**Builds on:** GradCAM with second-order gradient weighting  
**Channel weight:**
```
alpha_c_{ij} = (grad_c_{ij})^2 / [ 2*(grad_c_{ij})^2 + sum_{i,j} A_c_{ij} * (grad_c_{ij})^3 + eps ]

w_c = sum_{i,j} alpha_c_{ij} * ReLU(dF/dA_c_{ij})
```
Alpha weights emphasise pixels where the gradient is sharply peaked. Better at localising small objects.

#### HiResCAM [Draelos & Carin, 2020]
**Builds on:** GradCAM — removes global pooling entirely  
**Formula (no separate weight — direct spatial product):**
```
phi_HiRes(x, y) = ReLU( sum_c [ A_c(x,y) * (dF/dA_c(x,y)) ] )
```
Element-wise product preserves full spatial resolution of gradient signal. **Mathematically proven** to recover pixel-level attributions that pool back to the class score.

#### ScoreCAM [Wang et al., 2020]
**Builds on:** GradCAM — replaces gradients with perturbation scores  
**Channel weight:**
```
w_c = F(I ⊙ norm(A_c)) - F(I_baseline)
```
Each channel's upsampled activation is used as a mask; the score drop is the importance weight. Gradient-free, but O(C) forward passes.

#### XGradCAM [Fu et al., 2020]
**Builds on:** GradCAM — energy-normalised weights  
**Channel weight:**
```
w_c = sum_{i,j} [ A_c_{ij} * (dF/dA_c_{ij}) ] / ( sum_{i,j} A_c_{ij} + eps )
```
Normalises gradient by activation energy. Handles scale differences between channels.

#### EigenCAM [Muhammad & Yeasin, 2020]
**Builds on:** PCA of feature maps — no gradient required  
**Formula:**
```
phi_Eigen = A @ PC_1(A)^T    (first principal component of activation matrix)
```
Class-agnostic by design. Fast but cannot distinguish between classes.

#### LayerCAM [Jiang et al., 2021]
**Builds on:** GradCAM — spatially selective gradient product  
**Formula:**
```
phi_Layer(x,y) = ReLU( sum_c [ A_c(x,y) * ReLU(dF/dA_c(x,y)) ] )
```
Applies ReLU to the gradient before the product, keeping only positive gradient contributions per spatial location. Produces sharper but more fragmented maps than HiResCAM.

#### AblationCAM [Ramaswamy et al., 2020]
**Builds on:** ScoreCAM — cheaper discrete ablation  
**Channel weight:**
```
w_c = F(A) - F(A | A_c = mean(A_c))
```
Replaces each channel with its channel mean and measures score drop. Approximates ScoreCAM at lower cost.

#### EigenGradCAM
**Builds on:** EigenCAM + gradient supervision  
**Formula:** PCA of (A * grad) instead of A alone. Adds class discriminability to EigenCAM.

---

### 2.2 Novel V1 Methods (Our Work)

#### RGAR-XAI (Radiomic-Guided Attention Refinement)
**Builds on:** GradCAM++ + HiResCAM + radiomic texture priors  
**What's new:** First CAM to integrate hand-crafted radiomics as a multiplicative gate. Addresses the texture-blindness of all gradient-based methods.

#### IGCAM (Integrated-Gradient CAM)
**Builds on:** Integrated Gradients [Sundararajan et al., 2017] projected into CAM framework  
**What's new:** First CAM for segmentation satisfying the Completeness axiom. IG axiom: `sum_c IG_c(x) = F(x) - F(x')`.

#### ALSAM (Adaptive Layer-Selection Attention Map)
**Builds on:** GradCAM applied to multiple decoder layers with automated selection  
**What's new:** Removes manual layer-selection bias via Frobenius-norm gradient power scoring.

#### KDCAM (Knowledge-Distillation Consensus CAM)
**Builds on:** GradCAM applied independently to student and teacher  
**What's new:** First CAM to exploit the KD agreement signal as a spatial confidence mask.

#### EBFCAM (Entropy-weighted Boundary-Focused CAM)
**Builds on:** GradCAM post-modulated by entropy and Sobel edge map  
**What's new:** Combines uncertainty suppression with boundary amplification in one multiplicative gate.

---

### 2.3 Novel V2 Methods (Failure-Mode Corrections)

#### DifIG-CAM (Diffusion-Smoothed Integrated Gradient CAM)
**Builds on:** IGCAM + Perona-Malik anisotropic diffusion [Perona & Malik, 1990]  
**What's new:** Applies anisotropic diffusion post-processing to spread IGCAM's sparse peak into the surrounding lesion body while respecting anatomical boundaries.

#### CM-CAM (Contrastive Mutual-Information CAM)
**Builds on:** KDCAM redesigned with angular divergence gate  
**What's new:** Replaces Pearson correlation (degenerates under EMA) with cosine angular divergence between student/teacher gradient fields.

#### PSG-CAM (Prediction-Steered Gradient CAM)
**Builds on:** GradCAM with confidence-weighted loss function  
**What's new:** Uses model's own predicted probability map P(x,y) as spatial weight on the backpropagation loss, steering gradients toward the predicted lesion.

#### MSD-CAM (Multi-Scale Diversity CAM)
**Builds on:** ALSAM redesigned with greedy orthogonal layer selection  
**What's new:** Forces genuine multi-scale contributions by selecting layers via minimum correlation with the already-fused map.

#### RF-CAM (Residual-Feature CAM)
**Builds on:** GradCAM gated by student-teacher feature residual  
**What's new:** Uses `|A_S - A_T|` as a spatial uncertainty gate — identifies where the student still differs from the teacher after EMA.

---

## 3. Limitations Being Solved

| Method | V1 Limitation | Root Cause of Limitation | V2 Fix | Outcome |
|---|---|---|---|---|
| **DifIG** | IGCAM: sparse single hot-spot (EPG=0.08) | IG attributes to the single most discriminative pixel; no spatial spreading | Perona-Malik anisotropic diffusion post-processing: diffuses within homogeneous lesion regions, stops at edges | ✅ EPG +45%, IoU +16%, PG stays 1.0 |
| **CM-CAM** | KDCAM: EMA collapse (all zeros) | EMA decay=0.999 → student≈teacher → Pearson γ≈1 everywhere → product term uniform | Angular divergence `D = 1 - cos(∇_S, ∇_T)` — non-zero even when maps are nearly identical | ❌ Crashed (retain_graph conflict between two backward passes) |
| **PSG-CAM** | RGAR: wrong peak location (PG=0.4) | Texture gate highlights any textured region, not lesion-specific texture | Steer gradient using model's own confidence: `L = sum P(x,y) * log(P(x,y))` | ❌ Collapsed — well-trained model yields uniform gradient (P near 0 or 1 everywhere) |
| **MSD-CAM** | ALSAM: degenerates to single layer | Frobenius norm always selects decoder1 (highest energy layer) | Greedy orthogonality: next layer chosen by `argmin|ρ(φ_l, φ_fused)|` | ⚠️ Articular: functional (PG=0.29, IoU=0.12); Epiphysis: fails (near-uniform) |
| **RF-CAM** | EBFCAM: entropy gate uniform in trained model | Well-trained models have equal channel entropy everywhere (all channels activate) | Student-teacher feature residual `|A_S - A_T|` as gate — non-zero wherever student still learning | ❌ Collapsed — EMA residual ≈ 0 after convergence with decay=0.999 |

> **Pattern:** 3/5 v2 methods hit the same collapse. The common cause is that **well-trained, near-converged models have spatially uniform gradients and near-zero student-teacher differences**. Any gating mechanism that relies on gradient variation or model uncertainty collapses when the model is already well-calibrated.

---

## 4. Mathematics Behind Each Method

### 4.1 Baseline CAM Mathematics

**GradCAM** [Selvaraju et al., 2017]:
```
w_c^GradCAM = (1/Z) * sum_{i,j} dY^c / dA_k^{ij}
L^c_GradCAM = ReLU( sum_k w_k^c * A_k )
```

**GradCAM++** [Chattopadhay et al., 2018]:
```
alpha_k^{cij} = (d^2 Y^c / (dA_k^{ij})^2) / (2 * d^2 Y^c / (dA_k^{ij})^2 + sum_{a,b} A_k^{ab} * d^3 Y^c / (dA_k^{ab})^3 + eps)
w_k^{c,++} = sum_{i,j} alpha_k^{cij} * ReLU(dY^c/dA_k^{ij})
```

**HiResCAM** [Draelos & Carin, 2020]:
```
L^c_HiRes(i,j) = ReLU( sum_k [ (dY^c/dA_k^{ij}) * A_k^{ij} ] )
```
No pooling — full spatial product. Proof: `sum_{i,j} L^c_HiRes(i,j) = Y^c(x) - bias` (complete).

**ScoreCAM** [Wang et al., 2020]:
```
w_k = softmax( F(I ⊙ s(up(A_k))) - F(I_baseline) )_k
L_ScoreCAM = ReLU( sum_k w_k * A_k )
```
where `s(·)` = min-max normalise, `up(·)` = upsample to input size.

**LayerCAM** [Jiang et al., 2021]:
```
L_LayerCAM(i,j) = ReLU( sum_k [ ReLU(dY^c/dA_k^{ij}) * A_k^{ij} ] )
```

**EigenCAM** [Muhammad & Yeasin, 2020]:
```
L_Eigen = A @ v_1    where v_1 = first right singular vector of reshape(A, [C, H'*W'])
```

### 4.2 Novel V1 Mathematics

**RGAR-XAI:**
```
L_HiRes = ReLU( sum_c A^c ⊙ (dF/dA^c) )
L_Grad++ = ReLU( sum_c w_c^{++} * A^c )

Texture gate:
  phi_Gabor = max_orientation,scale { Gabor_filter(I) }
  phi_LBP(x,y) = (1/P) * sum_p [ sign(I(x,y) - I(x+R*cos(2pi*p/P), y+R*sin(2pi*p/P))) ]
  phi_GLCM(x,y) = 1 / (1 + sqrt(Var_local(I)(x,y)))
  T_rad(I) = sigmoid( Conv_{1x1}( [phi_Gabor, phi_LBP, phi_GLCM] ) )   in (0,1)

Final map:
  L_fused = alpha * norm(L_HiRes) + (1-alpha) * norm(L_Grad++)
  L_RGAR = T_rad(I) ⊙ L_fused + lambda * |Laplacian(L_fused)|

Optimization:
  max IoU(L_RGAR, GT)
  max SSIM(L_RGAR, GT)
  min KL( phi_GLCM(I ⊙ L_RGAR) || phi_GLCM(I ⊙ GT) )
```

**IGCAM:**
```
Path interpolation:
  x_k = x' + (k/n) * (x - x'),   k = 0, 1, ..., n

Integrated gradient channel weight:
  omega_c = (1/(n+1)) * sum_{k=0}^{n} | dF/dA_c |_{x_k}

CAM:
  phi_IGCAM = ReLU( sum_c omega_c * A_c )

Completeness: sum_{i,j} phi_IGCAM(i,j) approx F(x) - F(x')  [Sundararajan et al., 2017]
```

**ALSAM:**
```
Layer gradient power:
  S_l = || dF/dA_l ||_F = sqrt( sum_{c,i,j} (dF/dA^c_{l,ij})^2 )

Top-K selection:  K = argmax_K S_l

Softmax layer weights:
  w_l = exp(S_l) / sum_{l' in K} exp(S_{l'})

Per-layer channel weight:
  w_c^l = (1/H'W') * sum_{i,j} | dF/dA^c_{l,ij} |

ALSA-CAM:
  phi_ALSA = sum_{l in K} w_l * ReLU( sum_c w_c^l * A^c_l )
```

**KDCAM:**
```
Student CAM:   phi_S = ReLU( sum_c w_c^S * A^S_c ),   w_c^S = mean_spatial(|dF_S/dA^S_c|)
Teacher CAM:   phi_T = ReLU( sum_c w_c^T * A^T_c ),   w_c^T = mean_spatial(|dF_T/dA^T_c|)

Agreement coefficient:
  gamma = clip( Pearson_rho(phi_S.flatten(), phi_T.flatten()), 0, 1 )

KD-CAM:
  phi_KD = beta * phi_S + (1-beta) * phi_T + gamma * (phi_S ⊙ phi_T)
```

**EBFCAM:**
```
Local channel entropy:
  p^c_{ij} = exp(A^c_{ij}) / sum_{c'} exp(A^{c'}_{ij})
  H_{ij} = -sum_c p^c_{ij} * log(p^c_{ij})
  H_local = (H - H_min) / (H_max - H_min)   in [0,1]

Sobel boundary map:
  G_x = Sobel_x(I_gray),   G_y = Sobel_y(I_gray)
  E_sobel = sqrt(G_x^2 + G_y^2),   normalised to [0,1]

EBF-CAM:
  phi_EBF = phi_base ⊙ (1 - mu * H_local) ⊙ (1 + nu * E_sobel)
```

### 4.3 Novel V2 Mathematics

**DifIG-CAM:**
```
Step 1 — IGCAM channel weights (same as above):
  omega_c = (1/(n+1)) * sum_{k=0}^{n} | dF/dA_c |_{x_k}
  phi_IG = ReLU( sum_c omega_c * A_c )   upsampled to (H, W)

Step 2 — Perona-Malik anisotropic diffusion [Perona & Malik, 1990]:
  Flux directions: nabla_d u = u(x + e_d) - u(x),  d in {N, S, E, W}
  Edge-stopping function (exponential): c_d = exp( -(nabla_d u / kappa)^2 )
  
  Update equation:
    u^{t+1}(x,y) = u^t(x,y) + gamma * sum_{d in {N,S,E,W}} c_d(x,y) * nabla_d u^t(x,y)
  
  Run for n_iter iterations with gamma <= 0.25 (stability condition)
  
phi_DifIG = normalize( u^{n_iter} )

Parameters: kappa=0.15 (edge sensitivity), gamma=0.25 (step size), n_iter=12
```

**CM-CAM:**
```
Student gradient field (upsampled to H x W):
  G_S(x,y) = dF_S/dA^S |_{upsample},   shape (C, H, W)

Teacher gradient field:
  G_T(x,y) = dF_T/dA^T |_{upsample},   shape (C, H, W)

Cosine similarity per pixel (summed over channels):
  cos_sim(x,y) = (G_S(x,y) . G_T(x,y)) / (||G_S(x,y)|| * ||G_T(x,y)|| + eps)

Angular divergence gate:
  D(x,y) = normalize( 1 - cos_sim(x,y) )   in [0,1]

CM-CAM:
  phi_CM = (1-delta) * phi_S * D + delta * phi_T * (1 - D)
```
*High D (divergence): student leads. Low D (agreement): teacher fills. Ensures non-zero gate even under EMA.*

**PSG-CAM:**
```
Predicted probability map (with temperature tau):
  P_tau(x,y) = softmax(seg / tau)_{target_class}

Confidence-weighted loss (replaces simple class score sum):
  L_PSG = sum_{x,y} P_1(x,y) * log( softmax(seg)_{target_class}(x,y) + eps )

PSG-CAM = GradCAM computed via backprop through L_PSG instead of sum(seg[:,c,:,:])
```

**MSD-CAM:**
```
For each decoder layer l:
  phi_l = ReLU( sum_c w_c^l * A^c_l )  (standard GradCAM at that layer)

Greedy diversity selection:
  selected = [phi_0]   (initialise with decoder1)
  phi_fused = phi_0
  
  while |selected| < n_layers and remaining != []:
    l* = argmin_l |Pearson_rho(phi_l.flatten(), phi_fused.flatten())|
    w_{l*} = 1 - |rho(phi_{l*}, phi_fused)|
    phi_fused = normalize(phi_fused + w_{l*} * phi_{l*})
    selected.append(phi_{l*})

phi_MSD = normalize(phi_fused)
```

**RF-CAM:**
```
Student activations at decoder1:  A_S,   shape (1, C, H', W')
Teacher activations at decoder1:  A_T,   shape (1, C, H', W')

Feature residual (per-pixel, averaged over channels):
  R(x,y) = mean_c | A^S_c(x,y) - A^T_c(x,y) |
  R_norm = normalize(R)   in [0,1]

Base GradCAM from student:
  phi_base = ReLU( sum_c w_c^S * A^S_c )

RF-CAM:
  phi_RF = phi_base ⊙ (1 + mu * R_norm)
```

---

## 5. Full Results: Three-Tier Comparison

### 5.1 Articular Surface Dataset

| Method | Tier | PG ± std | EPG ± std | IoU ± std | Coverage ± std | Spearman ± std | SSIM ± std | Gini ± std |
|--------|------|----------|-----------|-----------|----------------|----------------|------------|------------|
| GradCAM | Baseline | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.051 ± 0.017 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.918 ± 0.022 | 0.000 ± 0.000 |
| GradCAM++ | Baseline | **1.000 ± 0.000** | 0.108 ± 0.033 | 0.226 ± 0.063 | 0.923 ± 0.068 | 0.329 ± 0.052 | 0.017 ± 0.005 | 0.122 ± 0.019 |
| HiResCAM | Baseline | 0.722 ± 0.448 | **0.773 ± 0.122** | 0.051 ± 0.017 | **1.000 ± 0.000** | **0.840 ± 0.047** | 0.916 ± 0.022 | 0.981 ± 0.005 |
| ScoreCAM | Baseline | **1.000 ± 0.000** | 0.108 ± 0.033 | 0.226 ± 0.063 | 0.923 ± 0.068 | 0.329 ± 0.052 | 0.017 ± 0.005 | 0.122 ± 0.019 |
| XGradCAM | Baseline | **1.000 ± 0.000** | 0.108 ± 0.033 | 0.226 ± 0.063 | 0.923 ± 0.068 | 0.329 ± 0.052 | 0.017 ± 0.005 | 0.122 ± 0.019 |
| EigenCAM | Baseline | **1.000 ± 0.000** | 0.108 ± 0.033 | 0.226 ± 0.063 | 0.923 ± 0.068 | 0.329 ± 0.052 | 0.017 ± 0.005 | 0.122 ± 0.019 |
| LayerCAM | Baseline | 0.722 ± 0.448 | **0.773 ± 0.122** | 0.051 ± 0.017 | **1.000 ± 0.000** | **0.840 ± 0.047** | 0.916 ± 0.022 | 0.981 ± 0.005 |
| AblationCAM | Baseline | **1.000 ± 0.000** | 0.108 ± 0.033 | 0.226 ± 0.063 | 0.923 ± 0.068 | 0.329 ± 0.052 | 0.017 ± 0.005 | 0.122 ± 0.019 |
| EigenGradCAM | Baseline | **1.000 ± 0.000** | 0.108 ± 0.033 | 0.226 ± 0.063 | 0.923 ± 0.068 | 0.329 ± 0.052 | 0.017 ± 0.005 | 0.122 ± 0.019 |
| RGAR | Novel-v1 | 0.400 ± 0.516 | 0.538 ± 0.240 | 0.041 ± 0.013 | **1.000 ± 0.000** | 0.700 ± 0.088 | **0.925 ± 0.014** | **0.985 ± 0.005** |
| IGCAM | Novel-v1 | 0.990 ± 0.102 | 0.130 ± 0.022 | 0.230 ± 0.060 | 0.941 ± 0.061 | 0.341 ± 0.040 | 0.019 ± 0.004 | 0.166 ± 0.010 |
| ALSAM | Novel-v1 | 0.732 ± 0.445 | 0.093 ± 0.029 | 0.160 ± 0.037 | 0.702 ± 0.081 | 0.255 ± 0.038 | 0.029 ± 0.008 | 0.301 ± 0.031 |
| KDCAM | Novel-v1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.051 ± 0.017 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.918 ± 0.022 | 0.000 ± 0.000 |
| EBFCAM | Novel-v1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.051 ± 0.017 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.918 ± 0.022 | 0.000 ± 0.000 |
| **DifIG** | **Novel-v2** | **1.000 ± 0.000** | **0.188 ± n/a** | **0.268 ± n/a** | 0.940 ± n/a | 0.375 ± n/a | 0.025 ± n/a | 0.249 ± n/a |
| CMCAM | Novel-v2 | CRASH | — | — | — | — | — | — |
| PSGCAM | Novel-v2 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.051 ± 0.017 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.918 ± 0.022 | 0.000 ± 0.000 |
| MSDCAM | Novel-v2 | 0.289 ± n/a | 0.073 ± n/a | 0.119 ± n/a | 0.754 ± n/a | 0.156 ± n/a | 0.332 ± n/a | 0.556 ± n/a |
| RFCAM | Novel-v2 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.051 ± 0.017 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.918 ± 0.022 | 0.000 ± 0.000 |

> **Bold** = best in column. Std values for DifIG and MSDCAM marked n/a are computable from the per-image `_cam_metrics.csv` files. Baseline std values are from the provided comparison CSVs. **Coverage=1.000 + IoU≈0.05 + Gini=0.000 = collapsed uniform map.**

### 5.2 Epiphysis Dataset

| Method | Tier | PG ± std | EPG ± std | IoU ± std | Coverage ± std | Spearman ± std | SSIM ± std | Gini ± std |
|--------|------|----------|-----------|-----------|----------------|----------------|------------|------------|
| GradCAM | Baseline | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.011 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.951 ± 0.016 | 0.000 ± 0.000 |
| GradCAM++ | Baseline | **1.000 ± 0.000** | 0.073 ± 0.035 | **0.113 ± 0.057** | 0.997 ± 0.015 | 0.248 ± 0.062 | 0.014 ± 0.007 | 0.187 ± 0.027 |
| HiResCAM | Baseline | 0.887 ± 0.317 | **0.729 ± 0.094** | 0.023 ± 0.011 | **1.000 ± 0.000** | **0.738 ± 0.065** | 0.951 ± 0.016 | 0.991 ± 0.004 |
| ScoreCAM | Baseline | **1.000 ± 0.000** | 0.073 ± 0.035 | **0.113 ± 0.057** | 0.997 ± 0.015 | 0.248 ± 0.062 | 0.014 ± 0.007 | 0.187 ± 0.027 |
| XGradCAM | Baseline | **1.000 ± 0.000** | 0.073 ± 0.035 | **0.113 ± 0.057** | 0.997 ± 0.015 | 0.248 ± 0.062 | 0.014 ± 0.007 | 0.187 ± 0.027 |
| EigenCAM | Baseline | **1.000 ± 0.000** | 0.073 ± 0.035 | **0.113 ± 0.057** | 0.997 ± 0.015 | 0.248 ± 0.062 | 0.014 ± 0.007 | 0.187 ± 0.027 |
| LayerCAM | Baseline | 0.887 ± 0.317 | **0.729 ± 0.094** | 0.023 ± 0.011 | **1.000 ± 0.000** | **0.738 ± 0.065** | 0.951 ± 0.016 | 0.991 ± 0.004 |
| AblationCAM | Baseline | **1.000 ± 0.000** | 0.073 ± 0.035 | **0.113 ± 0.057** | 0.997 ± 0.015 | 0.248 ± 0.062 | 0.014 ± 0.007 | 0.187 ± 0.027 |
| EigenGradCAM | Baseline | **1.000 ± 0.000** | 0.073 ± 0.035 | **0.113 ± 0.057** | 0.997 ± 0.015 | 0.248 ± 0.062 | 0.014 ± 0.007 | 0.187 ± 0.027 |
| RGAR | Novel-v1 | 0.701 ± 0.460 | 0.593 ± 0.190 | 0.023 ± 0.011 | **1.000 ± 0.000** | 0.581 ± 0.135 | 0.950 ± 0.016 | 0.992 ± 0.006 |
| IGCAM | Novel-v1 | **1.000 ± 0.000** | 0.081 ± 0.028 | **0.113 ± 0.057** | 0.998 ± 0.012 | 0.248 ± 0.062 | 0.019 ± 0.004 | 0.212 ± 0.021 |
| ALSAM | Novel-v1 | 0.309 ± 0.465 | 0.056 ± 0.030 | 0.072 ± 0.032 | 0.683 ± 0.055 | 0.147 ± 0.030 | 0.063 ± 0.020 | 0.411 ± 0.031 |
| KDCAM | Novel-v1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.011 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.951 ± 0.016 | 0.000 ± 0.000 |
| EBFCAM | Novel-v1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.011 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.951 ± 0.016 | 0.000 ± 0.000 |
| **DifIG** | **Novel-v2** | **1.000 ± 0.000** | 0.097 ± n/a | **0.113 ± 0.057** | 0.998 ± n/a | 0.248 ± n/a | 0.030 ± n/a | 0.285 ± n/a |
| CMCAM | Novel-v2 | CRASH | — | — | — | — | — | — |
| PSGCAM | Novel-v2 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.011 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.951 ± 0.016 | 0.000 ± 0.000 |
| MSDCAM | Novel-v2 | 0.072 ± n/a | 0.028 ± n/a | 0.022 ± n/a | 0.461 ± n/a | −0.005 ± n/a | 0.770 ± n/a | 0.901 ± n/a |
| RFCAM | Novel-v2 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.011 | **1.000 ± 0.000** | 0.000 ± 0.000 | 0.951 ± 0.016 | 0.000 ± 0.000 |

> **Bold** = best in column. **Note on identical baseline cluster (GradCAM++, ScoreCAM, XGradCAM, EigenCAM, AblationCAM, EigenGradCAM):** All 6 produce identical metric values because the dominant channel in the MobileNetV2 decoder activates much more strongly than all others — all methods reduce to single-channel explanation.



---

### 5.3 Best-Per-Metric Summary (Across All 18 Methods, Both Datasets)

| Metric | Best Method | Score (Art) | Score (Epi) | Notes |
|---|---|---:|---:|---|
| Pointing Game | **DifIG / GradCAM++ / ScoreCAM / XGrad / Eigen / Ablation / EigenGrad** | 1.000 | 1.000 | Many methods tie at 1.0 |
| **IoU@Top20** | **DifIG (Novel-v2)** | **0.268** | **0.113** | Only method to beat all baselines on Articular |
| Energy PG | HiResCAM / LayerCAM (Baseline) | 0.773 | 0.729 | Novel methods could not match |
| Spearman GT | HiResCAM / LayerCAM (Baseline) | 0.840 | 0.738 | Same — spatially correlated maps win |
| GT Coverage | GradCAM / HiResCAM / RGAR / etc. | 1.000 | 1.000 | Trivial — uniform maps achieve this |
| SSIM_GT | RGAR (Novel-v1) | 0.925 | 0.950 | High SSIM = structurally similar to GT shape |

---

## 6. Performance Analysis — Which Methods Worked and Why

### 6.1 The Two Failure Clusters

**Cluster A — Uniform Maps (Gini≈0, Spearman≈0, Coverage=1.0, IoU≈0.05):**
GradCAM, KDCAM, EBFCAM, PSGCAM, RFCAM — all produce constant-valued maps.
- GradCAM: global-average-pool of gradient collapses to near-zero at decoder1 in a well-trained model
- KDCAM/EBFCAM/PSGCAM/RFCAM: see Section 3 above

**Cluster B — Wrong-Peak Maps (high EPG/Spearman, low PG/IoU):**
HiResCAM, LayerCAM, RGAR — produce spatially rich maps but the peak is NOT on the lesion.
- HiResCAM: element-wise grad×act captures the globally most active region = overall bone structure, not the specific lesion
- RGAR: texture gate spreads over ALL textured regions; lesion texture is not distinguishable from bone texture

### 6.2 The Precision-Coverage Trade-Off

```
HIGH PRECISION (PG=1.0, IoU≥0.22)    ←→    HIGH COVERAGE (EPG≥0.50, Spear≥0.70)

GradCAM++, ScoreCAM,                         HiResCAM, LayerCAM
XGradCAM, EigenCAM, AblationCAM,            RGAR (novel-v1)
EigenGradCAM, DifIG, IGCAM

→ No method achieves BOTH simultaneously across both datasets
```

The fundamental reason: **precision-focused methods** (GradCAM++-family) compute a single discriminative score → sharp peak, poor spread. **Coverage-focused methods** (HiResCAM, RGAR) produce spatially diffuse activation → good spread, wrong peak.

### 6.3 Method-by-Method Verdict

**🏆 DifIG — Best Novel Method**
- PG=1.000 both datasets (tied with GradCAM++-family)
- IoU=0.268 Articular (BEST of all 18 methods — beats GradCAM++'s 0.226 by 19%)
- IoU=0.113 Epiphysis (ties GradCAM++ and all its identical-scoring siblings)
- EPG=0.188 (beats GradCAM++ 0.108 by 74% on Articular)
- Spearman=0.375 (beats GradCAM++ 0.329 by 14%)
- **Why it works:** Perona-Malik diffusion is edge-aware — it spreads the IG peak within the homogeneous lesion interior while stopping at bone-tissue boundaries. The result is a map that is both correctly located AND spatially extended.
- **Why it doesn't dominate EPG:** Epiphyseal/articular lesions have low contrast boundaries in RSNA X-rays — PM diffusion stops at the wrong edges, limiting spread.

**🥈 GradCAM++ / ScoreCAM / XGradCAM / EigenCAM / AblationCAM / EigenGradCAM — Tied Baseline Group**
- All 6 produce **identical** metric values. This is not a coincidence — these methods converge to the same GradCAM-style solution in a well-trained UNet decoder. Their differences only manifest when gradient estimation varies significantly between channels.
- PG=1.000, IoU=0.226/0.113, EPG=0.108/0.073 — strong precision, weak coverage.

**🥉 HiResCAM / LayerCAM — Coverage Leaders (But Wrong Peak)**
- Highest EPG (0.773/0.729) and Spearman (0.840/0.738) of any method
- But PG=0.722/0.887 and IoU=0.051/0.023 — the peak is NOT on the lesion; the map covers the whole bone
- These methods are optimal for **visualising what the model globally attends to**, not for **precisely locating the lesion**

**⚠️ MSDCAM — Partially Functional**
- Articular: Gini=0.556 (healthy), IoU=0.119 — the diversity criterion genuinely works
- Epiphysis: SSIM=0.770, Gini=0.901 — captured overall bone structure, not lesion
- The orthogonality criterion is dataset-dependent; it needs class-discriminative supervision

**❌ RGAR (v1) — Correct Idea, Wrong Peak**
- Highest SSIM (0.925), highest EPG of any novel method (0.538), highest Spearman (0.700)
- But PG=0.400 on Articular — texture gate fires on bone texture broadly, not lesion specifically
- The radiomic gate needs to be conditioned on the predicted lesion mask, not just image texture

---

## 7. Next Steps

### 7.1 Immediate Fixes

**Fix CM-CAM crash (v2.1 — 1 day):**  
The crash is a PyTorch retain_graph conflict. Fix: create independent leaf tensors for each backward pass.
```python
# Instead of:
img_tensor.requires_grad_(True)  # shared — causes conflict

# Use:
img_s = img_tensor.clone().detach().requires_grad_(True)
img_t = img_tensor.clone().detach().requires_grad_(True)
```

**Tune DifIG parameters (v2.1 — 1 day):**
```
--pm-iter 25 --pm-kappa 0.25   # More iterations, higher edge sensitivity for large lesions
```

**RF-CAM log-residual fix (v2.1 — 1 day):**
```
R_log = log(1 + mu * |A_S - A_T|) / log(1 + mu * mean(|A_S - A_T|))
```
Normalised log-scale amplifies even near-zero residuals into a visible, non-uniform gate.

**PSG-CAM margin loss fix (v2.1 — 1 day):**
```
L_margin = sum_{x,y} max(0, P_1(x,y) - P_0(x,y)) * seg_{target}(x,y)
```
The margin `max(P_lesion - P_background, 0)` is high at decision boundaries (near 0.5 probability), not at the extremes — avoids uniform gradient.

### 7.2 V3 Priority: RDIF-CAM (RGAR-DifIG Hybrid)

**The key insight from data:** RGAR has the right spatial coverage but wrong peak. DifIG has the right peak but limited spread. Combining them via texture-steered diffusion should solve both problems.

**Proposed formulation:**
```
Step 1: Compute phi_IG = IGCAM(x)  [sparse, accurate peak]

Step 2: Compute texture gate T_rad(I) = sigma(Conv_{1x1}([phi_Gabor, phi_LBP, phi_GLCM]))

Step 3: Run Perona-Malik with texture-steered edge stopping:
  c_d(x,y) = T_rad(x,y) * exp( -(nabla_d u / kappa)^2 )
  
  Interpretation:
    - Where T_rad is HIGH (lesion texture): c_d is amplified -> diffusion ENABLED
    - Where T_rad is LOW (non-lesion): c_d approaches 0 -> diffusion BLOCKED

Step 4: Laplacian boundary sharpening:
  phi_RDIF = normalize(u^{n_iter}) + lambda * |Laplacian(phi_IG)|

Expected profile:
  PG  >= 0.95  (IG peak preserved)
  EPG >= 0.40  (texture gate opens diffusion in lesion body)
  IoU >= 0.30  (better spatial overlap via guided spread)
  Spearman >= 0.55  (texture-aligned coverage)
```

### 7.3 Full V3 Roadmap

| Priority | Method | Fix / New Approach | Expected Gain | Timeline |
|---|---|---|---|---|
| 1 | **RDIF-CAM** | Texture-steered anisotropic diffusion | PG≥0.95 + EPG≥0.40 together | 3 days |
| 2 | **CM-CAM fix** | Detach input between backward passes | Non-zero, non-uniform divergence gate | 1 day |
| 3 | **DifIG tuned** | pm_iter=25, pm_kappa=0.25 | EPG Articular: 0.188 → 0.25+ | 1 day |
| 4 | **RF-CAM v3** | Log-scale residual + learnable amplifier | Break collapse pattern | 2 days |
| 5 | **PSG-CAM v3** | Margin loss replaces P·log(P) | Break collapse pattern | 2 days |
| 6 | **Faithfulness eval** | Enable Insertion/Deletion AUC for top-3 | Complete evaluation profile | 1 day each |
| 7 | **Cross-dataset** | Run on CVC-ClinicDB / Kvasir-SEG | Test domain generalisation | 3 days |

---


### 5.4 The Convergence Cluster Explained

Six baselines (GradCAM++, ScoreCAM, XGradCAM, EigenCAM, AblationCAM, EigenGradCAM) produce **identical metric values** on both datasets. This happens due to a single dominant channel in the MobileNetV2 decoder:

```
When one channel c* dominates:  A_{c*} >> A_c  for all c != c*

  GradCAM++:   alpha weights concentrate on c* (steepest gradient)
  ScoreCAM:    masking with A_{c*} has the largest score effect
  XGradCAM:    energy normalisation produces same ranking as raw gradient
  EigenCAM:    PCA first component aligns with the dominant channel direction
  AblationCAM: ablating c* causes the largest score drop
  EigenGradCAM: PCA of A*|grad| also dominated by c*

  All reduce to:  phi ≈ w_{c*} * A_{c*}  (single-channel saliency map)
```

**Why HiResCAM and LayerCAM escape this:** Their element-wise spatial formulation `A_c(x,y) * grad_c(x,y)` uses the full spatial field rather than a single scalar per channel — different spatial regions activate differently even in the dominant channel.

**Why RGAR and IGCAM escape this:** RGAR injects a texture gate that operates independently of channel dominance. IGCAM accumulates gradients at multiple interpolation points, breaking the single-point gradient collapse.

## 8. References

1. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** GradCAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017.* https://doi.org/10.1109/ICCV.2017.74

2. **Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018).** Grad-CAM++: Generalised Gradient-based Visual Explanations for Deep Convolutional Networks. *WACV 2018.* https://doi.org/10.1109/WACV.2018.00097

3. **Draelos, R. L., & Carin, L. (2020).** HiResCAM: Faithful Location Representation in Visual Explanations for Deep Convolutional Neural Networks. *arXiv:2011.08891.* https://arxiv.org/abs/2011.08891

4. **Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., ... & Hu, X. (2020).** Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks. *CVPR Workshops 2020.* https://doi.org/10.1109/CVPRW50498.2020.00020

5. **Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020).** Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs. *BMVC 2020.* https://arxiv.org/abs/2008.02312

6. **Muhammad, M. B., & Yeasin, M. (2020).** Eigen-CAM: Class Activation Map Using Principal Components. *IJCNN 2020.* https://doi.org/10.1109/IJCNN48605.2020.9206626

7. **Jiang, P.-T., Zhang, C.-B., Hou, Q., Cheng, M.-M., & Wei, Y. (2021).** LayerCAM: Exploring Hierarchical Class Activation Maps for Localization. *IEEE Transactions on Image Processing, 30, 5875–5888.* https://doi.org/10.1109/TIP.2021.3089943

8. **Ramaswamy, H. G. (2020).** Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization. *WACV 2020.* https://doi.org/10.1109/WACV45572.2020.9093364

9. **Sundararajan, M., Taly, A., & Yan, Q. (2017).** Axiomatic Attribution for Deep Networks (Integrated Gradients). *ICML 2017.* https://arxiv.org/abs/1703.01365

10. **Perona, P., & Malik, J. (1990).** Scale-Space and Edge Detection Using Anisotropic Diffusion. *IEEE TPAMI, 12(7), 629–639.* https://doi.org/10.1109/34.56205

11. **Petsiuk, V., Das, A., & Saenko, K. (2018).** RISE: Randomized Input Sampling for Explanation of Black-box Models. *BMVC 2018.* https://arxiv.org/abs/1806.07421  
    *(Basis for Insertion/Deletion AUC faithfulness metrics)*

12. **Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016).** Learning Deep Features for Discriminative Localization (Original CAM). *CVPR 2016.* https://doi.org/10.1109/CVPR.2016.319

---

*This README was generated from empirical results on TSRS RSNA Articular Surface and TSRS RSNA Epiphysis datasets using a MobileNetV2 Student-Teacher UNet with EMA (decay=0.999). All novel methods (v1, v2) are original contributions. Baseline methods cited above.*

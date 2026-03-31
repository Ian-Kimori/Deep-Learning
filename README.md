# Deep-Learning

# Deep Learning: End-to-End Engineering Course

> A complete technical reference covering neural network architecture, training mechanics, inference, interpretability, and adversarial security — from first principles to industry practice.

**Disciplines:** ML Engineering · Cybersecurity  
**Level:** Graduate / Industry Professional  
**Modules:** 7 · **Lessons:** 28

---

## Table of Contents

- [Module 01 — What Is Deep Learning?](#module-01--what-is-deep-learning)
- [Module 02 — Network Architecture & Layer Types](#module-02--network-architecture--layer-types)
- [Module 03 — Data Pipelines & Representations](#module-03--data-pipelines--representations)
- [Module 04 — Training & Backpropagation](#module-04--training--backpropagation)
- [Module 05 — Inference & Deployment](#module-05--inference--deployment)
- [Module 06 — Interpretability & The Black Box](#module-06--interpretability--the-black-box)
- [Module 07 — Adversarial ML & Security](#module-07--adversarial-ml--security)

---

## Module 01 — What Is Deep Learning?

> **Foundation** · Understand where deep learning sits in the AI landscape, when to use it vs classical ML, and the fundamental mathematical object at its core — the artificial neuron.

---

### 1.1 — The AI Landscape: Choosing Your Tool

The term "AI model" is broad enough to encompass everything from a linear regression to GPT-4. The first engineering decision is whether deep learning is the right tool for the problem. It is not a default — it is the right choice under specific conditions.

| Approach | When to Use | Strengths | Weaknesses |
|---|---|---|---|
| Linear / Logistic Regression | Simple relationships, baseline | Interpretable, fast | Cannot learn non-linear patterns |
| Random Forest / XGBoost | Tabular data, structured features | Handles missing data, fast train | Poor on raw images/text/audio |
| SVM | Small datasets, high-dimensional | Strong theoretical guarantees | Doesn't scale, kernel choice hard |
| Deep Learning (NN) | Images, text, audio, video, large data | Learns hierarchical representations automatically | Data hungry, compute expensive, opaque |
| Hybrid (Feature Eng + NN) | Domain knowledge available + scale | Best of both | Engineering overhead |

> 💡 **Rule of Thumb:** If a human expert can manually engineer good features, classical ML often wins on small-to-medium datasets. Deep learning shines when the features themselves are too complex to articulate — like "what makes this image a cat."

---

### 1.2 — The Artificial Neuron: Mathematical Foundation

Every deep learning model, regardless of complexity, is built from one primitive: the artificial neuron. It is a mathematical function that takes a weighted sum of inputs, adds a bias, and passes the result through a non-linear activation function.

**The Neuron Equation:**

```
output = activation( Σ(wᵢ · xᵢ) + b )

Where:
  x = input vector
  w = learned weight vector
  b = learned bias scalar
  activation = non-linear function (ReLU, sigmoid, etc.)
```

In matrix form for an entire layer of N neurons receiving input vector x:

```
z = W·x + b
a = activation(z)

W ∈ ℝ^(N×M)   — weight matrix (N neurons, M inputs each)
b ∈ ℝ^N        — bias vector
a ∈ ℝ^N        — activation output vector
```

This is the entire computation at one layer. The "deep" in deep learning simply means stacking many such layers — each one transforming the output of the previous.

---

### 1.3 — Why Depth Matters: Representation Learning

A single-layer network (a perceptron) can only learn linear decision boundaries. Adding depth creates a hierarchy of learned representations. This is not metaphorical — it is what actually happens when you visualize learned features in trained networks.

| Layer Depth (CNN Example) | What Is Learned |
|---|---|
| Layer 1–2 | Edges, color gradients, basic textures |
| Layer 3–5 | Corners, curves, repeated patterns |
| Layer 6–10 | Object parts (eyes, wheels, windows) |
| Final layers | High-level semantic concepts ("dog face", "car") |

> **Key Insight:** Deep learning does not require hand-crafted features. The network learns what features are useful directly from raw data, given enough data and compute. This is the core breakthrough that separates deep learning from classical ML. The feature engineering is replaced by architecture engineering.

---

## Module 02 — Network Architecture & Layer Types

> **Foundation** · A deep neural network is a directed graph of mathematical functions. Every layer type is a different transformation — each designed to exploit a specific structure in the data.

---

### 2.1 — Core Layer Types and Their Purpose

Choosing which layers to use is architecture design — one of the most consequential decisions in building a model. Each layer type has a mathematical specialization.

| Layer | Formula | Used For |
|---|---|---|
| Dense / Fully Connected | `a = activation(W·x + b)` | Global relationships |
| Convolutional (Conv2D) | `a[i,j] = Σ(K * I)[i,j] + b` | Spatial patterns (images) |
| LSTM / GRU | `h_t = f(h_{t-1}, x_t)` | Sequential / temporal data |
| Multi-Head Attention | `Attn = softmax(QKᵀ/√d_k)·V` | Transformers / LLMs |
| BatchNorm / LayerNorm | `x̂ = (x − μ) / √(σ² + ε)` | Training stability |
| Dropout | `x_i = 0 with prob p (train only)` | Regularization |
| Embedding | `e = W_embed[token_id]` | Discrete → dense vector |
| Residual / Skip Connection | `out = F(x) + x` | Deep network training (ResNet) |

---

### 2.2 — Activation Functions: The Non-Linearity Engine

Without activation functions, a neural network of any depth is mathematically equivalent to a single linear transformation. Non-linearity is what gives neural networks their expressive power.

| Activation | Formula | Range | Used Where | Problem Solved |
|---|---|---|---|---|
| ReLU | `max(0, x)` | [0, ∞) | Hidden layers (CNNs) | Simple, fast, no vanishing in positive range |
| Leaky ReLU | `max(0.01x, x)` | (−∞, ∞) | Hidden layers | Fixes "dying ReLU" — dead neurons |
| GELU | `x·Φ(x)` | (−∞, ∞) | Transformers (BERT, GPT) | Smooth, probabilistic gating |
| Sigmoid | `1/(1+e⁻ˣ)` | (0, 1) | Binary output layer | Outputs probability |
| Softmax | `eˣⁱ/Σeˣʲ` | (0,1), sums to 1 | Multi-class output | Probability distribution over classes |
| Tanh | `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)` | (−1, 1) | RNNs, gates | Zero-centered, better gradient flow |

> ⚠️ **The Vanishing Gradient Problem:** Sigmoid and Tanh saturate — their gradients approach zero at large input values. In deep networks, backpropagated gradients are multiplied together across layers. Multiplying many near-zero values makes the gradient exponentially small before reaching early layers. ReLU largely solved this, which is why deep networks became trainable post-2012.

---

### 2.3 — The Transformer Architecture (Modern LLM Foundation)

Every major LLM today — GPT, Claude, Gemini, Llama — is built on the transformer. It replaced RNNs entirely for sequence tasks by processing all positions simultaneously using attention.

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V

Q = query matrix  (what am I looking for?)
K = key matrix    (what do I contain?)
V = value matrix  (what do I output if matched?)
d_k = key dimension (scaling prevents softmax saturation)
```

Multi-head attention runs this computation H times in parallel with different learned projections, then concatenates results — allowing the model to attend to information from different representation subspaces simultaneously.

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.shape[-1]
    # Compute raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # Normalize to probability distribution
    weights = F.softmax(scores, dim=-1)
    # Weighted sum of values
    return torch.matmul(weights, V)
```

---

## Module 03 — Data Pipelines & Representations

> **Foundation** · Every modality — images, video, text, audio, tabular — must be converted to numerical tensors before a model can process it. This pipeline is where most production bugs live.

---

### 3.1 — The Universal Rule: Everything Becomes a Tensor

A tensor is an N-dimensional array of floating-point numbers. It is the single data structure that flows through every layer of every neural network, regardless of the original data format.

**Pipeline:**

```
Raw Files (JPEG, MP4, TXT, CSV)
    → Preprocessing (resize, normalize, tokenize)
    → Dataset object (index-addressable)
    → DataLoader (batch, shuffle, parallel workers)
    → GPU Tensor (CUDA memory)
    → Forward Pass (model computation)
```

---

### 3.2 — Data Representations by Modality

| Modality | Raw Format | Preprocessing | Tensor Shape |
|---|---|---|---|
| Image | JPEG, PNG | Resize, normalize pixels [0,1] or [-1,1] | `[B, C, H, W]` e.g. [32, 3, 224, 224] |
| Video | MP4, AVI | Sample frames, resize, normalize | `[B, T, C, H, W]` e.g. [8, 16, 3, 224, 224] |
| Text | UTF-8 string | Tokenize → token IDs → embedding lookup | `[B, L, D]` e.g. [32, 512, 768] |
| Audio | WAV, MP3 | STFT → mel-spectrogram | `[B, F, T]` frequency bins × time steps |
| Tabular | CSV | Normalize numerics, embed categoricals | `[B, F]` batch × features |

*B = batch size, C = channels, H = height, W = width, T = time steps, L = sequence length, D = embedding dimension, F = features.*

---

### 3.3 — Text Tokenization Deep Dive (LLM Input)

For language models, converting text to tensors is a multi-step process. The tokenizer is a learned component, not a simple rule.

```python
# Step 1: Raw text
text = "Hello, world!"

# Step 2: BPE tokenization → subword token IDs
# "Hello" → [15496], "," → [11], " world" → [995], "!" → [0]
token_ids = tokenizer.encode(text)  # → [15496, 11, 995, 0]

# Step 3: Embedding lookup — each ID → dense vector of dim D
# Embedding matrix W_e ∈ ℝ^(vocab_size × D)
embeddings = embedding_layer(token_ids)  # → [4, 768]

# Step 4: Add positional encoding (sinusoidal or learned)
embeddings = embeddings + positional_encoding[:len(token_ids)]

# Now shape [seq_len, embed_dim] ready for transformer layers
```

> 💡 **Why Subword Tokenization?** Character-level: too long sequences. Word-level: can't handle "unrecognized" words. BPE (Byte Pair Encoding) splits rare words into known subwords — "unbelievably" might become ["un", "believ", "ably"]. GPT-4 uses ~100,000 token vocabulary.

---

### 3.4 — Data Augmentation as Regularization

Augmentation artificially expands the training set by applying label-preserving transformations. A model trained on augmented data generalizes better because it sees more variance in the input distribution.

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# Validation: NO augmentation — only normalize
val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
```

---

## Module 04 — Training & Backpropagation

> **Core Mechanics** · The learning algorithm. Forward pass, loss computation, gradient calculation via the chain rule, and weight updates via optimizers. This is the mathematical engine of all deep learning.

---

### 4.1 — The Training Loop: Full Picture

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        
        # 1. FORWARD PASS: data flows input → output
        predictions = model(batch_x)
        
        # 2. LOSS: scalar measure of how wrong we are
        loss = criterion(predictions, batch_y)
        
        # 3. ZERO GRADIENTS: clear accumulated grads from last step
        optimizer.zero_grad()
        
        # 4. BACKWARD PASS: compute ∂loss/∂W for every W
        loss.backward()
        
        # 5. GRADIENT CLIPPING: prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 6. WEIGHT UPDATE: optimizer adjusts all weights
        optimizer.step()
        
        # 7. LR SCHEDULE (optional)
        scheduler.step()
```

---

### 4.2 — Backpropagation: The Chain Rule Unfolded

Backpropagation is not magic — it is a systematic application of the calculus chain rule across a computational graph. Every operation in the forward pass must have a defined derivative so the gradient can flow backward.

**Chain Rule: Two-Layer Network Example:**

```
Forward:
  z₁ = W₁·x + b₁
  a₁ = ReLU(z₁)
  z₂ = W₂·a₁ + b₂
  ŷ  = softmax(z₂)
  L  = CrossEntropy(ŷ, y)

Backward (chain rule):
  ∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂W₂
  ∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁

Each term is the local gradient of that operation.
```

> ⚠️ **Does Data Flow Backward?** No. Input data is stationary. What flows backward are gradients — partial derivatives of the loss with respect to each weight. The same training data enters fresh through the forward pass in each epoch. Gradients are purely internal mathematical signals used to update weights.

---

### 4.3 — Optimizers: How Weights Are Updated

The gradient tells you which direction to move. The optimizer determines how far and how to adapt that movement over time.

**SGD (Stochastic Gradient Descent):**

```
W ← W − η · ∂L/∂W

η = learning rate (most important hyperparameter)
Simple, but slow and sensitive to learning rate choice.
```

**Adam (Adaptive Moment Estimation) — Industry Standard:**

```
m_t = β₁·m_{t-1} + (1−β₁)·g_t    ← 1st moment (momentum)
v_t = β₂·v_{t-1} + (1−β₂)·g_t²   ← 2nd moment (variance)
m̂_t = m_t / (1−β₁ᵗ)               ← bias correction
v̂_t = v_t / (1−β₂ᵗ)               ← bias correction
W ← W − η · m̂_t / (√v̂_t + ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3
```

AdamW adds weight decay decoupled from the gradient update — it is the default for training transformers. The learning rate is almost always combined with a schedule: linear warmup for the first N steps, then cosine decay to zero.

---

### 4.4 — Loss Functions by Task

| Task | Loss Function | Formula | Notes |
|---|---|---|---|
| Binary Classification | Binary Cross-Entropy | `−[y·log(p) + (1−y)·log(1−p)]` | Output: sigmoid |
| Multi-class | Categorical Cross-Entropy | `−Σ yᵢ·log(pᵢ)` | Output: softmax |
| Regression | MSE / MAE | `(y−ŷ)²` or `|y−ŷ|` | MAE more robust to outliers |
| Language Modeling | Cross-Entropy over vocab | `−log P(next token)` | Perplexity = e^(avg loss) |
| Object Detection | Focal Loss | Modified cross-entropy | Addresses class imbalance |

---

## Module 05 — Inference & Deployment

> **Core Mechanics** · Inference is the entire point — using a trained model to make predictions on new data. It is its own engineering discipline with distinct concerns from training.

---

### 5.1 — Training vs Inference: Fundamental Differences

| Aspect | Training | Inference |
|---|---|---|
| Gradients | Computed and stored | Never computed (`no_grad`) |
| Weights | Updated every step | Completely frozen |
| Dropout | Active (zeros random neurons) | Disabled (full network active) |
| BatchNorm | Running mean/var updated | Fixed statistics from training |
| Memory | Activations stored for backprop | Activations discarded immediately |
| Speed | Slow (gradient bookkeeping) | Significantly faster |
| Purpose | Minimize loss, adjust weights | Predict on new unseen data |

```python
# Switch model to evaluation mode (disables dropout, fixes BatchNorm)
model.eval()

with torch.no_grad():  # disables gradient computation engine
    for batch_x in test_loader:
        predictions = model(batch_x)
        # predictions are the model's outputs — no weight updates occur

# Common mistake: forgetting model.eval() causes Dropout to stay active
# → non-deterministic, lower-quality predictions at inference time
```

---

### 5.2 — Inference Optimization Techniques

| Technique | What It Does | Tradeoff |
|---|---|---|
| Quantization (INT8/FP16) | Reduce weight precision from FP32 to lower bits | ~2–4× speedup, ~1% accuracy drop |
| ONNX Export | Framework-agnostic model format, runtime optimization | Portability, hardware-specific gains |
| TensorRT | NVIDIA compiler — fuses ops, optimizes for GPU kernels | Large speedup, NVIDIA-only |
| KV Cache (LLMs) | Cache key/value tensors across auto-regressive steps | Memory for speed — essential for LLMs |
| Batching | Process multiple requests together on GPU | Latency vs throughput tradeoff |
| Speculative Decoding | Small model drafts tokens, large model verifies | ~2–3× LLM throughput gain |

---

### 5.3 — Post-Training Techniques

| Technique | Purpose | How |
|---|---|---|
| Fine-tuning | Adapt pretrained model to specific task/domain | Continue training on small task-specific dataset |
| LoRA | Parameter-efficient fine-tuning | Add low-rank adapter matrices; only train adapters (~0.1% params) |
| RLHF | Align LLM to human preferences | Train reward model on human rankings, optimize with PPO |
| DPO | Alignment without RL complexity | Direct optimization on preferred/rejected pairs |
| Distillation | Compress large model into small one | Train small model to mimic large model's output distribution |

---

## Module 06 — Interpretability & The Black Box

> **Advanced** · Do we understand what deep learning models actually learn? Partially and imperfectly. This module covers what is known, what is not, and the frontier research attempting to close the gap.

---

### 6.1 — The Black Box Problem

A trained neural network with billions of parameters is a mathematical function in an incomprehensibly high-dimensional space. No human can inspect the weights and understand what behavior they encode. This is not merely inconvenient — in safety-critical systems, it is dangerous.

> **Key Insight:** The network has learned something — but we cannot directly read what it learned the way we can read code. We can only probe it from the outside. This is the central challenge of ML interpretability. Every interpretability method is an indirect measurement, not a direct audit.

---

### 6.2 — Interpretability Techniques

| Method | Approach | Tells You | Limitations |
|---|---|---|---|
| Saliency Maps | Gradient of output w.r.t. input pixels | Which input regions most affect prediction | Noisy, doesn't explain internal mechanism |
| Grad-CAM | Class activation mapping via gradients | Where in image the model "looked" | Coarse localization only |
| Probing Classifiers | Train linear classifier on frozen hidden layer | What information is encoded in a layer | Presence ≠ usage |
| Attention Visualization | Plot attention weights as heatmaps | Token-to-token attention patterns | Attention ≠ explanation (Jain & Wallace 2019) |
| SHAP / LIME | Local approximations around predictions | Feature importance per prediction | Approximate, not mechanistic |
| Mechanistic Interp. | Reverse-engineer circuits in the network | Specific algorithms implemented | Extremely slow, doesn't scale |
| Sparse Autoencoders | Decompose superposed neuron activations | Monosemantic features per neuron | Active research, not production-ready |

---

### 6.3 — Superposition: Why Individual Neurons Are Hard to Read

A naive assumption is that each neuron in a trained network encodes exactly one concept. Empirical research has shown this is wrong. Neurons are **polysemantic** — they respond to multiple unrelated concepts simultaneously.

This happens because the model has more concepts to represent than it has neurons. It encodes multiple features per neuron using near-orthogonal directions in high-dimensional activation space. This is superposition — and it makes neuron-level interpretation fundamentally unreliable.

> 🔬 **Anthropic's Mechanistic Interpretability Research:** Research has identified specific computational circuits in transformers — "induction heads" that implement in-context learning, and "curve detectors" in CNNs. Sparse autoencoders are being used to decompose polysemantic neurons into interpretable monosemantic features. This is an active frontier with no production-ready solution yet.

---

## Module 07 — Adversarial ML & Security

> **Advanced · Security** · The complete threat model across every stage of the ML lifecycle — from data poisoning during creation to adversarial examples and prompt injection during production use.

---

### 7.1 — Stage 1: Data Collection & Preparation

#### 🔴 Data Poisoning — Critical
Attacker contaminates the training dataset. Poisoned samples alter learned weights permanently. Label flipping changes ground truth labels subtly enough to go undetected in large corpora.

- **Mitigation:** Dataset provenance tracking, statistical outlier detection

#### 🔴 Backdoor / Trojan Attack — Critical
Inject training samples with a specific trigger pattern (e.g., a pixel patch) mapped to a target label. Model behaves normally on clean inputs but misclassifies any input containing the trigger — even after deployment.

- **Mitigation:** Neural Cleanse, STRIP detection, certified defenses

#### 🟠 Supply Chain Poisoning — High
Poison public datasets (Hugging Face, Common Crawl) before they are consumed by downstream model trainers. Attackers have successfully demonstrated this against open web-scraped datasets.

- **Mitigation:** Hash verification, curated data sources, audit trails

---

### 7.2 — Stage 2: Training

#### 🔴 Gradient Leakage (DLG) — Critical
In federated learning, gradients shared between nodes can reconstruct original training data with high fidelity. The Deep Leakage from Gradients (DLG) attack demonstrated recovery of private images and text from gradient updates alone.

- **Mitigation:** Differential privacy (DP-SGD), gradient compression, secure aggregation

#### 🟠 Membership Inference — High
Determine whether a specific record was in the training dataset. Models overfit and behave differently on seen vs unseen samples — this confidence gap is the attack signal. Severe for medical / biometric models.

- **Mitigation:** Differential privacy, regularization to reduce overfitting

#### 🟠 Model Inversion — High
Iteratively query the model's output probabilities to reconstruct training data. Has been demonstrated on facial recognition models — recovering representative face images for each identity class.

- **Mitigation:** Output confidence truncation, query rate limiting

---

### 7.3 — Stage 3: Model Storage & Distribution

#### 🔴 Malicious Serialization (RCE) — Critical
PyTorch's legacy `.pkl` format allows arbitrary code execution on `torch.load()`. A malicious model file can spawn a reverse shell when loaded by a victim. This is a real, documented attack class — not theoretical.

- **Mitigation:** Use safetensors format, never load untrusted `.pt`/`.pkl` files

```python
# DANGEROUS — arbitrary code execution possible
model = torch.load("untrusted_model.pt")

# SAFE — safetensors format, no code execution
from safetensors.torch import load_file
model_weights = load_file("model.safetensors")
```

#### 🟠 Model Extraction / Theft — High
Attacker makes large numbers of queries to a deployed model and trains a surrogate that approximates its decision boundary. The surrogate can be analyzed offline to find adversarial examples, bypassing production audit logging.

- **Mitigation:** Rate limiting, query watermarking, output perturbation

#### 🟡 Weight Tampering — Medium
If model artifacts are stored without integrity controls, an attacker can modify weights directly — analogous to binary tampering in traditional software. Can introduce targeted misclassifications or hidden behaviors.

- **Mitigation:** Cryptographic signing, hash verification, immutable model registry

---

### 7.4 — Stage 4: Deployed Inference

#### 🔴 Adversarial Examples — Critical
Imperceptible perturbations δ added to input x such that `‖δ‖ < ε` but `model(x+δ) = target_class`.

Attack variants:
- **FGSM** (Fast Gradient Sign Method) — single-step: `x_adv = x + ε·sign(∇ₓL)`
- **PGD** (Projected Gradient Descent) — iterative, stronger
- **CW attack** — optimization-based, highest effectiveness
- **Physical-world attacks** — printed patches that fool real cameras

Mitigation: adversarial training, input preprocessing defenses, certified robustness

#### 🔴 Prompt Injection (LLMs) — Critical
Malicious instructions in user input or retrieved documents override the model's intended behavior.

- **Direct injection:** User directly instructs the model to ignore its system prompt
- **Indirect injection:** Attacker plants instructions in web pages or database entries the LLM retrieves — model executes attacker's intent without user knowledge

Mitigation: input sanitization, privilege separation, output filtering

#### 🟠 Jailbreaking — High
Craft inputs that bypass safety alignment. Role-play framing, encoded instructions (Base64, leetspeak), many-shot compliance examples, and token-level GCG (Greedy Coordinate Gradient) attacks that optimize adversarial suffixes.

- **Mitigation:** RLHF alignment, red-teaming, adversarial fine-tuning

#### 🟠 Sponge Examples (DoS) — High
Inputs crafted to maximize computational cost during inference — long sequences hitting worst-case attention complexity O(n²), or inputs triggering maximum computation paths. Used to exhaust GPU resources and deny service.

- **Mitigation:** Input length limits, compute budgets, anomaly detection

#### 🟠 Training Data Extraction — High
Repeatedly query an LLM to extract memorized training data — PII, source code, copyrighted content. Demonstrated against GPT-2 by Carlini et al., recovering verbatim training text including personal information.

- **Mitigation:** Differential privacy in training, deduplication, output monitoring

#### 🟡 Evasion via Feature Space — Medium
In security ML (malware/fraud detection), attacker mutates malicious artifact to move it into benign feature space — while preserving malicious functionality. The semantic preservation constraint is the key difficulty.

- **Mitigation:** Ensemble models, behavioral analysis beyond static features

---

### 7.5 — Defense-in-Depth: The Security Posture

> 🛡️ **Treat the ML Model as Critical Infrastructure.** Apply the same threat modeling you would to any production system. The model file is a binary artifact — sign it. The training data is a dependency — audit it. The inference endpoint is an API — rate limit and monitor it. The model's outputs are user-facing — filter and log them. No single defense is sufficient; the entire lifecycle requires controls.

| Lifecycle Stage | Primary Threats | Key Controls |
|---|---|---|
| Data collection | Poisoning, supply chain | Provenance, hash verification, outlier detection |
| Training | Gradient leakage, membership inference | DP-SGD, secure aggregation, regularization |
| Storage | Weight tampering, malicious serialization | Safetensors, signing, immutable registry |
| Distribution | Model theft, extraction | Rate limiting, watermarking, auth |
| Inference | Adversarial inputs, prompt injection | Input validation, adversarial training, monitoring |
| Monitoring | Drift exploitation, shadow attacks | Distribution monitoring, anomaly detection, red-teaming |

---

## Quick Reference: The Full Training Pipeline

```
Raw Data → Preprocessing → Tensors → DataLoader → Mini-batches
    ↓
Forward Pass: Input → Hidden Layers → Output → Prediction
    ↓
Loss Computation: L(y, ŷ)
    ↓
Backward Pass: ∂L/∂W via chain rule (gradients flow backward)
    ↓
Weight Update: W ← W − η·∇W (optimizer)
    ↓
Repeat for N epochs
    ↓
Trained Model (frozen weights)
    ↓
Inference: new input → forward pass only → prediction
```

---

## Quick Reference: Activation Functions

| Activation | Formula | Best For |
|---|---|---|
| ReLU | `max(0, x)` | CNNs, dense layers |
| GELU | `x·Φ(x)` | Transformers |
| Sigmoid | `1/(1+e⁻ˣ)` | Binary output |
| Softmax | `eˣⁱ/Σeˣʲ` | Multi-class output |
| Tanh | `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)` | RNNs, gates |

---

## Quick Reference: Attack Severity Matrix

| Severity | Stage | Attack |
|---|---|---|
| 🔴 Critical | Data | Data poisoning, backdoor injection |
| 🔴 Critical | Training | Gradient leakage (DLG) |
| 🔴 Critical | Storage | Malicious serialization (RCE) |
| 🔴 Critical | Inference | Adversarial examples, prompt injection |
| 🟠 High | Data | Supply chain poisoning |
| 🟠 High | Training | Membership inference, model inversion |
| 🟠 High | Storage | Model extraction / theft |
| 🟠 High | Inference | Jailbreaking, sponge examples, data extraction |
| 🟡 Medium | Storage | Weight tampering |
| 🟡 Medium | Inference | Feature space evasion |

---

*Deep Learning: End-to-End Engineering Course — ML Engineering × Cybersecurity*

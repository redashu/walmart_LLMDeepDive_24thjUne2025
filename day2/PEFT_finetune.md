## Why Fine-tune a Pretrained LLM?

Pretrained LLMs (e.g., LLaMA, Mistral, GPT variants) are trained on general-purpose corpora. However, real-world applications often require domain-specific knowledge and/or task alignment. That’s where fine-tuning comes in.

### Key Reasons to Fine-tune

| Objective                                   | Description                                                                                   |
|----------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Domain Adaptation**                        | Inject vocabulary, patterns, and logic from specific domains (e.g., retail, legal, finance, healthcare). |
| **Task-Specific Behavior**                   | Teach the model to perform classification, summarization, translation, or Q&A tailored to business needs. |
| **Improved Accuracy**                        | Outperform general LLMs on narrow use cases (e.g., Walmart product categorization or customer support). |
| **Controlled Behavior**                      | Reduce hallucinations, increase safety, or enforce stylistic constraints.                      |
| **Better Performance in Low-Resource Languages or Styles** | Adapt to code, emojis, dialects, or non-standard inputs.                                      |

---

### Types of Fine-tuning

#### 1. Supervised Fine-tuning (SFT)

This is the standard fine-tuning method, where you train the model on input-output pairs using supervised learning.

**Characteristics:**
- Uses labeled datasets: `(input_text, target_output)`
- Optimizes cross-entropy loss.
- Often the first stage in instruction-tuned or alignment models.

**Examples:**
- Fine-tuning GPT-2 to generate product descriptions.
- Fine-tuning LLaMA on QA pairs in legal documents.

```python
# Pseudocode using HuggingFace Transformers
trainer = Trainer(
    model=llm,
    train_dataset=train_data,
    args=TrainingArguments(...),
    tokenizer=tokenizer
)
trainer.train()
```

#### 2. Instruction Fine-tuning (IFT)

Instruction tuning goes beyond simple input-output learning. The idea is to teach models how to follow natural language instructions, often across a variety of tasks.

Think of it as multi-task supervised tuning with tasks framed as "Write a summary of...", "Translate this to French...", etc.

**Key Models:**
- T5 (“Text-to-text Transfer Transformer”) — foundational instruction-tuned model.
- FLAN-T5 — Google's extensive instruction-tuned model.
- OpenAssistant, Alpaca, Dolly — community-driven IFT models.

**Dataset Examples:**
- Super-NaturalInstructions
- FLAN Collection
- Open-Orca, Databricks-Dolly

**Advantages:**
- Makes the model zero/few-shot capable across a wide range of downstream tasks.
- Crucial for building instruction-following agents and chatbots.

#### 3. Parameter-Efficient Fine-tuning (PEFT)

Rather than updating all parameters, PEFT strategies update only a small subset of trainable parameters or introduce additional layers/modules. This reduces compute, cost, and memory footprint.

| Method         | Description                                         | Notes                                 |
|----------------|-----------------------------------------------------|---------------------------------------|
| LoRA           | Injects low-rank matrices into attention layers     | ~0.1% of full model parameters        |
| QLoRA          | LoRA + 4-bit quantization                          | Enables fine-tuning of 65B+ models on single GPU |
| Adapters       | Adds small bottleneck layers between transformer blocks | Original PEFT approach                |
| Prefix Tuning  | Learns task-specific prefix vectors prepended to attention layers | Lightweight, effective in multi-task  |
| P-Tuning/v2    | Learns continuous prompt embeddings (instead of tokens) | Used with frozen LLMs                 |

PEFT is critical in enterprise settings where compute is constrained and reproducibility matters.

#### 4. Reinforcement Learning-based Fine-tuning

These methods optimize behavior using reward signals, typically in alignment stages:

- **RLHF (Reinforcement Learning from Human Feedback):**
  - Three steps: SFT → Reward Model Training → PPO-based RL
  - Popularized by OpenAI (ChatGPT)
  - Used to fine-tune models toward human-preferred outputs

- **RLAIF (RL from AI Feedback):**
  - Replaces human preferences with model-generated preference ranking
  - Cost-effective alternative when human annotation is infeasible

#### 5. Continual Fine-tuning / Domain Adaptation

Model is incrementally fine-tuned on new domain-specific data without forgetting prior knowledge (a.k.a. Catastrophic Forgetting).

**Use Cases:**
- Weekly updates to a model on newly added Walmart product data.
- Fine-tuning LLaMA on real-time support tickets from different regions.

#### 6. Multi-task Fine-tuning

Train a model on multiple tasks simultaneously to encourage task-agnostic generalization.

- Often combined with instruction tuning.
- Common in FLAN, T0, and MT-DNN frameworks.

#### 7. Contrastive Fine-tuning

Used in embedding models (like Sentence-BERT), this optimizes the model to pull semantically similar samples together and push unrelated ones apart.

- Training objective: contrastive loss or triplet loss.
- Ideal for similarity search, clustering, and retrieval.

#### 8. Low-resource Fine-tuning / Few-shot Fine-tuning

Designed for low-data regimes:

- Often combines PEFT + data augmentation.
- Popular in verticals like biomedicine, legal, where labeled data is scarce.

#### 9. Multi-modal Fine-tuning

Fine-tune models that combine text + other modalities (vision, audio, tabular).

- E.g., LLaVA, Flamingo, GPT-4V: fine-tuned with aligned (image, caption) pairs.
- Requires cross-modal objectives (contrastive, alignment loss).
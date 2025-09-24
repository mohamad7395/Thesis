# ABSA Data Augmentation Framework

This repository provides **data augmentation pipelines for Aspect-Based Sentiment Analysis (ABSA)**.  
It supports two complementary approaches:  

1. **Agentic Pipeline (LangGraph + Ollama)** – generates sentences with aspect–polarity pairs using a generator + evaluator agent.  
2. **Prompting Pipeline (Naive Generation)** – directly prompts an LLM to produce aspect–polarity sentences without explicit validation.  

The augmented data is used with the [InstructABSA](https://github.com/yourlink/InstructABSA) framework for training and evaluation.  

---

## 🚀 Features
- Augment datasets for **ABSA in the Restaurant domain**   
- Two strategies:
  - **Agentic**: validated samples, slower but higher quality.  
  - **Prompting**: faster generation, noisier but scalable.  
- Uses local LLMs with [Ollama](https://ollama.com/) and Hugging Face Transformers.  
- Seamlessly integrates with **InstructABSA** for downstream experiments.  

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourname/absa-augmentation.git
cd absa-augmentation
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```
### 3. Install Ollama

Follow instructions from [Ollama](https://ollama.com/)

### 4. Pull required models
```bash
ollama pull qwen2.5:14b
ollama pull llama3:8b-instruct
```

## 📜 Usage

## Agentic Pipeline (LangGraph + Ollama)

Run the controlled agent-based data generation:

```bash
python run_agent.py
```

## Prompting Pipeline (Naive LLM prompts)

```bash
python run_prompting.py
```

## Workflow 


SemEval Dataset
        │
        ├── Agentic Pipeline  ──> Augmented Data (validated)
        └── Prompting Pipeline ─> Augmented Data (naive)
        
Augmented Data ──> InstructABSA ──> Model Training & Evaluation




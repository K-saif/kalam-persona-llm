# APJ Abdul Kalam - LLM Fine-tuning

This project focuses on fine-tuning Large Language Models (LLMs) to specialize in content related to **APJ Abdul Kalam**, including his inspirational teachings, personal experiences, and wisdom. The repository implements both **Continued Pre-Training (CPT)** and **Supervised Fine-Tuning (SFT)** approaches using the Unsloth library for efficient training on limited hardware.

## 📋 Project Overview

The goal is to create a specialized LLM that can:
- Generate responses about APJ Abdul Kalam's life, lessons, and philosophy
- Answer questions based on his famous speeches and writings
- Provide inspirational insights from his perspective
- Maintain context through conversation

## 🗂️ Repository Structure

### Core Training Scripts

- **`unsloth_train.py`** - Main CPT (Continued Pre-Training) script using Qwen 2.5 model with LoRA adapters
- **`SFT_fine_tune.py`** - Supervised Fine-Tuning script for chat-format instruction data
- **`push_to_huggingface.py`** - Utility to push trained models to Hugging Face Hub

### Datasets (`Datasets/`)

Contains diverse training data organized into categories:

- **`kalam_chat_format.jsonl`** - Chat-formatted conversations for SFT
- **`kalam_cpt.jsonl`** - Raw text data for CPT pre-training (650+ paragraphs)
- **`kalam_qa_pairs.json`** - Question-Answer pairs dataset
- **`kalam_sft_final_fixed.jsonl`** - Finalized SFT data

#### Subcategories
- **`GK_and_Personal_Info/`** - General knowledge and personal information about APJ Abdul Kalam
- **`My_journey/`** - Content about his personal journey
- **`Wings_of_fire/`** - Data from his autobiography "Wings of Fire"

### Q&A Extractor Agent (`extractor_agent/`)

Intelligent agent for extracting and structuring Q&A pairs from messy data:

- **`qa_extractor_agent.py`** - Main extraction script using local LLM
- **`QA_EXTRACTOR_SETUP.md`** - Detailed setup and usage guide
- **`run_agent.sh`** / **`run_agent.bat`** - Easy execution scripts
- **`SAMPLE_OUTPUT.json`** - Example output format

### Trained Models

- **`kalam_cpt_lora_v2/`** - CPT LoRA adapter (latest version)
  - `adapter_config.json` - LoRA configuration
  - `adapter_model.safetensors` - Model weights
  - `tokenizer.json` - Tokenizer files

- **`kalam_cpt_output_v2/`** - CPT training outputs and checkpoints
  
- **`kalam_final_model/`** - Final merged model with full weights
  
- **`kalam_sft_output/`** - SFT training outputs and checkpoints

### Utilities (`utils/`)

Helper scripts for data processing:

- **`generate_qa.py`** - Generate Q&A pairs from raw text
- **`qa_to_chatML.py`** - Convert Q&A to ChatML format
- **`txt_to_jsonl.py`** - Convert text files to JSONL format
- **`paste_script_advanced.py`** - Advanced data processing script

### Build Cache (`unsloth_compiled_cache/`)

Pre-compiled trainer implementations for Unsloth (BCO, CPO, DPO, KTO, PPO, ORPO, etc.)

## 🚀 Quick Start

### Prerequisites

```bash
pip install unsloth torch transformers datasets trl peft
```

Optional: For Q&A extraction with local LLM:
```bash
# Install Ollama from https://ollama.ai
# Or LM Studio from https://lmstudio.ai
```

### 1. Continued Pre-Training (CPT)

```bash
python unsloth_train.py
```

This trains the model on raw Kalam-related text to specialize the base model.

### 2. Supervised Fine-Tuning (SFT)

```bash
python SFT_fine_tune.py
```

This fine-tunes the CPT model on chat-format Q&A pairs.

### 3. Extract Q&A Pairs (Optional)

Navigate to the extractor_agent folder:

```bash
cd extractor_agent
./run_agent.sh  # On Linux/Mac
# or
run_agent.bat   # On Windows
```

See [QA_EXTRACTOR_SETUP.md](extractor_agent/QA_EXTRACTOR_SETUP.md) for detailed instructions.

### 4. Push to Hugging Face (Optional)

```bash
python push_to_huggingface.py
```

## 📊 Data Format

### CPT Data Format (`.jsonl`)
```json
{"text": "Full paragraph of APJ Abdul Kalam content..."}
```

### SFT Data Format (`.jsonl`)
```json
{
  "messages": [
    {"role": "user", "content": "Question about APJ Abdul Kalam"},
    {"role": "assistant", "content": "Answer from Kalam's perspective"}
  ]
}
```

### Q&A Pairs Format (`.json`)
```json
{
  "context": "Background information",
  "questions": [
    {"question": "...", "answer": "..."}
  ]
}
```

## 🛠️ Configuration

Key parameters in training scripts:

- **`max_seq_length`**: 1024 tokens (adjust based on available VRAM)
- **`load_in_4bit`**: Quantization for memory efficiency
- **`r` (LoRA rank)**: 16 (increase for more capacity)
- **`lora_alpha`**: 16 (scaling factor)
- **`per_device_train_batch_size`**: Start with 1 for 8GB VRAM

## 📝 Dataset Statistics

- **Total Categories**: 8+ subcategories
- **Content Sources**: 
  - General Knowledge & Personal Info
  - Autobiography (Wings of Fire)
  - Personal Journey & Experiences
  - Inspirational & Life Lessons

## 🔗 References

- **Unsloth Documentation**: https://github.com/unslothai/unsloth
- **APJ Abdul Kalam**: Former President of India (1931-2015)
- **Famous Works**: "Wings of Fire", "Ignited Minds"

## 📄 License

Please refer to individual file headers for licensing information.

## 🤝 Contributing

To add more Kalam-related content:

1. Add raw text to appropriate `Datasets/` subfolder
2. Convert to JSONL format using `utils/txt_to_jsonl.py`
3. Extract Q&A pairs using the extractor agent
4. Re-train or fine-tune models as needed


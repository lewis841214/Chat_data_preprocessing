# Chat Data Preprocessing Pipeline

A comprehensive framework for preprocessing and cleaning structured conversation data for training language models. This pipeline handles the full process from raw data to clean, deduplicated outputs.

## Overview

The Chat Data Preprocessing Pipeline takes **structured conversation data** as input and produces:
- **Clean, deduplicated conversations** - For conversational model training
- **Deduplicated question-answer pairs** - For instruction-tuning and QA systems

## Features

- **Flexible deduplication algorithms**:
  - MinHash LSH
  - SimHash
  - Semantic deduplication
  - DBSCAN clustering
  - BERTopic-based deduplication
  - Bloom filter
  - Suffix array-based methods
  
- **Customizable processing**: Configure each stage of the pipeline to suit your requirements

- **Key processing capabilities**:
  - Near-duplicate detection and removal
  - Intelligent representative selection from duplicate clusters
  - Support for both conversation-mode and QA-pair processing

## Usage

The pipeline accepts structured conversation data and can process either full conversations or extract and deduplicate question-answer pairs based on your configuration.

### Input Format

Structured conversation data with messages containing roles and content.

### Output

- Deduplicated, cleaned conversations
- Deduplicated question-answer pairs for instruction fine-tuning

## Configuration

The pipeline supports extensive configuration options including:
- Deduplication method selection
- Similarity thresholds
- Processing parameters

## Project Structure

- `pipeline.py` - Main orchestration script
- `config_handler.py` - Configuration handling
- `processor.py` - Base processor interface
- `/configs` - Configuration files
- `/platform` - Platform-specific data handlers
- `/cleaning` - Content cleaning modules
- `/filtering` - Quality filtering modules
- `/formatting` - Text formatting modules
- `/deduplication` - Near-duplicate detection algorithms

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chat-data-preprocessing.git
cd chat-data-preprocessing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline:
```bash
source venv/bin/activate
python pipeline.py --config configs/default_config.yaml
```

## Input/Output Format

### Input Format

The pipeline accepts data in various formats and standardizes them. The most straightforward format is:

```json
// file.json
{
  "platfrom": ,
  "created_at": timestamp with time zone,
  "last_message_at": timestamp with time zone,
  "conversation": [
    {
      "role": "User",
      "content": "What is the capital of France?",
      "do_train": false
    },
    {
      "role": "Assistant",
      "content": "The capital of France is Paris.",
      "do_train": true
    }
  ]
}
```

### Output Format

The output is a list of standardized conversation objects in JSON format:

```json
// file.json
{
  "platfrom": ,
  "created_at": timestamp with time zone,
  "last_message_at": timestamp with time zone,
  "conversation": [
    {
      "role": "User",
      "content": "What is the capital of France?",
      "do_train": false
    },
    {
      "role": "Assistant",
      "content": "The capital of France is Paris.",
      "do_train": true
    }
  ]
}
```

## Extending the Pipeline

### Adding a New Platform Handler

1. Create a new handler in the `platform` directory (e.g., `reddit_handler.py`).
2. Implement the `transform` method to convert platform-specific data to the standard format.
3. Register the handler in `platform/handler_factory.py`.

### Adding a New Deduplication Method

1. Create a new method in the `deduplication` directory (e.g., `simhash.py`).
2. Implement the `process` method.
3. Register the method in `deduplication/dedup_manager.py`.

## Pipeline Flow

```
┌───────────────────────┐
│ Large History Chat    │
│ Corpus                │
│ - Open source         │
│ - Crawled             │
│ - Commercial          │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Platform Specific     │
│ Data Handler          │
│ - API data adapters   │
│ - Format conversion   │
│ - Platform metadata   │
│ - Source labeling     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 1. Remove Useless     │
│    Content            │
│ - Filter stopwords    │
│ - Language filtering  │
│ - URL filtering       │
│ - Paragraph removal   │
│ - Exact deduplication │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 2. Quality Filtering  │
│ - FastText language   │
│   detection           │
│ - KenLM quality       │
│   assessment          │
│ - GPT evaluation      │
│ - Logistic Regression │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 3. Text Formatting    │
│ - Remove punctuation  │
│ - Clean HTML tags     │
│ - Fix unicode issues  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 4. Near-Duplicate     │
│    Detection          │
│ - MinHash & LSH       │
│ - SimHash             │
│ - Semantic dedup      │
│ - Suffix Array method │
│ - DBSCAN clustering   │
│ - BERTopic approach   │
│ - Bloom filters       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 5. Final Corpus       │
│    Generation         │
│ - JSON format         │
└───────────────────────┘
```

## License

MIT 
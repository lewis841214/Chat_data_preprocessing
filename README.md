# Chat Data Preprocessing Pipeline

A comprehensive framework for preprocessing and cleaning chat data for training language models. This pipeline handles the full process from raw data to a clean, deduplicated corpus.

## Features

- **Platform-specific data handlers** - Process data from various sources (Reddit, Twitter, Discord, etc.)
- **Content cleaning** - Remove useless content like stopwords, URLs, and irrelevant paragraphs
- **Quality filtering** - Detect language, assess quality with KenLM, and apply ML classifiers
- **Text formatting** - Clean HTML tags, fix Unicode issues, and normalize text
- **Near-duplicate detection** - Several algorithms for efficient deduplication (MinHash, SimHash, Semantic, etc.)
- **Configurable pipeline** - YAML-based configuration with sensible defaults

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

1. Create a configuration file (or use the default one):
```bash
python config_handler.py --output configs/my_config.yaml
```

2. Customize the configuration by editing the YAML file.

3. Run the pipeline:
```bash
python pipeline.py --config configs/my_config.yaml
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

## License

MIT 
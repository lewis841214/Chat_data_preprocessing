# [data preprocessing (reference)](https://docs.google.com/presentation/d/1Hua0zIVnyK7itodxY-JGpyxEOrdqcEfXyOczcOiZmRU/edit#slide=id.g2f5efb43de4_0_0)

## Framework to start:
1. [chat-data-pipeline](https://github.com/AlekseyKorshuk/chat-data-pipeline)
   - A framework to handle the:
        - Defined chat sturcture
        - Cleaning
        - Filtering
        - Deduplication


## 1.1. Data Preprocessing Flow Chart

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
│ Data Handler(optional)|
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

This flow chart illustrates the complete preprocessing pipeline, highlighting where Near-Duplicate Detection fits within the overall process. The current document focuses specifically on step 4 (Near-Duplicate Detection).

1. Given a large history chat corpus
2. Reformulate it into pre-defined structure in .json format: 
    [
        {}
    ]
3. Given a large history chat corpus from various sources:
   - Open source corpus (~2.9 billion entries)
   - Crawled corpus (~3.1 billion entries) 
   - Commercial corpus

4. Process through platform-specific data handlers:
   - Convert platform-specific APIs and formats
   - Extract and normalize metadata
   - Label data by source
   - Handle platform-specific features

5. Remove useless content:
   - Filter out stopwords
   - Language filtering
   - URL filtering
   - Paragraph removal
   - Exact match deduplication

6. Obtain high-quality Traditional Chinese and English data (~1.4 billion entries):
   - Apply FastText for language detection
   - Use KenLM for quality assessment
   - GPT evaluation
   - Logistic Regression classification

7. Format text:
   - Remove punctuation
   - Clean HTML tags
   - Fix unicode encoding issues

8. Optional deduplication:
   - [MinHash for efficient similarity detection](minhash_based_deduplication.md) 
   - [Semantic deduplication for content-based similarity](https://docs.google.com/presentation/d/1Hua0zIVnyK7itodxY-JGpyxEOrdqcEfXyOczcOiZmRU/edit#slide=id.g2d2800a15f9_0_209) paper link: [paper](https://arxiv.org/pdf/2308.12284), [facebook method](https://github.com/facebookresearch/SemDeDup)
   - [Suffix Array-based substring deduplication](https://aclanthology.org/2021.naacl-main.262/) - Identifies duplicate fragments within and across documents
   - [SimHash for approximate matching](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/SimHash.pdf) - Locality-sensitive hashing algorithm for near-duplicate detection
   - [DBSCAN clustering](https://en.wikipedia.org/wiki/DBSCAN) - Density-based approach for clustering similar documents
   - [Hierarchical Agglomerative Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) - Groups similar documents using Jaccard or other similarity metrics
   - [Onion method](https://arxiv.org/abs/2107.06499) - Progressive n-gram based deduplication approach used in large multilingual corpora
   - [BERTopic](https://github.com/MaartenGr/BERTopic) - Topic-based clustering that can be adapted for semantic deduplication
   - [Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter) - Space-efficient probabilistic data structure for approximate set membership testing

9. Final corpus generation

### JSON Structure
Reformulate processed data into pre-defined structure in JSON format:

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

## Chat History Deduplication Methods

Chat data presents unique deduplication challenges due to its conversational nature, multi-turn interactions, and varying quality. These specialized approaches are recommended for effective chat history deduplication:

### 9.1 Conversation-Level Techniques

- **Conversation Fingerprinting**: Generate signatures for entire conversations rather than individual messages
  - Use sliding window approaches to capture context across turns
  - Weight user queries more heavily than system responses for similarity calculation
  - Example implementation: [ConvFingerprint](https://github.com/ueqri/chat-deduplication)

- **Dialog Act-Based Deduplication**: Group similar conversations based on dialog acts and interaction patterns
  - Classify utterances by intent (question, statement, greeting, etc.)
  - Compare conversation flows rather than exact text
  - Particularly effective for task-oriented dialogs

### 9.2 Turn-Level Methods

- **Response Diversity Preservation**: When deduplicating, prioritize keeping diverse responses to similar queries
  - Cluster similar user queries but maintain variety in corresponding responses
  - Use techniques like Maximal Marginal Relevance (MMR) to balance similarity with diversity

- **Context-Aware Embedding**: Generate embeddings that capture conversational context
  - Use techniques like DialoGPT or conversation-BERT to embed multi-turn context
  - Apply cosine similarity thresholds on these contextualized embeddings
  - More effective than single-utterance embedding approaches

### 9.3 Implementation Strategy

For large-scale chat history deduplication, we recommend this multi-stage approach:

1. **Coarse Filtering**: Use lightweight methods first
   - N-gram overlap for exact/near-duplicate conversations
   - Conversational template detection (scripted interactions)
   
2. **Fine-Grained Deduplication**: Apply more sophisticated methods to remaining data
   - Conversation-level embeddings for semantic similarity
   - Dialog flow comparison for structural similarity

3. **Quality-Aware Selection**: When duplicates are found, retain the highest quality instance
   - Consider response coherence, specificity, and helpfulness
   - Preserve conversations with unique contextual information

4. **Sampling**: After deduplication, apply stratified sampling to maintain diversity
   - Balance common vs. rare conversation types
   - Ensure representation across domains and interaction patterns

This tiered approach balances computational efficiency with deduplication effectiveness for chat data, preserving conversational diversity while removing truly redundant content.

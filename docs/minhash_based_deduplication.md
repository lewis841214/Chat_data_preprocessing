# MinHash-Based Corpus Deduplication

## 1. Overview

MinHash and Locality-Sensitive Hashing (LSH) provide an efficient way to detect and remove near-duplicate documents from a large corpus. This method significantly reduces pairwise comparison costs while maintaining high accuracy in identifying similar documents.

## 2. Steps (Pseudocode)

**Input**: Corpus with *n* documents

1. **Preprocess Documents**: Tokenize and normalize text.

2. **Compute MinHash Signatures**:
   - Generate *k* random hash functions.
   - Compute MinHash signatures of length *k* for each document.

3. **Apply Locality-Sensitive Hashing (LSH)**:
   - Split each signature into *b* bands of *r* rows.
   - Hash each band into a bucket.
   - Retrieve candidate document pairs from matching buckets.

4. **Compute Jaccard Similarity**:
   - For each candidate pair, estimate Jaccard similarity using MinHash.
   - Filter pairs where *similarity < t* (threshold).

5. **Cluster Similar Documents**:
   - Use connected components, hierarchical clustering, or DBSCAN.

6. **Deduplicate Corpus**:
   - Select a representative document per cluster.
   - Remove or merge duplicates.

**Output**: Deduplicated corpus and mapping of removed duplicates.

## 3. MinHash Explanation

MinHash is a probabilistic technique for estimating Jaccard similarity efficiently.

### Probability Interpretation

MinHash is based on the idea that for a random hash function *h*:

The probability that two sets *A* and *B* share the same minimum hashed value is equal to their Jaccard similarity:

$$P(min(h(A)) = min(h(B))) = \frac{|A \cap B|}{|A \cup B|}$$

With *k* independent hash functions, the fraction of matching MinHash values approximates the true Jaccard similarity:

$$\text{Jaccard}(A, B) \approx \frac{1}{k} \sum_{i=1}^{k} [min(h_i(A)) = min(h_i(B))]$$

More hash functions (*k*) improve the estimation accuracy.

## 4. Locality-Sensitive Hashing (LSH) for Speedup

LSH speeds up similarity search by reducing unnecessary comparisons:

### Divide MinHash Signatures into *b* Bands:
- Each band has *r* rows, so *k = b × r*.

### Hash Each Band into Buckets:
- Similar documents land in the same bucket with high probability.

### Compare Only Documents in the Same Bucket:
- Avoids computing Jaccard similarity for all *O(n²)* pairs.

### Visual Illustration

```
Document 1      Document 2      Document 3      Document 4      Document 5
MinHash         MinHash         MinHash         MinHash         MinHash
Signature       Signature       Signature       Signature       Signature
[               [               [               [               [
 8,              8,              2,              8,              2,   ┐
 1,              5,              1,              1,              5,   │ Band 1
 4,              7,              4,              4,              7,   │ (r=3 rows)
                                                                      ┘
 2,              3,              9,              2,              9,   ┐
 9,              1,              3,              9,              1,   │ Band 2
 5,              6,              8,              5,              8,   │ (r=3 rows)
                                                                      ┘
 7,              4,              6,              7,              6,   ┐
 3,              9,              5,              3,              4,   │ Band 3
 6,              2,              7,              6,              2,   │ (r=3 rows)
 ...             ...             ...             ...             ...  ┘
]               ]               ]               ]               ]
    |               |               |               |               |
    v               v               v               v               v
+------------+----------------+----------------+----------------+----------------+
| Band 1     |                |                |                |                |
| (rows 1-3) | hash(8,1,4)    | hash(8,5,7)    | hash(2,1,4)    | hash(8,1,4)    | hash(2,5,7)    |
+------------+----------------+----------------+----------------+----------------+----------------+
| Band 2     |                |                |                |                |
| (rows 4-6) | hash(2,9,5)    | hash(3,1,6)    | hash(9,3,8)    | hash(2,9,5)    | hash(9,1,8)    |
+------------+----------------+----------------+----------------+----------------+----------------+
| Band 3     |                |                |                |                |
| (rows 7-9) | hash(7,3,6)    | hash(4,9,2)    | hash(6,5,7)    | hash(7,3,6)    | hash(6,4,2)    |
+------------+----------------+----------------+----------------+----------------+----------------+
      |               |               |               |               |
      v               v               v               v               v
+------------+----------------+----------------+----------------+----------------+
| Bucket     | Bucket         | Bucket         | Bucket         | Bucket         |
| for Band 1 | "a7f2"         | "b3d1"         | "c5e8"         | "a7f2"         | "c5e8"         |
+------------+----------------+----------------+----------------+----------------+----------------+
| Bucket     | Bucket         | Bucket         | Bucket         | Bucket         |
| for Band 2 | "d9c4"         | "f2e6"         | "g1h3"         | "d9c4"         | "k7j2"         |
+------------+----------------+----------------+----------------+----------------+----------------+
| Bucket     | Bucket         | Bucket         | Bucket         | Bucket         |
| for Band 3 | "m4n6"         | "p8q2"         | "r3s5"         | "m4n6"         | "t9u1"         |
+------------+----------------+----------------+----------------+----------------+----------------+
      |               |               |               |               |
      +---------------+---------------+---------------+---------------+
                              |
                              v
                   +------------------------+
                   | Identified Candidates: |
                   | - Doc1 & Doc4: Band 1  |
                   |   AND Band 2 AND Band 3|
                   | - Doc2 & Doc5: None    |
                   | - Doc3 & Doc5: Band 1  |
                   +------------------------+
                              |
                              v
                   +------------------------+
                   | Calculate Jaccard      |
                   | similarity only for    |
                   | candidate pairs        |
                   +------------------------+
```

In this improved illustration:
1. Each document has a MinHash signature of length k (k = 9 in this example)
2. The signatures are divided into b=3 bands, each with r=3 rows (so k = b × r = 3 × 3 = 9)
3. Each band (containing r=3 values) is hashed as a unit to a bucket
4. Documents that share the same bucket in any band become candidate pairs
5. Only Doc1 & Doc4 share buckets in all three bands (high similarity)
6. Doc3 & Doc5 share a bucket only in Band 1 (potential similarity)
7. Only these candidate pairs undergo full Jaccard similarity calculation

The probability of becoming a candidate pair increases with the number of bands (b) and decreases with the number of rows per band (r). This allows tuning the trade-off between precision and recall.

### Time Complexity Comparison

| Method | Pairwise Comparison Cost |
|--------|--------------------------|
| Exact Jaccard | *O(n²)* |
| MinHash | *O(n²)* |
| MinHash + LSH | *O(n × s)* where *s* is average bucket size |

## 5. Clustering Similar Documents

After obtaining the sparse similarity matrix from LSH, we cluster documents to identify duplicates.

### Options for Clustering

1. **Graph-Based (Connected Components)**
   - Build a graph where edges exist if *similarity ≥ t*.
   - Use Union-Find or BFS/DFS to find connected components.

2. **Hierarchical Clustering**
   - Use Agglomerative Clustering with Jaccard distance.
   - Stop merging when similarity < *t*.

3. **DBSCAN (Density-Based Clustering)**
   - Treat documents as points in MinHash space.
   - Define neighbors based on similarity threshold *t*.

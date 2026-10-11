# Retrieval-Augmented Generation for Scientific Literature

Fictional paper for testing document ingestion. Every study result below is invented and is not a NexusRAG benchmark result.

## Abstract

This fictional study compares retrieval-augmented generation (RAG) with
generation that uses no retrieved passages. The example system answers
questions about scientific literature and cites source passages. In the
invented experiment with 1,200 claims, the system produced 41% fewer
unsupported statements than the baseline. These numbers provide sample
text for parsing and question answering; they are not measured results.

## Introduction

An answer can read clearly and still misstate a paper's findings. This
example asks whether retrieving passages and checking their relevance
could reduce unsupported statements. It describes a fictional system
that retries retrieval when the initial passages receive low scores.

## Methods

The example pipeline parses documents and splits them into overlapping
chunks. It combines embedding search and keyword matching to rank
passages, then sends the selected passages to a generator. The generator
produces an answer with inline citations. When the initial retrieval
score falls below a threshold, the system searches again with terms from
the retrieved passages.

## Results

In the invented held-out set of 1,200 claims, the system achieved an nDCG@10 of
0.71, compared with 0.58 for a dense-only retriever and 0.49 for a sparse
baseline. The corrective loop contributed roughly four points of that
improvement. Fictional human ratings marked 88% of retrieved-source
answers as supported by their citations, against 62% for the baseline.
The example assigns an extra 180 milliseconds per query when retrieval
was retried. None of these values comes from an actual evaluation.

## Conclusion

The fictional results illustrate a comparison of retrieval methods and
cited answers. A real study would need recorded queries, source data,
model versions, timing conditions and human review before drawing
conclusions about answer accuracy.

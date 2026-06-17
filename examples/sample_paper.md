# Retrieval-Augmented Generation for Scientific Literature

## Abstract

We present a study on retrieval-augmented generation (RAG) applied to
scientific literature question answering. Large language models often
produce fluent but unsupported claims when asked about technical topics.
Our system grounds every answer in retrieved passages and reports a
confidence score derived from retrieval quality. Across a benchmark of
1,200 scientific claims, the grounded system reduced unsupported
statements by 41% relative to an ungrounded baseline while preserving
answer fluency. These results suggest that retrieval grounding is an
effective lever for trustworthy scientific assistants.

## Introduction

Scientific question answering demands both fluency and factual accuracy.
A model that summarizes a paper convincingly but misattributes its
findings is worse than useless, because the errors are hard to detect.
Prior work shows that retrieval grounding narrows this gap by forcing the
generator to condition on real source text. We extend that line of work
with a self-correction step that inspects retrieval quality before any
answer is generated, and abstains when the supporting evidence is weak.

## Methods

Our pipeline has three stages. First, documents are parsed and split into
overlapping chunks that preserve section structure. Second, a hybrid
retriever combines dense embeddings with sparse keyword matching to rank
candidate passages for a query. Third, a generator conditions on the
top-ranked passages and emits an answer alongside inline citations. A
corrective loop re-runs retrieval with a reformulated query whenever the
initial passages fall below a relevance threshold, which guards against
confident answers built on thin evidence.

## Results

On a held-out set of 1,200 claims, the full system achieved an nDCG@10 of
0.71, compared with 0.58 for a dense-only retriever and 0.49 for a sparse
baseline. The corrective loop contributed roughly four points of that
improvement, mostly on queries where the original phrasing was ambiguous.
Human raters judged 88% of grounded answers as faithful to their cited
sources, against 62% for the ungrounded baseline. Latency increased by
about 180 milliseconds per query when the corrective loop fired, which we
consider an acceptable cost for the gain in faithfulness.

## Conclusion

Grounding scientific question answering in retrieved evidence, and gating
generation on retrieval quality, yields answers that are both fluent and
verifiable. The main finding of this study is that a lightweight
corrective retrieval loop measurably reduces unsupported claims at modest
latency cost. Future work will extend the approach to multi-document
synthesis and to longer reasoning chains over several papers.

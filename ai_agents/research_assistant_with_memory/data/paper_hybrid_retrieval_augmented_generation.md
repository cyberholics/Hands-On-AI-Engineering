# Hybrid Sparse-Dense Retrieval for Grounded Generation

## Abstract

Retrieval-augmented generation systems condition a language model's output on passages retrieved
from an external corpus, which reduces hallucination and allows knowledge to be updated without
retraining the model. Most deployed systems choose either a sparse lexical retriever or a dense
neural retriever, but each has failure modes the other does not share: sparse retrieval misses
paraphrases and synonyms, while dense retrieval can miss rare entities, numbers, and exact-match
terms that were underrepresented during encoder training. We propose a hybrid retriever that fuses
sparse and dense rankings with reciprocal rank fusion, and a generator that is explicitly trained
to attribute claims to specific retrieved passages. We evaluate on long-form question answering and
multi-document summarization and find that the hybrid retriever improves both retrieval recall and
the factual grounding of generated text compared to either retriever used alone.

## Introduction

A generation model that answers questions purely from its parameters cannot cite a source, cannot
be corrected by updating a document store, and will confidently state facts that are wrong or
outdated. Retrieval-augmented generation is attractive precisely because it decouples what the
model knows from what it was trained on: the retriever supplies current, verifiable evidence, and
the generator's job narrows to synthesizing that evidence into fluent, grounded prose.

The retrieval step still determines the ceiling on system quality. Dense retrievers trained on
general question-answering data are strong at matching paraphrased intent but are known to
underperform on queries containing rare proper nouns, product codes, or numeric identifiers,
because these tokens are sparsely represented in the training distribution of the encoder. Sparse
retrievers handle exact-match terms well by construction but cannot bridge a vocabulary gap. This
motivates combining both signals rather than picking one.

## Method

Our retriever runs a sparse BM25 search and a dense bi-encoder search independently over the same
corpus and merges the two ranked lists with reciprocal rank fusion, which scores each passage by
the sum of the inverse of its rank in each list. This avoids the need to calibrate sparse and dense
similarity scores onto a common scale, which is a persistent difficulty with weighted score
combination.

Retrieved passages are passed to the generator with an explicit instruction to cite the passage
each claim is drawn from using inline bracketed markers. We additionally fine-tune the generator on
a small set of examples where every sentence in the target output is paired with its supporting
passage, so the model learns the citation behavior rather than being asked to produce it purely
through prompting.

## Results

The hybrid retriever improves passage recall at the top 10 results by 6 to 12 points over the
stronger of the two individual retrievers, with the largest gains concentrated on queries containing
named entities or numeric values that the dense retriever alone tends to miss. Human evaluators
judged hybrid-retrieval outputs as more fully grounded, with fewer unsupported claims, than outputs
generated from dense-only retrieval, even when both conditions were given the same number of
retrieved passages. Citation accuracy, meaning whether a cited passage actually supports the
sentence it is attached to, was also higher for the fine-tuned citation-aware generator than for a
generator prompted to cite without fine-tuning.

## Limitations

Reciprocal rank fusion has no learned parameters, which makes it robust but also unable to adapt
its balance of sparse and dense signal to a particular domain or query type; a learned fusion model
might do better but requires labeled relevance data to train. The citation-aware generator still
occasionally attaches a citation to a sentence that only partially follows from the cited passage,
particularly for sentences that combine information from two passages into one claim. Latency is
also higher than either retriever alone, since two full retrieval passes must complete before
fusion and generation can begin, which matters for applications with tight response-time budgets.

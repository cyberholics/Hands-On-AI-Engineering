# Dense Retrieval for Open-Domain Question Answering

## Abstract

Open-domain question answering systems must locate the small number of passages relevant to a
question inside a large document collection before an answer can be generated. Classical systems
rely on sparse lexical matching such as TF-IDF or BM25, which score passages by term overlap
with the query. This paper studies a purely dense retrieval approach in which questions and
passages are each encoded into a shared vector space by a bi-encoder, and relevant passages are
retrieved by nearest-neighbor search over passage embeddings. We show that a bi-encoder trained
with in-batch negatives on a modest number of question-passage pairs outperforms BM25 on passage
retrieval accuracy across several open-domain QA benchmarks, and that gains in retrieval quality
translate directly into gains in downstream answer accuracy when the retrieved passages are fed
to a reader model.

## Introduction

Retrieval is the first and most consequential step in an open-domain question answering pipeline.
If the retriever fails to surface a passage containing the answer, no downstream reader, however
capable, can recover. Sparse retrieval methods have been the default choice for decades because
they require no training, generalize across domains, and are fast at scale. Their weakness is
also well known: they cannot bridge a vocabulary gap between how a question is phrased and how
the answer passage is written. A question asking "who is the bad guy in the movie" will not match
a passage that only ever uses the word "antagonist."

Dense retrieval addresses this by learning a semantic representation of text rather than relying
on exact term overlap. The central design question is how to train the encoders so that questions
and their relevant passages land close together in vector space while irrelevant passages are
pushed apart, using only the supervision that is realistically available: pairs of questions and
one or a few gold passages, without hard negative labels for every passage in the collection.

## Method

We use two independent BERT-based encoders, one for questions and one for passages, and represent
each text as the vector of its [CLS] token. Training uses a contrastive objective: for each
question in a training batch, its gold passage is treated as the positive example, and every other
passage's gold passage in the same batch is treated as a negative example. This in-batch negative
sampling scheme is what makes training tractable, since it avoids having to define hard negatives
by hand while still exposing the model to a large number of negatives per step as batch size grows.

At inference time, every passage in the collection is encoded once and stored in an index that
supports approximate nearest-neighbor search. A question is encoded at query time and the index
returns the passages whose vectors are closest under inner product similarity. No further
re-ranking is applied in the base configuration, isolating the contribution of the dense encoders
themselves.

## Results

Across five open-domain QA datasets, dense retrieval improves top-20 passage retrieval accuracy by
9 to 19 points over a BM25 baseline, with the largest gains on datasets where questions are phrased
conversationally rather than as keyword-style queries. When the same retrieved passages are passed
to an extractive reader, end-to-end answer exact-match accuracy improves by a comparable margin,
confirming that retrieval quality is the binding constraint on system accuracy in this setting.
Combining dense and sparse scores with a simple linear combination yields a further small
improvement over either method alone, suggesting the two approaches make partially independent
errors.

## Limitations

The approach depends on having question-passage training pairs; without in-domain supervision,
transfer to a new domain with substantially different vocabulary or document structure is weaker
than the in-domain numbers suggest. The bi-encoder architecture also cannot model fine-grained
token-level interactions between the question and passage at scoring time, since the two are
encoded independently before comparison; a cross-encoder re-ranker recovers some of this
interaction but at a large increase in inference cost, since it must be run separately for every
candidate passage rather than once per query. Finally, the index must be fully rebuilt whenever
the passage collection changes, which is expensive for corpora that update frequently.

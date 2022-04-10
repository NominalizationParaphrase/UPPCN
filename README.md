# Unsupervised Paraphrasability Prediction for Compound Nominalizations

## Abstract
Commonly found in academic and formal texts, a nominalization uses a deverbal noun to describe an event associated with its corresponding verb. Nominalizations can be difficult to interpret because of ambiguous semantic relations between the deverbal noun and its arguments. Automatic
generation of clausal paraphrases for nominalizations can help disambiguate their meaning. However, previous work has not identified cases where it is awkward or impossible to paraphrase a compound nominalization. This paper investigates unsupervised prediction of paraphrasability to determine whether
the prenominal modifier can be re-written as a noun or adverb in a clausal paraphrase. We adopt the approach of overgenerating candidate paraphrases followed by candidate ranking with a neural language model. In experiments on an English dataset, we show that features from an Abstract Meaning Representation graph lead to statistically significant improvement in both paraphrasability prediction and paraphrase generation.

## Tools
1. Spacy == 2.2.4
2. Pandas
3. BERT-based Masked Language Model : https://demo.allennlp.org/masked-lm
4. TextualEntailment : https://demo.allennlp.org/textual-entailment/roberta-snli
5. GPT & DistilBert : https://github.com/awslabs/mlm-scoring
6. Sentence Embeddings : https://huggingface.co/sentence-transformers/stsb-roberta-large
7. AMR : https://github.com/ufal/perin

# Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates

Conference: ACL 2018
Paper review process: Done
Publish month(arxiv): 2018.04
Tags: nlp, tokenization

### Overview

- A sentence can be represented in multiple subword sequences even with the same vocabulary. Previous methods don’t consider this problem.
- Instead of maximizing the log likelihood on a single subword sequence with the largest likelihood, the authors propose an efficient subword regularization method that maximizes the likelihood of all possible combinations.
- The method improves machine translation performance. The results are even more significant on the OOD corpus.

### (motivation) A sentence can be represented in multiple subword sequences even with the same vocabulary

![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled.png)

BPE encodes a sentence into a unique subword sequence. Suppose in an NMT scenario where X is the source language passage and Y is the target language passage. Expressing `P(Y | X)` as the formula above for the sequence of subwords X, Y can be incorrect when X, Y can actually be multiple subword sequences. 

![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%201.png)

Subword regularization acknowledges the issue and addresses it by randomly sampling subword sequences.

### (Key idea) A new regularization method for open-vocabulary NMT

1. Standard NMT is trained to maximize the log-likelihood L(θ)
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%202.png)
    
2. To consider that X, Y can be segmented into multiple subword sequences, we should like to optimize L_marginal.
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%203.png)
    
3. Since exact optimization of L_marginal is impossible, we sample `k` subword sequences from P(x | X) and P(y | Y) and optimize the log-likelihood for these sequences. 
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%204.png)
    
4. In particular, a single randomized segment is sampled for x, y (i.e. `k = 1`).

### (method) n-best decoding

During decoding, the authors suggest an **n-best decoding** algorithm:

1. Given n-best subword sequences of X (x_1, … x_n), 
2. the best translation y* should have the largest length-normalized log-likelihood during decoding. In particular, y* should maximize the following score for some x.
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%205.png)
    

### (method) Unigram language model for subword segmentation

**Motivation**

Using BPE-based tokenizers to compute `P(x | X)` for subword normalization is non-trivial because it is based on a greedy and deterministic encoding method. A **unigram language model** that assumes that each subword occurs independently is needed to model the probability of each subword sequence `P(x | X)`.

**Definition**

![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%206.png)

The authors propose a unigram language model that assumes that each subword occurs independently. Here, you can write `P(x)` of a subword sequence `x` as the product of the probability of each subword occurring..

![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%207.png)

The most probable sequence x* from `S(X)` a set of segmentation candidates built from X can be obtained efficiently using the [Viterbi algorithm](https://ratsgo.github.io/data%20structure&algorithm/2017/11/14/viterbi/).

**Training unigram model**

![Screen Shot 2023-03-05 at 3.52.40 PM.png](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Screen_Shot_2023-03-05_at_3.52.40_PM.png)

The unigram model should be trained to maximize the likelihood L above. The algorithm jointly optimizes vocabulary set `V` and occurrence probabilities:

1. Make a reasonably big seed vocabulary from the training corpus.
    1. All single characters are included.
    2. Most frequent substrings in the corpus which is similar to WordPiece. 
2. Repeat the following steps until `|V|` reaches the desired vocabulary size.
    1. Optimize p(x) with the EM algorithm given `V`.
    2. For each subword `x_i`, compute how much the likelihood L is reduced when `x_i` is removed from the current vocabulary. 
    3. Keep the top η% of subwords (η=80%). 

### (method) **Sampling subwords**

**segmentation means subword sequence*

1. Sample `l-best` segmentations according to the probability P(x | X)
2. Single(because `k=1`) segmentation is sampled from the `l-best` segmentations weighted on smoothed probabilities.
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%208.png)
    

- `l-best` search is performed in linear time with the Forward-DP Backward-A* algorithm
- To sample all possible segmentations(for `l → ∞` experiments) for exact computation, the authors describe a Forward-Filtering and Backward-Sampling(FFBS) algorithm.

## Results

- Main result on NMT
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%209.png)
    
    - Unigram LM without subword regularization(l=1) and BPE show similar performance.
    - `l=64` shows significant BLEU improvements.
    - n-best decoding provides further gains for `l ≠ 1` , but degrades the performance for `l = 1`. This indicates that the model is confused about multiple segmentations when they are not explored at training time.
- Results on OOD corpus
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%2010.png)
    
    - The results are even more significant on the OOD corpus.
- Hyperparameter: subword regularization has two hyperparameters: `l`: size of sampling candidates, `α`: smoothing constant.
    - Tuning the value of `α` < 1 was crucial.
    
    ![Untitled](Subword%20Regularization%20Improving%20Neural%20Network%20Tr%207f23ecbb8d904733816e65b805d87248/Untitled%2011.png)
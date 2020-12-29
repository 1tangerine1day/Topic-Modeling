# Topic Modeling

[![hackmd-github-sync-badge](https://hackmd.io/EB5NqKdiQvKITSTmuxo7-g/badge)](https://hackmd.io/EB5NqKdiQvKITSTmuxo7-g)

$p(word|corpus) = \sum_{topic}p(word|topic)*p(topic|corpus)$
## NFM (Non-negative Factor Model):
* MF (matrix factorization):

    $M = U*I$
    * $M$ : rating matrix / tf-idf matrix (m*n)
    * $U$ : user matrix / corpus' latent vector (m*k)
    * $I$ : iteam matrix / words' latent vector (k*n)
    * $k$ : number of topic

## LSA\LSI (Latent Semantic Analysis\Latent Semantic Indexing)
* SVD decomposition: 

    $M = U S \bar{I}^T$ , $M$ can be real or complex matrix (m*n)
    * $U$ : real or complex unitary matrix (m*m)
    * $S$ :  rectangular diagonal matrix with non-negative real numbers on the diagonal (m*n)
    * $I$ : real or complex unitary matrix (n*n)
    <br><img src="https://i.imgur.com/E4rPrwM.png" width="50%" />

* unitary matrix:

    $M\bar{M}^T = \bar{M}^TM = I$

* LSA(matrix factorization):

    $M = U S \bar{I}^T$ -> $M = \hat{U} \hat{S} \hat{I}$ 
    
    * $\hat{U}$ : initial randamly (m*r)
    * $\hat{S}$ : keep the non zero value of $S$ (r*r)
    * $\hat{I}$ : initial randamly (r*n) 
    <br><img src="https://i.imgur.com/ZMeTv0u.png" width="70%" />

* LSA(dimension reduction、truncated SVD):
    Sort the singular values in $\hat{S}$ and keep the first $k$ elements
    <br><img src="https://i.imgur.com/jgdiuec.png " width="70%" />
    
* LSA(prediction):
    <br><img src="https://i.imgur.com/c6p8ldV.png" width="70%" />
    
    for topic modeling:
    * Rating matrix -> tf-idf/bag-of-words matrix (corpus * words)
    * Users' preformance matrix -> corpus' latent vector
    * Items' features matrix -> words' latent vector
    * $k$ -> number of topic

* optimize:
    * root-mean-square error
    * stochastic gradient descent

## LDA(Latent Dirichlet Allocation)
* Plate Notation
    <br><img src="https://i.imgur.com/PHl0yAB.png" width="70%" />
    * $K$ number of topic
    * $M$ corpus
    * $N$ words
    * $Z_{ij}$ is the topic for the $j$-th word in document $i$
    * $W_{ij}$ is the specific word (observed word)

* Generative Process
    1. Choose $θ_i〜Dir(\alpha)$, where $i \in \{1,2 ... M\}$
    2. Choose $φ_k〜Dir(\beta)$, where $k \in \{1,2 ... K\}$
    3. For each of word position $i,j$, where $i \in \{1,2 ... M\}$ and $j \in \{1,2 ... N_i\}$
        * choose a topic $Z_{ij}〜Multinomial(θ_i)$
        * choose a word $W_{ij}〜Multinomial(φ_{z_{i,j}})$

## dataset
* newAPI: https://newsapi.org/
    * Apple
    * FB
    * Google

## test case
* NMF
    ```
    Topic 0 : ['despite', 'plea', 'Kardashian', 'Kim', 'execution', 'Bernard', 'Indiana', 'Brandon', 'federal', 'EU']
    Topic 1 : ['Trump', 'president', 'elect', 'run', 'mate', 'finalist', 'Donald', 'beat', 'include', 'safe']
    Topic 2 : ['阅读全文', 'fall', 'foul', 'billionaire', 'profile', 'controversial', 'law', 'democracy', 'medium', 'figure']
    Topic 3 : ['home', 'family', 'shoot', 'dead', 'Mr', 'dentist', 'Goodson', 'appointment', 'de', 'say']
    Topic 4 : ['Apple', 'new', 'year', 'Google', 'event', 'vaccine', 'rise', 'smart', 'find', 'daily']
    ```
* LSI
    ```
    Topic 0 : ['return', 'say', 'vaccine', 'appointment', 'Mr', 'Goodson', 'dentist', 'shoot', 'dead', 'family']
    Topic 1 : ['high', 'White', 'fatality', 'House', 'daily', 'hold', 'coronavirus', 'relate', 'rise', 'record']
    Topic 2 : ['vaccine', 'Johnson', 'Boris', 'negotiation', 'EU', 'continue', 'Drug', 'deem', 'Food', 'adviser']
    Topic 3 : ['finalist', 'mate', 'Donald', 'elect', 'Trump', 'beat', 'president', 'run', 'include', 'Boris']
    Topic 4 : ['Indiana', 'Bernard', 'execution', 'Kim', 'Kardashian', 'Brandon', 'plea', 'federal', 'despite', 'set']
    ```
    
* LDA
    ```
    Topic 0 : ['vaccine', 'Administration', 'deem', 'Drug', 'Food', 'adviser', 'panel', 'safe', 'effective', 'Apple']
    Topic 1 : ['Apple', 'Google', 'Black', 'deal', 'new', 'Facebook', 'Friday', 'return', 'good', 'app']
    Topic 2 : ['Apple', 'figure', 'profile', 'billionaire', 'foul', 'democracy', 'law', 'controversial', 'fall', 'medium']
    Topic 3 : ['Apple', 'continue', 'photo', 'thousand', 'ask', 'categorise', 'Great', 'Barrier', 'volunteer', 'Reef']
    Topic 4 : ['Apple', 'rise', 'event', 'say', 'Google', 'relate', 'hold', 'fatality', 'White', 'House']
    ```


### reference
* https://medium.com/ai-academy-taiwan/svd-%E5%AF%A6%E4%BD%9C%E6%8E%A8%E8%96%A6%E7%B3%BB%E7%B5%B1-f90f98b9831b
* https://blog.rosetta.ai/%E6%B7%B1%E5%85%A5%E6%8E%A2%E8%A8%8E-latent-dirichlet-allocation-lda-%E8%88%87%E5%9C%A8%E6%8E%A8%E8%96%A6%E7%B3%BB%E7%B5%B1%E4%B8%8A%E7%9A%84%E6%87%89%E7%94%A8-2441d57ecc8a
* https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
* In practice, LSI is much faster to train than LDA, but has lower accuracy. (https://www.quora.com/Whats-the-difference-between-Latent-Semantic-Indexing-LSI-and-Latent-Dirichlet-Allocation-LDA)

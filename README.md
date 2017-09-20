# The Graphs in Machine Learning List

This is a list of resources, links and papers on the use of Graph Representations in Machine Learning. It actually extends beyond that as we see some cases and examples are actually about the interseciton of these two things. Graphs are useful in machine learning problems, 
machine learning is useful in the analysis of Graphs and data sets that can be representared as graphs.

**could we mke this an [AWESOME] list?** 

## Areas of Intersection

### Graph Analysis Problems (that may have ML approaches/solutions)

 - Graph Classification
   - Classification of data/entities based on the nature of their Graph, each full (sub-)graph is assigned a class label
   - Data / Application Spaces
        - Predictive toxocology
   - References
        - https://www.csc2.ncsu.edu/faculty/nfsamato/practical-graph-mining-with-R/slides/pdf/Classification.pdf
 - Vertex Classification
    - Within a graph, each vertex is assigned a class label
 - Graph Based Similarity
    - Determining Graph Isomorphisms are hard (NP-complete/hard? what's teh complexity here), we often need to approximate, we can take ML based approaches to do so.
    - use vector based classification via the kernel trick, SVMs?
 - Frequent Subgraph Mining
    - Explanation - http://data-mining.philippe-fournier-viger.com/introduction-frequent-subgraph-mining/
    - Boosting with gBoost - https://link.springer.com/article/10.1007/s10994-008-5089-z
 - Graph based Anomaly Detection
    - a survey paper: https://link.springer.com/article/10.1007%2Fs10618-014-0365-y

### Machine Learning on Graphs

  - GNN Community - Community detection with Graph Neural Networks **[maybe demoable]**
    - [paper](https://arxiv.org/pdf/1705.08415.pdf)
    - [github](https://github.com/joanbruna/GNN_community)

  - Link Prediction in Convolutional Neural Networks
   - [paper](https://arxiv.org/pdf/1706.02263.pdf)

  - Graph Convolutional Networks - Thomas Kipf
   - [talk](https://github.com/tkipf/gcn)
   - [github](https://github.com/tkipf/gcn)
   - [website](http://tkipf.github.io/graph-convolutional-networks/)
   - [paper](https://openreview.net/pdf?id=SJU4ayYgl)

  - Spectrum-based deep neural networks for fraud detection
   - [paper](https://arxiv.org/abs/1706.00891)

  - CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters
   - [paper](https://arxiv.org/pdf/1705.07664.pdf)

  - Modularity based Community Detection with Deep Learning
   - [paper](http://www.cs.wustl.edu/~zhang/publications/ijcai16-DL.pdf)

  - Geometric Deep Learning
   - [website](http://geometricdeeplearning.com/)     

### Graph based methods in Machine Learning

 - Spectral Features
   - references:
     - http://www.machinelearning.org/proceedings/icml2007/papers/444.pdf

 - Learning Graph Representations
  Learning vertex representations of Graph Structure

  - Learning deep representations for graph clustering
    Use deep neural networks (stacked autoencoder) to learn features
    - https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527

  - (Deep Neural Networks for Learning Graph Representations)
  Learn low dimensional representation on a vertex that encodes the local structure of the graph. Could also be described as graph embeddings.
  [https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12423/11715]

 
 - node2vec - Scalable Feature Learning for Networks **[maybe demoable]**

   description: node2vec is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks.
   - [website](http://snap.stanford.edu/node2vec/)
   - [paper](https://arxiv.org/abs/1607.00653)
   - [github](https://github.com/aditya-grover/node2vec)
   
 - graph2vec - Learning Distributed Representations of Graphs **[maybe demoable]**
   - description:
   - [paper](https://arxiv.org/abs/1707.05005)
   - [github](https://github.com/allentran/graph2vec)
      
 - DeepWalk - Online Learning of Social Representations **[maybe demoable]**
    DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information.
    Unweighted graphs.
   - [paper](https://arxiv.org/abs/1403.6652)
   - [github](https://github.com/phanein/deepwalk)
   
 ### Graph Based Semi Supervised Learning
  
  - [Introductory Presentation](http://mlg.eng.cam.ac.uk/zoubin/talks/lect3ssl.pdf)
    - Graph based semi-supervised learning
    - Active graph-based semi supervised learning
 
  - Semi-Supervised Learning with graphs
   - (thesis)[http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf]

  - Book - Introduction to Semi-Supervised Learning
   - (book chapter on based-based)[https://books.google.es/books?id=c_haJrQ0ScAC&pg=PA44&lpg=PA44&dq=learn+graph+edge+weighting&source=bl&ots=KY2SR00zjX&sig=jA0Vfp7bgEVjsOpjZJ7gqx6CYDc&hl=en&sa=X&ved=0ahUKEwiHmLnKg4HWAhUGZlAKHZ81ASo4ChDoAQhEMAU#v=onepage&q=learn%20graph%20edge%20weighting&f=false]


### graph-based machine learning

 - Adaptive edge weighting for graph-based learning algorithms
   - (paper)[https://research.aalto.fi/files/11519875/art_10.1007_s10994_016_5607_3.pdf]

 - 

### Temporal Graphs

 - Structural-RNN: Deep Learning on Spatio-Temporal Graphs
   - https://arxiv.org/abs/1511.05298

 - greycat.ai
   - https://github.com/datathings/greycat

 - learning spatio temporal varying data
   - http://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-083.pdf

### Application Areas

 - Predictive Toxocology
 - Computer Network Analysis / Management
    - Resouce Allocation
    - Fault tolerance
    - Security
    - Performance Monitoring
 - Social Network Analysis
 - Fraud Detection
 - Transactional Data Analysis
 - Supply Chain Optimisation
 



## Other Areas

### Graph Signal Processing

 - Common tasks in signal-processing on graphs include filtering, de-noising, in-painting, compression, clustering, partitioning, sparsification, features extraction, classification and regression.
   - References: 
     - http://perraudin.info/gsp.php
     - https://arxiv.org/pdf/1210.4752.pdf



## Software

### Graph Analysis

  - [SPMF](http://www.philippe-fournier-viger.com/spmf/) - Data Mining Library 


## Datasets

  - [WEBKB](http://www.cs.cmu.edu/~webkb/) a dataset of classified web pages from University Science detpartment's websites.
  - [PTC](https://relational.fit.cvut.cz/dataset/PTC) predictive toxocology
  - [SNAP](http://snap.stanford.edu/data/index.html) Stanford Large Network Dataset Collection
  - Datasets used in the [node2vec](http://snap.stanford.edu/node2vec/#datasets)
    - Blog Catalog
    - Protein-Protein Interaction
    - Wikipedia Network
    - Facebook Network
    - arXiv ASTRO-PH
  

## Other lists

 - [Awesome Network Analysis](https://github.com/briatte/awesome-network-analysis)
 - [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)







## Prerequisites
These are are the following prerequise to run this project.
1)Python 2.7
2)Numpy
3)NLTK
## About
I have implement unsupervised approach for the finding the summarise text of an input paragraph.The approach which I used to solve this problem is Text Rank model it is an Graph based ranking algorithm.A graph based ranking is done on the basis of links to one vertex to another the higher the link the higher the votes will on the basis of that we calculate the sentences which have the highest dependency in the contest. 
## How to run ?
1. Clone the repository
2. Run the following script
```
    python run.py --p '<Directory of the file store>'
```

## Text Rank Model
Text Rank Model is a ranking algorithms based on graph which give a way of deciding the important context in a text. Graph based ranking algorithm used voting for deciding weight of a text. The linkage of vertex to another one,is basically generating a vote a that particular vertex.Higher the number of votes that are cast for a vertex, the higher the importance
of the vertex. Moreover, the importance of the vertex casting the vote determines how important the vote itself is, and this information is also taken into account by the ranking model. Hence, the score associated with a vertex is determined based on the votes that are cast for it, and the score of the vertices casting these votes.

For Text summarization ,our goal is to rank entire sentence and therefore vertex is added to the graph of each sentence in the text.The content overlap is use to derive the relation of the sentences.Which find out the intersection of word in a sentence.Such a relation between two sentences can be seen as a process of “recommendation”: a sentence that addresses certain concepts in a text, gives the reader a recommendation refer to other sentences in the text that address the same concepts, and therefore a link can be drawn between any two such sentences that share common content.The overlap of two sentences can be determined simply as the number of common tokens between the lexical representations of the two sentences, or it can be run through syntactic filters, which only count words of a certain syntactic category, e.g. all open class words, nouns and verbs, 

## Implementation

### Date Preparation 
In this phase we take CNN NEWS data-set and we get tokenize with sentence (Bi-Gram ) tokenizer and on filtering sentence we do normalize each and every sentences by removing unicode character and stop words which create noise during training.After that we add vocabulary to the text by using POS tags.
### Load Data Function 
Load data function is use to read text file and split the text file into train text and its corresponding label by using split function the encoding is UTF-8 and tokenize the sentence.It split the data into two text and its corresponding label.
```
def load_data(path):
    sentences = []
    ftext=io.open(path, encoding='utf-8')
    rest=ftext.split('@highlight')
    print(rest[1:])
    with ftext as f:
        for line in f:
            line = line.strip()
            if line and not is_heading(line):
                for sent in sent_tokenize(line):
                    sentences.append(sent)
    
    return sentences
```

### Text Summerization
Once we did that an make co existing matrix which gives how many time a particular token repeat it self in n gram based on that an adjacency matrix is drawn.From that adjacency matrix we make similarity tree which find out particular token repeat itself.On completing that We apply Text rank model.



### Similarity Matrix Function
The build_similarity_matrix function is responsible generating an similarity matrix which give the it call the sentence similarity function for each sentence. To get sentence similarity scale.

```

def build_similarity_matrix(sentences):
    S = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            
            S[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return normalize_matrix(S)
```
### Sentence Similarity
The sentence similarity function find out the overlap in an sentence which give the similarity score of two sentences.
```

def sentence_similarity(sent1, sent2):
    overlap = len(set(sent1).intersection(set(sent2)))

    if overlap == 0:
        return 0
    
    return overlap / (np.log10(len(sent1)) + np.log10(len(sent2)))
```

### Pagerank Function
The PageRank function took similarity matrix parameter an calculate the score of each vertex here  d is a damping factor that can be set between
0 and 1, which has the role of integrating into the model the probability of jumping from a given vertex to another random vertex in the graph. On the other hand eps stop the algorithm when difference of two consecutive vertex are equal to or smaller than eps value. 
```
def pagerank(A, eps=0.0001, d=0.85):
    R = np.ones(len(A))
    
    while True:
        r = np.ones(len(A)) * (1 - d) + d * A.T.dot(R)
        if abs(r - R).sum() <= eps:
            return r
        R = r
```
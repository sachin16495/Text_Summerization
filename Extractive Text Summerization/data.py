import io

from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.tokenize import sent_tokenize

eos_tokens = set([".", "!", "?"])

pos = {
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
}

stemmer = porter.PorterStemmer()

stop_words = set(stopwords.words("english"))


tags = set(["NN", "NNS", "NNP", "JJ", "JJR", "JJS"])

def build__text__vocabulary(sentences):
    word_to_ixp = {}
    ix_to_words = {}
    
    for sent in sentences:
        for word in sent:
            if word not in word_to_ixp:
                word_to_ixp[word] = len(word_to_ixp)
                ix_to_words[len(ix_to_words)] = word
    
    return word_to_ixp, ix_to_words

def filter_sentences(sentences, lowercase=True, stem=True):
    norm_sents = [normalize_sentence(s, lowercase) for s in sentences]
    filtered_sents = [filter_words(sent) for sent in norm_sents]

    if stem:
        return [stem_sentence_create(sent) for sent in filtered_sents]
    
    return filtered_sents

def filter_words(sentence):
    filtered_sentence = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag not in tags:
            continue
        
        if word.lower() in stop_words:
            continue
        
        filtered_sentence.append(word)
    
    return filtered_sentence

def is_heading(s):
    if s[-1] in eos_tokens:
        return False
    
    return True

def load_data(path):
    sentences = []
    ftext=io.open(path, encoding='utf-8')
    ftex=io.open(path, encoding='utf-8')
    rest=(ftex.read()).split('@highlight')#, encoding='utf-8')
    print("Text")
    print(rest[1])
    print("Actual Summarry")
    str= ''.join(e for e in rest[1:])
    print(str)
    with ftext as f:
        for line in f:
            line = line.strip()
            if line and not is_heading(line):
                for sent in sent_tokenize(line):
                    sentences.append(sent)
    
    return sentences

def normalize_sentence(sentence, lowercase=True):
    if lowercase:
        sentence = sentence.lower()
    
    return sentence.replace(u"\u2013", u"-").replace(
        u"\u2019", u"'").replace(u"\u201c", u"\"").replace(
        u"\u201d", u"\"")

def stem_sentence_create(sentence):
    return [stemmer.stem(word) for word in sentence]
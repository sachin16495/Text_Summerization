import argparse
from data import build__text__vocabulary
from data import filter_sentences
from data import load_data
from model import build_coo_matrix_generate
from model import build_similarity_matrix
from model import get_topk_sentences_generate
from model import pagerank
import os

def summarize(sentences, k=5):
    filtered_sentences = filter_sentences(sentences)

    S = build_similarity_matrix(filtered_sentences)

    ranks = pagerank(S)

    return get_topk_sentences_generate(ranks, sentences, k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to text direcory")
    parser.add_argument("-l", "--len", type=int,
        help="Number of keywords and sentences to extract")

    args = parser.parse_args()
    print("Current Path")
    print(os.getcwd())
    k = 5
    if args.len:
        k = args.len
    file_path=args.path
    filr=os.listdir(file_path)
    i=0
    for sn in filr:
        summary=''
        print(sn)
        sentences = load_data(file_path+'/'+sn)
        summary = summarize(sentences, k)
        summary=" ".join(summary)
            #print(" ".join(summary))
            #return
        i=i+1
        #print("; ".join(extract_keywords(sentences, k)))
        print("Predicted summary")
        print(summary)
        if i==500:
            break
    
    

if __name__ == '__main__':
    main()
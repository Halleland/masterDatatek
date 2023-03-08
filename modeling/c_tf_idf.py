import re
import nltk
import numpy as np
from tqdm.auto import tqdm

def get_tokens_for_docs(docs):
    # first lowercase and remove punctuation
    alpha = re.compile(r'[^a-zA-Z ]+')
    docs['context_page_description'] = docs['context_page_description'].apply(lambda x: 
        alpha.sub('', x.lower())
    )
    tokens=docs['context_page_description'].apply(lambda x: 
        nltk.tokenize.wordpunct_tokenize(x)
    )
    return tokens

def remove_stopwords(classes):
    nltk.download('stopwords')
    for c in classes.keys():
        stopwords = set(nltk.corpus.stopwords.words('english'))
        # stopwords from nltk are all lowercase (so are our tokens)
        classes[c]['tokens'] = [
            word for word in classes[c]['tokens'] if word not in stopwords
        ]
    return classes

def find_vocab(classes):
    vocab = set()
    for c in classes.keys():
        vocab = vocab.union(set(classes[c]['tokens']))
        classes[c]['vocab'] = set(classes[c]['tokens'])
    return vocab

def find_c_tf_idf_from_hdbscan_model(df, labels):
    classes = {}
    for label in set(labels):
        classes[label] = {
            'vocab': set(),
            'tokens': [],
            'tfidf_array': None
        }

    docs = df['context_page_description'].to_frame()
    docs['class'] = labels

    docs['tokens'] = get_tokens_for_docs(docs)
    docs.apply(lambda row:
        classes[row['class']]['tokens'].extend(row['tokens']), axis=1
    )

    classes = remove_stopwords(classes)
    vocab = find_vocab(classes)
    
    tf = np.zeros((len(classes.keys()), len(vocab)))

    for c, _class in enumerate(classes.keys()):
        for t, term in enumerate(tqdm(vocab)):
            tf[c, t] = classes[_class]['tokens'].count(term)

    idf = np.zeros((1, len(vocab)))

    # calculate average number of words per class
    A = tf.sum() / tf.shape[0]

    for t, term in enumerate(tqdm(vocab)):
        # frequency of term t across all classes
        f_t = tf[:,t].sum()
        # calculate IDF
        idf_score = np.log(1 + (A / f_t))
        idf[0, t] = idf_score
    
    tf_idf = tf*idf
    return tf_idf, classes

def get_top_terms(tf_idf, classes, n=5):
    top_idx = np.argpartition(tf_idf, -n)[:, -n:]
    vocab = find_vocab(classes)
    vlist = list(vocab)
    top_terms = []
    for c, _class in enumerate(classes.keys()):
        topn_idx = top_idx[c, :]
        topn_terms = [vlist[idx] for idx in topn_idx]
        top_terms.append(topn_terms)
    return top_terms
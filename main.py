from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import completeness_score, v_measure_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
import seaborn as sns


def clustering_test(x, y, method='kmeans', x_test=None, y_test=None):
    result = {}
    if method == 'kmeans':
        clustering = KMeans(n_clusters=len(set(y)))
    elif method == 'dbscan':
        clustering = DBSCAN()
    elif method == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=len(set(y)))
    else:
        raise ValueError('Unknown method')
    predicted = clustering.fit_predict(x)
    result['v_measure_score'] = v_measure_score(y, predicted)
    result['completeness_score'] = completeness_score(y, predicted)
    result['n_clusters'] = len(set(predicted))
    t = TSNE().fit_transform(x)
    result['scatter_real'] = scatter(t, y)
    result['scatter_predicted'] = scatter(t, predicted)
    if x_test is not None and y_test is not None:
        predicted_test = clustering.predict(x_test)
        result['v_measure_test'] = v_measure_score(y_test, predicted_test)
        result['completeness_score_test'] = completeness_score(y_test, predicted_test)
    return result


def scatter(x, target):
    palette = np.array(sns.color_palette("hls", len(set(target))))
    figure = plt.figure(figsize=(15, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=30,
                    c=palette[np.array(target)])
    return figure


def count_words(data, target):
    result = defaultdict(dict)
    for id, text in enumerate(data):
        current_target = target[id]
        for word in word_tokenize(text):
            result[current_target][word] = result[current_target].get(word, 0) + 1
    return result


def get_top_words_by_cluster(result_dict, count=3):
    result = dict()
    for cluster in result_dict:
        result[cluster] = sorted(result_dict[cluster], key=lambda x: result_dict[cluster][x], reverse=True)[:3]
    return result

def get_stemmed_text(text):
    porter = PorterStemmer()
    return " ".join((porter.stem(word) for word in word_tokenize(text)))


def get_stemmed_data(data):
    return [get_stemmed_text(text) for text in data]


if __name__ == '__main__':
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    data_source = train.data[:100]
    target = train.target[:100]
    vectorizers = {'count': CountVectorizer, 'tf-idf': TfidfVectorizer}
    data = {'raw': data_source, 'stemmed': get_stemmed_data(data_source)}
    vec = dict()
    for data_type in data:
        vec[data_type] = dict()
        for v_type in vectorizers:
            vectorizer = vectorizers[v_type]()
            vec[data_type][v_type] = vectorizer.fit_transform(data[data_type]).toarray()
    results = {}
    for data_type in data:
        results[data_type] = dict()
        for v_type in vectorizers:
            results[data_type][v_type] = dict()
            for method in ('kmeans', 'dbscan', 'agglomerative'):
                results[data_type][v_type][method] = clustering_test(vec[data_type][v_type], target, method)
    for data_type in results:
        print(f"\n{data_type}")
        for vec_type in results[data_type]:
            print(f"\n{vec_type}")
            for method in results[data_type][vec_type]:
                print(f"{method}: {results[data_type][vec_type][method]}")

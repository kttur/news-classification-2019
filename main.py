from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import completeness_score, v_measure_score
from matplotlib import pyplot as plt
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
    if x_test is not None and y_test is not None:
        predicted_test = clustering.predict(x_test)
        result['v_measure_test'] = v_measure_score(y_test, predicted_test)
        result['completeness_score_test'] = completeness_score(y_test, predicted_test)
    return result


def scatter(x, target):
    palette = np.array(sns.color_palette("hls", len(set(target))))
    f = plt.figure(figsize=(15, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=30,
                    c=palette[np.array(target)])
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    return f


if __name__ == '__main__':
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    count_vectorizer = CountVectorizer()
    tf_idf_vectorizer = TfidfVectorizer()
    vec = dict()
    vec['count'] = count_vectorizer.fit_transform(train.data[:500]).toarray()
    vec['tf-idf'] = tf_idf_vectorizer.fit_transform(train.data[:500]).toarray()
    results = {}
    for vectorizer in ('count', 'tf-idf'):
        results[vectorizer] = {}
        for method in ('kmeans', 'dbscan', 'agglomerative'):
            results[vectorizer][method] = clustering_test(vec[vectorizer], train.target[:500], method)
    for vectorizer in results:
        print(f"\n{vectorizer}")
        for method in results[vectorizer]:
            print(f"{method}: {results[vectorizer][method]}")

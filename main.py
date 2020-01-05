from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import completeness_score, v_measure_score


def clustering_test(x, y, method='kmeans', x_test=None, y_test=None):
    result = {}
    if method == 'kmeans':
        clustering = KMeans(n_clusters=len(set(y)))
    else:
        raise ValueError('Unknown method')
    predicted = clustering.fit_predict(x)
    result['v_measure_score'] = v_measure_score(y, predicted)
    result['completeness_score'] = completeness_score(y, predicted)
    if x_test is not None and y_test is not None:
        predicted_test = clustering.predict(x_test)
        result['v_measure_test'] = v_measure_score(y_test, predicted_test)
        result['completeness_score_test'] = completeness_score(y_test, predicted_test)
    return result


if __name__ == '__main__':
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    count_vectorizer = CountVectorizer()
    tf_idf_vectorizer = TfidfVectorizer()
    matrices = dict()
    matrices['count'] = count_vectorizer.fit_transform(train.data)
    matrices['tf-idf'] = tf_idf_vectorizer.fit_transform(train.data)
    results = {}
    for vectorizer in ('count', 'tf-idf'):
        results[vectorizer] = {}
        for method in ('kmeans', ):
            results[vectorizer][method] = clustering_test(matrices[vectorizer], train.target, method)
    print(results)

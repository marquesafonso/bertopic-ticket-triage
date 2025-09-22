import os
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP

def train_supervised_classifier(
        docs,
        y,
        class_weight_dict=None
    ):
    basepath = os.getcwd()
    topic_model_path = os.path.join(basepath, os.path.normpath("baseline_topic_model/"))
    os.makedirs(topic_model_path, exist_ok=True)

    # Create a base classifier
    empty_dimensionality_model = BaseDimensionalityReduction()
    if class_weight_dict:
        clf = LogisticRegression(class_weight=class_weight_dict)
    else:
        clf = LogisticRegression()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    count_vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2))

    # Create a fully supervised BERTopic instance
    topic_model= BERTopic(
            umap_model=empty_dimensionality_model,
            hdbscan_model=clf,
            vectorizer_model=count_vectorizer,
            ctfidf_model=ctfidf_model
    )
    topic_model.fit_transform(docs, y=y)
    topic_model.save(topic_model_path, serialization="safetensors", save_ctfidf=False)
    return topic_model

def train_zshot_model(docs, zeroshot_topic_list):
    basepath = os.getcwd()
    topic_model_path = os.path.join(basepath, os.path.normpath("zshot_topic_model/"))
    topic_model = BERTopic(
        embedding_model="ibm-granite/granite-embedding-small-english-r2",
        umap_model=UMAP(n_neighbors=15, n_components=6, min_dist=0.15, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=4, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1,2)),
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=.85,
        representation_model=KeyBERTInspired()
    )
    topic_model.fit_transform(docs)
    topic_model.save(topic_model_path, serialization="safetensors", save_ctfidf=False)
    return topic_model
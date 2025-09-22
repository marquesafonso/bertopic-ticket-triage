import os
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
# from bertopic.representation import KeyBERTInspired

# from umap import UMAP
# from hdbscan import HDBSCAN
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer

def train_supervised_classifier(
        docs,
        y,
        class_weight_dict=None
    ):
    basepath = os.getcwd()
    topic_model_path = os.path.join(basepath, os.path.normpath("topic_model/"))
    os.makedirs(topic_model_path, exist_ok=True)

    # Create a base classifier
    empty_dimensionality_model = BaseDimensionalityReduction()
    if class_weight_dict:
        clf = LogisticRegression(class_weight=class_weight_dict)
    else:
        clf = LogisticRegression()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Create a fully supervised BERTopic instance
    topic_model= BERTopic(
            umap_model=empty_dimensionality_model,
            hdbscan_model=clf,
            ctfidf_model=ctfidf_model
    )
    topic_model.fit_transform(docs, y=y)
    topic_model.save(topic_model_path, serialization="safetensors", save_ctfidf=False)
    return topic_model
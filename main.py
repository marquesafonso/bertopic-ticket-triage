import logging
from src.dataset import load_dataset
from src.model import train_supervised_classifier

def main():
    logging.basicConfig(
        level=logging.INFO, 
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    ## Prepare datasets
    logging.info("Prepare datasets...")
    dataset, queue_labels, id2label, label2id, class_weight_dict = load_dataset()
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    docs, y = train_dataset["text"], train_dataset["labels"]
    
    model = train_supervised_classifier(docs=docs, y=y, class_weight_dict=class_weight_dict)
    def predict(row):
        topic, _ = model.transform(row["text"])
        return topic[0]

    test_dataset = test_dataset.map(lambda x: {
        "preds" : predict(x)
    })

    ## Baseline is 29.11%
    accuracy = test_dataset.filter(lambda x: x["preds"] == x["labels"]).num_rows * 100 / test_dataset.num_rows
    logging.info(f"Accuracy: {accuracy}")



if __name__ == "__main__":
    main()

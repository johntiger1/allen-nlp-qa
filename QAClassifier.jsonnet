{
    "dataset_reader" : {
        "type": "PubMedQADatasetReader-json",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqaa.json",
    "validation_data_path": "/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqal.json",
    "model": {
        "type": "QAClassifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}

from allennlp.data import Vocabulary, DataLoader

from QADatasetReader import PubMedQADatasetReader
from QAModel import QAClassifier
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Auc
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate

import torch
import gc
import tempfile

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
        cuda_device=0
    )
    return trainer

def run_training_loop(model, train_loader, dev_loader, vocab, use_gpu=False, batch_size =32):
    # move the model over, if necessary, and possible
    gpu_device = torch.device("cuda:0" if use_gpu  else "cpu")
    model = model.to(gpu_device)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    gc.collect()

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this course, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
        trainer.train()
    del train_loader
    del dev_loader
    gc.collect()
    return model

def main():

    args = lambda x: None
    args.batch_size = 64
    import time

    start_time = time.time()
    # mr = MortalityReader()
    # instances = mr.read("/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv")
    # for inst in instances[:10]:
    #     print(inst)
    print("we are running with the following info")
    print("Torch version {} Cuda version {} cuda available? {}".format(torch.__version__, torch.version.cuda,
                                                                       torch.cuda.is_available()))
    # We've copied the training loop from an earlier example, with updated model
    # code, above in the Setup section. We run the training loop to get a trained
    # model.

    dataset_reader = PubMedQADatasetReader()


    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data= dataset_reader.read("/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqaa.json")
    dev_data = dataset_reader.read("/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqal.json")
    vocab = Vocabulary.from_instances(train_data + dev_data)

    vocab_size = vocab.get_vocab_size("tokens")
    # turn the tokens into 300 dim embedding. Then, turn the embeddings into encodings
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=300, num_embeddings=vocab_size)})
    encoder = CnnEncoder(embedding_dim=300, ngram_filter_sizes = (2,3,4,5),
                         num_filters=5) # num_filters is a tad bit dangerous: the reason is that we have this many filters for EACH ngram f


    model = QAClassifier(vocab,embedder, encoder)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    model = run_training_loop(model,train_loader, dev_loader, vocab, use_gpu=True, batch_size=args.batch_size)

    # Now we can evaluate the model on a new dataset.
    test_data = dataset_reader.read(
        "/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqal.json")
    test_data.index_with(model.vocab)
    data_loader = DataLoader(test_data, batch_size=args.batch_size)

    # results = evaluate(model, data_loader, -1, None)
    # print(results)

    # will cause an exception due to outdated cuda driver? Not anymore!
    results = evaluate(model, data_loader, 0, None)

    print("we succ fulfilled it")
    with open("nice_srun_time.txt", "w") as file:
        file.write("it is done\n{}\nTook {}".format(results, time.time() - start_time))



if __name__ == "__main__":
    main()
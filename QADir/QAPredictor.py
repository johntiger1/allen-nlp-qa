from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


'''

A predictor necessarily takes inputs in json format and produces predictions over them
'''

@Predictor.register("QA-predictor")
class QAPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"text": sentence}) #wraps it into a json dictionary format

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["text"]
        return self._dataset_reader.text_to_instance(sentence) # turns text into an instance
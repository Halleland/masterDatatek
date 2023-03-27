import sys
sys.path.append('C:\\Users\\Martin\\Documents\\GitHub\\master\\modeling')
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import WECoherenceCentroid, Coherence
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA

import multimodal
import c_tf_idf
import multimodalModel
class Trainer:
    def __init__(self, custom_dataset_folder, params, top_k = 10, metrics_loader=None):
        self.model = None
        self.folder = custom_dataset_folder
        self.top_k = top_k
        self.data = self.get_dataset()
        
        self.metrics_loader = metrics_loader
        self.metrics = self.get_metrics()
        self.params = params
        
        
        
    def train(self, params):
        self.model = multimodalModel.MultimodalModel(**params)
        topics = self.model.fit_transform()
        all_words = [word.lower() for words in self.data.get_corpus() for word in words]
        bertopic_topics = [
            [
                vals.lower() 
                for vals in self.model.get_topic(i)
                if vals.lower() in all_words
            ]
            for i in range(len(set(topics)) - 1)
        ]

        output_tm = {"topics": bertopic_topics}
        return output_tm


    def evaluate(self, model_output, verbose=True):
        results = {}
        for scorers,_ in self.metrics:
            for scorer, name in scorers:
                score = scorer.score(model_output)
                results[name]=float(score)
        if verbose:
            for metric, score in results.items():
                print(f'{metric}:{str(score)}')
        return results
        
    
    def get_dataset(self):
        data = Dataset()
        data.load_custom_dataset_from_folder(self.folder)
        return data

    def get_metrics(self):
        if self.metrics_loader:
            metrics = self.metrics_loader.get_metrics()
        else:
            #npmi = Coherence(texts = self.data.get_corpus(), topk=self.top_k, measure="c_npmi")
            topic_diversity = TopicDiversity(topk=self.top_k)
            wetc = WECoherenceCentroid(topk=self.top_k)

            #coherence= [(npmi, "npmi")]
            coherence2 = [(wetc, 'wetc')]
            diversity = [(topic_diversity, "diversity")]

            metrics = [ (diversity, "Diversity"), (coherence2, "WE Coherence")]#,(coherence, "Coherence")
        return metrics

class LDATrainer(Trainer):
    def __init__(self, num_topics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_topics = num_topics

    def train(self, params):
        self.model = LDA(num_topics=self.num_topics)
        out = self.model.train_model(self.data)
        return out

class MetricsLoader:
    def __init__(self, path_to_datafolder, top_k=10) -> None:
        self.folder = path_to_datafolder
        self.top_k = top_k
        self.data = self.get_dataset()
        self.metrics = self.set_metrics()
        
    def get_dataset(self):
        data = Dataset()
        data.load_custom_dataset_from_folder(self.folder)
        return data
    
    def set_metrics(self):
        #npmi = Coherence(texts = self.data.get_corpus(), topk=self.top_k, measure="c_npmi")
        topic_diversity = TopicDiversity(topk=self.top_k)
        wetc = WECoherenceCentroid(topk=self.top_k)

        #coherence= [(npmi, "npmi")]
        coherence2 = [(wetc, 'wetc')]
        diversity = [(topic_diversity, "diversity")]

        metrics = [(diversity, "Diversity"), (coherence2, "WE Coherence")]#(coherence, "Coherence")
        return metrics

    def get_metrics(self):
        return self.metrics
"""
This class is used to test the init and fit functions of the GPTopic class
"""

import os 
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, f"{parentdir}/src")
from gptopic.GPTopic import GPTopic

sys.path.insert(0, parentdir) 

import openai
import pickle

import unittest

from src.gptopic.TopicRepresentation import Topic

from src.gptopic.Clustering import Clustering_and_DimRed
from src.gptopic.TopwordEnhancement import TopwordEnhancement
from src.gptopic.TopicPrompting import TopicPrompting

class TestGPTopic_init_and_fit(unittest.TestCase):
    """
    Test the init and fit functions of the GPTopic class
    """

    @classmethod
    def setUpClass(cls, sample_size = 0.1):
        """
        load the necessary data and only keep a sample of it 
        """
        print("Setting up class...")
        cls.api_key_openai = os.environ.get('OPENAI_API_KEY')
        #openai.organization = os.environ.get('OPENAI_ORG')

        with open("../Data/Emebeddings/embeddings_20ng_raw.pkl", "rb")  as f:
            data_raw = pickle.load(f)

        corpus = data_raw["corpus"]
        doc_embeddings = data_raw["embeddings"]

        n_docs = int(len(corpus) * sample_size)
        cls.corpus = corpus[:n_docs]
        cls.doc_embeddings = doc_embeddings[:n_docs]

        print("Using {} out of {} documents".format(n_docs, len(data_raw["corpus"])))

        with open("../Data/Emebeddings/embeddings_20ng_vocab.pkl", "rb") as f:
            cls.embeddings_vocab = pickle.load(f)

    def test_init(self):
        """
        test the init function of the GPTopic class
        """
        print("Testing init...")
        gptopic = GPTopic(openai_api_key = self.api_key_openai)
        self.assertTrue(isinstance(gptopic, GPTopic))

        gptopic = GPTopic(openai_api_key = self.api_key_openai, 
                            n_topics= 20)
        self.assertTrue(isinstance(gptopic, GPTopic))
        
        gptopic = GPTopic(openai_api_key = self.api_key_openai, 
                            n_topics= 20,
                            corpus_instruction="This is a corpus instruction", 
                            document_embeddings = self.doc_embeddings,
                            vocab_embeddings= self.embeddings_vocab)
        self.assertTrue(isinstance(gptopic, GPTopic))

        # check if assertions are triggered

        with self.assertRaises(AssertionError):
            gptopic = GPTopic(openai_api_key = None, 
                                n_topics= 32,
                                openai_prompting_model="gpt-4",
                                max_number_of_tokens=8000,
                                corpus_instruction="This is a corpus instruction")

        with self.assertRaises(AssertionError):
            gptopic = GPTopic(openai_api_key = self.api_key_openai, 
                                n_topics= 0,
                                max_number_of_tokens=8000,
                                corpus_instruction="This is a corpus instruction")
            
        with self.assertRaises(AssertionError):
            gptopic = GPTopic(openai_api_key = self.api_key_openai, 
                                n_topics= 20,
                                max_number_of_tokens=0,
                                corpus_instruction="This is a corpus instruction")
            
    def test_fit(self):
        """
        test the fit function of the GPTopic class
        """
        print("Testing fit...")

        def instance_test(gptopic):
            gptopic.fit(self.corpus)

            self.assertTrue(hasattr(gptopic, "vocab"))
            self.assertTrue(hasattr(gptopic, "topic_lis"))

            self.assertTrue(isinstance(gptopic.vocab, list))
            self.assertTrue(isinstance(gptopic.vocab[0], str))

            self.assertTrue(isinstance(gptopic.topic_lis, list))
            try:
                self.assertTrue(type(gptopic.topic_lis[0]) == Topic)
            except AssertionError as e:
                print(e)
                print(type(gptopic.topic_lis[0]))
                print(gptopic.topic_lis[0])

            if gptopic.n_topics is not None:
                self.assertTrue(len(gptopic.topic_lis) == gptopic.n_topics)

            self.assertTrue(gptopic.topic_lis == gptopic.topic_prompting.topic_lis)
            self.assertTrue(gptopic.vocab == gptopic.topic_prompting.vocab)
            self.assertTrue(gptopic.vocab_embeddings == gptopic.topic_prompting.vocab_embeddings)

        
        gptopic1 = GPTopic(openai_api_key = self.api_key_openai, 
                            n_topics= 20,
                            document_embeddings = self.doc_embeddings,
                            vocab_embeddings = self.embeddings_vocab)
    
        gptopic2 = GPTopic(openai_api_key = self.api_key_openai,
                             n_topics= None,
                                document_embeddings = self.doc_embeddings, 
                                vocab_embeddings = self.embeddings_vocab)
        
        gptopic3 = GPTopic(openai_api_key=self.api_key_openai, 
                              n_topics = 1,
                                document_embeddings = self.doc_embeddings,
                                vocab_embeddings = self.embeddings_vocab,
                                n_topwords=10,
                                n_topwords_description=10,
                                topword_extraction_methods=["cosine_similarity"])
        
        clusterer4 = Clustering_and_DimRed(
            n_dims_umap = 10,
            n_neighbors_umap = 20,
            min_cluster_size_hdbscan = 10,
            number_clusters_hdbscan= 10 # use only 10 clusters
        )

        topword_enhancement4 = TopwordEnhancement(openai_key = self.api_key_openai)
        topic_prompting4 = TopicPrompting(
            openai_key = self.api_key_openai,
            enhancer = topword_enhancement4,
            topic_lis = None
        )

        gptopic4 = GPTopic(openai_api_key=self.api_key_openai,
                                n_topics= None,
                                    document_embeddings = self.doc_embeddings, 
                                    vocab_embeddings = self.embeddings_vocab,
                                    topic_prompting = topic_prompting4,
                                    clusterer = clusterer4,
                                    topword_extraction_methods=["tfidf"])
                        

        topic_gpt_list = [gptopic1, gptopic2, gptopic3, gptopic4]

        for topic_gpt in topic_gpt_list:
            instance_test(topic_gpt)
        



if __name__ == "__main__":
    unittest.main()
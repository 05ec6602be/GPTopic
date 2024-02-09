from gptopic.TopicRepresentation import Topic

import unittest
from sklearn.datasets import fetch_20newsgroups 

from gptopic.GPTopic import GPTopic


class QuickTestGPTopic_init_and_fit(unittest.TestCase):
    """
    Run some basic tests on GPTopic that do not require any saved data
    """


    @classmethod
    def setUpClass(cls, sample_size:int = 500):
        """
        download the necessary data and only keep a sample of it 
        params: 
            api_key: the openai api key
            sample_size: the number of documents to use for the test
        """

        data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')) #download the 20 Newsgroups dataset
        corpus = data['data']# just select the first 1000 documents for this example
        corpus = [doc for doc in corpus if doc != ""]
        corpus = corpus[:sample_size]

        cls.corpus = corpus

    def setUp(self):
        self.api_key_openai = api_key


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
                            corpus_instruction="This is a corpus instruction")
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
            self.assertTrue(type(gptopic.topic_lis[0]) == Topic)

            if gptopic.n_topics is not None:
                self.assertTrue(len(gptopic.topic_lis) == gptopic.n_topics)

            self.assertTrue(gptopic.topic_lis == gptopic.topic_prompting.topic_lis)
            self.assertTrue(gptopic.vocab == gptopic.topic_prompting.vocab)
            self.assertTrue(gptopic.vocab_embeddings == gptopic.topic_prompting.vocab_embeddings)

        
        gptopic1 = GPTopic(openai_api_key = self.api_key_openai, n_topics = 1)

        topic_gpt_list = [gptopic1]

        for topic_gpt in topic_gpt_list:
            instance_test(topic_gpt)


import sys

if __name__ == "__main__":
    for i, arg in enumerate(sys.argv):
        if arg == "--api-key":
            api_key = sys.argv.pop(i + 1)
            sys.argv.pop(i)
            break

    if api_key is None:
        print("API key must be provided with --api-key")
        sys.exit(1)
    unittest.main()
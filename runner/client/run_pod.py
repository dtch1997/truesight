from openweights.client.temporary_api import TemporaryApi
from openai import OpenAI

from openweights import OpenWeights
from dotenv import load_dotenv
load_dotenv()

ow = OpenWeights()    

class RunPodClientWrapper:
    def __init__(self, model):
        self.model = model
        self.client = ow
        
    def __enter__(self):
        return ow

    def __exit__(self, exc_type, exc_value, traceback):
        pass


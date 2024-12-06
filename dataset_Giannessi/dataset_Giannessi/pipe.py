from cltk.nlp import GreekPipeline
from maat.maat.converter import Converter
import jsonlines

with open('results.json', 'r') as f:
    reader = jsonlines.Reader(f)
    for obj in reader:
        print(obj['training_text'])

Conv = Converter(); #Convertitore testo 


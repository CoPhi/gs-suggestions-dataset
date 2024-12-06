from cltk.nlp import GreekPipeline
from maat.converter import Converter


"""
    Per eseguire questo script entrare nella dir del package ed eseguire : poetry run python pipe.py

"""
Conv = Converter(); #Convertitore testo

#Per ogni file che si converte bisogna assegnare un nome con un indice che si incrementa
text = "wffe"
print (Conv.convert (text))



# gs-suggestions-dataset

This project aim to provide an automatic suggestion system for supplements for fill the gaps in the Herculaneum Papyri and support the process of creating new critical editions.  
`NLTK` and `CLTK` are used for the ngrams models implementations.  

`Poetry` 2.1.3 is used for managing the projects dependencies. Here there is the [documentation](https://python-poetry.org/docs/). 

## Dataset

The dataset includes:
- [MAAT Corpus](https://zenodo.org/records/12553283)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [PDL-canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)

## Ngrams model Training

Execute the command below to start the ngrams model training:  
```bash
make training
```

It is possibile to configure the hyperparameters in the `config` directory. 

## Ngrams model Evaluation

Execute: 
```bash
make assessment
```

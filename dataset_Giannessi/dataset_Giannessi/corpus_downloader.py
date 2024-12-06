from cltk.data.fetch import FetchCorpus

corpus_downloader = FetchCorpus(language="grc")

for corpus in corpus_downloader._get_user_defined_corpora():
    corpus_downloader.import_corpus(corpus_name=corpus['name'])



from cltk.data.fetch import FetchCorpus

def main (): 
    corpus_downloader = FetchCorpus(language="grc")
    for corpus in corpus_downloader._get_user_defined_corpora():
        corpus_downloader.import_corpus(corpus_name=corpus['name'])

if __name__ == "__main__":
    main()
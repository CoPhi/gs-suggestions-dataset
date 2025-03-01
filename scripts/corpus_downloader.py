from cltk.data.fetch import FetchCorpus

"""
    Questo script usa la classe FetchCorpus per scaricare i corpus definiti nel file YAML presente nella cartella cltk_data
"""

def main():
    downloader = FetchCorpus(language="grc")
    corpora_list = downloader._get_user_defined_corpora()

    for corpus in corpora_list:
        print(f"Importing corpus: {corpus}")
        downloader.import_corpus(corpus_name=corpus['name'])


if __name__ == "__main__":
    main()

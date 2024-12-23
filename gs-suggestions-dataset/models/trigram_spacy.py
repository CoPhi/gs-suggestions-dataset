import spacy

# Preferisci usare la GPU se disponibile
spacy.prefer_gpu()

# Carica il modello di lingua inglese
nlp = spacy.load("el_core_news_sm")

# Esempio di testo
text = "<gap/> \n.......εψήσας δίδ[ου]\n...... [κ]ιρσοὺς· ἐὰν ...\n.......υς τοσούτους μ..\n"

# Processa il testo
doc = nlp(text)

# Tokenizzazione
print("Tokens:")
for token in doc:
    print(token.text)

# Part-of-Speech Tagging
print("\nPart-of-Speech Tags:")
for token in doc:
    print(f"{token.text}: {token.pos_}")

# Named Entity Recognition (NER)
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Parsing delle dipendenze
print("\nDependency Parsing:")
for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")
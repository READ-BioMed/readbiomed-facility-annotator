from nltk import word_tokenize, pos_tag, ne_chunk, Tree, sent_tokenize
import nltk
nltk.download("popular")

def extract_locations_nltk(plainText):
    tokenizedText = nltk.word_tokenize(plainText)
    pos_taggedText = nltk.pos_tag(tokenizedText)
    chunkedText = nltk.ne_chunk(pos_taggedText)

    locations = []
    for subtree in chunkedText:
        if type(subtree) == Tree and subtree.label() == "GPE":
            locations.append(" ".join([token for token, pos in subtree.leaves()]))
    
    return locations

if __name__ == "__main__":
    data = "Determinants of public phobia about infectious diseases in South Korea: effect of health communication and gender difference."
    a = extract_locations_nltk(data)
    print(a)
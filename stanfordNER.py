from stanfordcorenlp import StanfordCoreNLP

### Variable to change
stanfordNERnlp = StanfordCoreNLP(r'/Users/sonya/Downloads/stanford-corenlp-4.3.2', lang="en")
###

def extract_countries_StanfordNER(stanfordNERnlp, plaintext):
    IBO = stanfordNERnlp.ner(plaintext)
    i = 0
    locations = []
    while i < len(IBO):
        word, label = IBO[i]
        if label == "COUNTRY":
            continuous = "".join(word)
            j = i + 1
            while j < len(IBO):
                word1, label1 = IBO[j]
                if label1 == label:
                    continuous = continuous + " " + word1
                    j += 1
                    continue
                else:
                    i = j
                    locations.append(continuous)
                    break
        else:
            i += 1
    return locations

def extract_organization_StanfordNER(plaintext):
    IBO = stanfordNERnlp.ner(plaintext)
    i = 0
    organisations = []
    while i < len(IBO):
        word, label = IBO[i]
        if label == "ORGANIZATION":
            continuous = "".join(word)
            j = i + 1
            while j < len(IBO):
                word1, label1 = IBO[j]
                if label1 == label:
                    continuous = continuous + " " + word1
                    j += 1
                    continue
                else:
                    i = j
                    organisations.append(continuous)
                    break
        else:
            i += 1
    return organisations

if __name__ == "__main__":
    data = "Sequence analysis of hemagglutinin and nucleoprotein genes of measles viruses isolated in South Korea during the 2000 epidemic."

    a = extract_countries_StanfordNER(stanfordNERnlp, data)
    print(a)
from functools import reduce
import re

import spacy
Spacynlp = spacy.load("en_core_web_sm")

def extract_affiliation(affiliation):
    list = affiliation.split(",")
    return list

def extract_locations_Spacy(Spacynlp, plainText):
    locations = []
    entities = Spacynlp(plainText)
    for ent in entities.ents:
        if ent.label_ == "GPE" and delete_Spacy(ent.text, ent.label_):
            locations.append(ent.text)
    return locations

def delete_Spacy(ent, label):
    if ent.islower():
        return False
    if ent.isupper() and label != "GPE":
        return False
    if "influenza" in ent or "Influenza" in ent:
        return False
    if len(ent.split()) > 1:
        for e in ent.split():
            if not e.istitle() and e != "the" and e != "of":
                return False
    if re.findall("H\dN\d", ent)!=[]:
        return False
    return True

def combination(dict1, dict2):
    for i, j in dict2.items():
        if i in dict1.keys():
            dict1[i] += j
        else:
            dict1.update({f"{i}": dict2[i]})
    return dict1

if __name__ == "__main__":
    data = "Determinants of public phobia about infectious diseases in South Korea: effect of health communication and gender difference."
    a = extract_locations_Spacy(Spacynlp, data)
    print(a)

from stanfordcorenlp import StanfordCoreNLP

##############################################
#### VARIABLES TO CHANGE FOR NEW USERS #######
##############################################
stanfordNERnlp = StanfordCoreNLP(r"PLEASE_DEFINE", lang="en")
R_HOME = "PLEASE_DEFINE"


##############################################
################ IMPORTS  ####################
##############################################
import csv
import math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from Bio import Entrez
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from urllib.request import urlopen
from stanfordNER import extract_countries_StanfordNER, extract_organization_StanfordNER
import spacy
import time
import pandas as pd
import json
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
Spacynlp = spacy.load("en_core_web_sm")
from rpy2.robjects.packages import importr
rangeBuilder = importr("rangeBuilder")
import time

# get_geocoder_for_service("nominatim")
import re
from geotext import GeoText

from nltk.tokenize import sent_tokenize

### GEOPY ###
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.geocoders import get_geocoder_for_service
geolocator = Nominatim(user_agent="project")

config = dict(user_agent="project1")
def geocode(geocoder, config, query):
    cls = get_geocoder_for_service(geocoder)
    geolocator = cls(**config)
    location = geolocator.geocode(query, timeout=5, language="en")
    if len(str(location).split(",")) > 1:
        return ""
    return location

########

##############################################
################ Get Full Text ###############
##############################################
"""
E-utility can help to get full text with PMICD.
Full text is stored in "full_text.json", eg:{31411: full text}
Extractions of country and organisation based on StandfordNER are stored in the 
"entity_fullTextCountry_fullTextOrganisation.json", eg:{"31411": [["Sweden"], ["Collection Institut Pasteur"}
"""

def get_fullText_json(PMCID):
    print("Retrieving full text ...")
    # https: // eutils.ncbi.nlm.nih.gov / entrez / eutils / efetch.fcgi?db = pmc & id = 4304705
    data1 = {}
    num = 0
    for i in list(PMCID):
        num += 1
        if math.isnan(i):
            data = {num:[]}
            data1.update(data)
        else:
            efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pmc&id=%s" % (int(i))
            handle = urlopen(efetch)
            data_xml = handle.read()
            text = bytes.decode(data_xml)
            paras = re.findall(r"<p>.*</p>", text)
            pa = []
            for para in paras:
                para = re.sub(r"<p>|</p>", "", para)
                para = re.sub(r"<.*>|\[\]|\(\)", "", para)
                if para:
                    pa += [para]
            para = " ".join(pa)
            para = re.sub(r"[A-Z\.]{2,3}|;]|[\.,]]", "", para)
            data = {int(i): para}
            data1.update(data)
    with open("jsonFiles/fullText/full_text.json", "w") as fw:
        json.dump(data1, fw)
        print("Finished writing full text to jsonFiles/fullText/full_text.json.")


def parse_fullText():
    print("Retrieving countries and organisations from full text ...")
    # https: // eutils.ncbi.nlm.nih.gov / entrez / eutils / efetch.fcgi?db = pmc & id = 4304705
    fullText_sentence_entity = {}
    #         s = {'sup', 'xref', 'italic', 'bold'}
    with open("jsonFiles/fullText/full_text.json", "r") as fr:
        paras = json.load(fr)
        num = 1
        for key, value in paras.items():
            num += 1
            entity_country = []
            entity_organisation = []
            sentences = sent_tokenize(str(value))
            for sentence in sentences:
                sentence = sentence.replace("%", "percent")
                entity_country += extract_countries_StanfordNER(stanfordNERnlp, sentence)
                entity_organisation += extract_organization_StanfordNER(sentence)
            fullText_sentence_entity[key] = [entity_country, entity_organisation]
    with open("jsonFiles/fullText/country_organisation.json", "w") as fw:
        json.dump(fullText_sentence_entity, fw)
        print("Wrote full text countries and organisations into jsonFiles/FullText/country_organisation.json")

''' Get the specified entity (country or organisation) from full text. '''
def get_fullText_entity(entityType):
    print("Retrieving " + entityType + " count from full text ...")
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    with open("jsonFiles/fullText/country_organisation.json", "r") as fr:
        dic_entity = {}
        data = json.load(fr)

        if entityType == "country":
            entity = [v[0] for v in list(data.values())]
        else:
            entity = [v[1] for v in list(data.values())]

        for i in range(len(PMID)):
            dic = {}
            if len(entity[i]) != 0:
                for c in entity[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_entity[str(PMID[i])] = dic
    
    if entityType == "country":
        with open("jsonFiles/fullText/count_country.json", "w") as fw:
            json.dump(dic_entity, fw)
            print("Wrote country count to jsonFiles/fullText/count_country.json")
    else:
        with open("jsonFiles/fullText/count_org.json", "w") as fw:
            json.dump(dic_entity, fw)
            print("Wrote organisation count to jsonFiles/fullText/count_org.json")
        

def get_fullText_organisation_disambiguation():
    print("Removing ambiguous organisations from full text ...")
    with open("jsonFiles/fullText/count_org.json", "r") as fr:
        data = json.load(fr)
        dict1 = {}
        for key, value in data.items():
            dict2 = {}
            if value != {}:
                for k, v in value.items():
                    if re.search("Laboratory|Department|Faculty|Programme|Program|\.|-|&|\(|\)|;|,|/", k):
                        continue
                    if re.search("University", k) and len(k.split()) > 1:
                        dict2[k] = v
                        continue
                    if len(k.split()) <= 2:
                        continue
                    else:
                        dict2[k] = v
            dict1[key] = dict2

    dict3 = {}
    for key, value in dict1.items():
        dict4 = {}
        judge = False
        if value != {}:
            for k, v in value.items():
                try:
                    location = geolocator.geocode(k, language="en")
                    address = location.address
                    if address:
                        judge = True
                except:
                    continue
                if judge:
                    dict4[k] = v
        dict3[key] = dict4

    with open("jsonFiles/fullText/count_org_unambiguous.json", "w") as fw:
        json.dump(dict3, fw)
        print("Wrote unambiguous organisations to jsonFiles/fullText/count_org_unambiguous.json")


###########################################################
## Get title, abstract and affiliation Text and entities ##
###########################################################
"""
 This step is to get the title, abstract and affiliation plaintext based on the PMID and store these message 
 in a json file "jsonFilesdataset.json". There is a csv file "dataset.csv" storing PMID and Data-Location-country 
 annotated by ourselves.
"""

def get_data(data):
    dic = {}
    root = ET.fromstring(data)
    for medlineCitation in root.iter("MedlineCitation"):
        affiliations_list = []
        abstractText_list = []
        PMID = medlineCitation.find("PMID").text
        for article in medlineCitation.iter("Article"):
            articleTitle = article.find("ArticleTitle").text
            for abstract in article.iter("Abstract"):
                for abstractText in abstract.iter("AbstractText"):
                    abstractText_list.append(abstractText.text)
            for affiliationInfo in article.iter("AffiliationInfo"):
                affiliation = affiliationInfo.find("Affiliation").text
                affiliations_list.append(affiliation)
        affiliations = "".join(affiliations_list)
        if abstractText_list != [None]:
            abstractTexts = "".join(abstractText_list)
        else:
            abstractTexts = ""
        dic[PMID] = [articleTitle, abstractTexts, affiliations]
    return dic

def get_title_abstract_affiliation_json(PMID):
    print("Retrieving title, abstract and affiliations ...")
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = get_data(data_xml)
        data1.update(data)
    with open("jsonFiles/dataset.json", "w") as f:
        json.dump(data1, f)
        print("Wrote title, abstract and affiliations to jsonFiles/dataset.json")

def parse_title_abstract_affiliation():
    print("Extracting countries and organisations from titles and abstracts ... ")
    with open("jsonFiles/dataset.json", "r") as fr:
        data = json.load(fr)
        dic = {}
        j = 0
        for i in data.keys():
            j += 1
            abstract_entity_country = []
            abstract_entity_org = []
            title = re.sub("%", "percent", data[i][0])
            title_entity_country = extract_countries_StanfordNER(stanfordNERnlp, title)
            title_entity_org = extract_organization_StanfordNER(title)
            abstract = sent_tokenize(data[i][1])
            for a in abstract:
                a = re.sub("%", "percent", a)
                country = extract_countries_StanfordNER(stanfordNERnlp, a)
                org = extract_organization_StanfordNER(a)
                abstract_entity_country += country
                abstract_entity_org += org
            dic[i] = [[title_entity_country, title_entity_org], [abstract_entity_country, abstract_entity_org]]
    with open("jsonFiles/title_abstract_entities.json", "w") as fw:
        json.dump(dic, fw)
        print("Wrote organisation and country entities to jsonFiles/title_abstract_entities.json")


"""
Using the data in author_affiliation.csv, extracts the country, city and organisations found in the
listed affiliations and writes them in affiliation/country_org.json
"""
def get_affiliation_country_city_organization():
    print("Retrieving countries, cities and organisations from affiliations ...")
    file_name = "author_affiliation.csv"
    file = open(file_name, "r")
    dataframe = pd.read_csv(file)
    affiliations = dataframe.pop("Affiliation")

    i = 2
    num = 0
    affilitation_country_city_org = {}
    for affiliation in affiliations:
        countries = []
        cities = []
        organisations = []
        if str(affiliation) != "None":
            dic_country = {}
            dic_city = {}
            dic_org = {}
            for section in str(affiliation).split("$"):
                result = re.sub(r"[\w,-]+([\.|,][\w,-]+){0,3}@(([\w,-]+\.)|([\w,-]+)){1,3}", "", str(section))
                result = re.sub(r"\;[\s]", ",", result)
                result = re.sub(r"\([\w,-,\.,\,,\s]+\)", "", result)
                result = result.rstrip(".").rstrip(". ").split(",")
                country = GeoText(result[-1]).countries
                if country == []:
                    country = re.sub(r"[0-9,\-,_]+", "", result[-1]).strip()
                    country = re.sub(r"\w{1,2}\s", "", country).strip()
                    country = re.sub(r"\.\s.+", "", country).strip()
                countries.append("".join(country))
                if re.search("^[0-9,\s,A-Z]+$", result[-2]):
                    city = GeoText(result[-3]).cities
                    if city == []:
                        city = re.sub(r"[0-9,\-,_]+", "", result[-3]).strip()
                    o1 = []
                    for o in result[0:-3]:
                        if re.search("[0-9]|\.", o) or len(o.strip().split()) == 1:
                            continue
                        else:
                            o1.append(o)

                    try:
                        organisations.append(o1[-1])
                    except:
                        organisations.append(result[0])

                else:
                    city = GeoText(result[-2]).cities
                    if city == []:
                        city = re.sub(r"[0-9,\-,_]+", "", result[-2]).strip()
                    o1 = []
                    for o in result[0:-2]:
                        if re.search("[0-9]|\.", o) or len(o.strip().split()) == 1:
                            continue
                        else:
                            o1.append(o)
                    try:
                        organisations.append(o1[-1])
                    except:
                        organisations.append(result[0])

                cities.append("".join(city))
            for country in countries:
                if country:
                    dic_country[country.strip()] = dic_country.get(country.strip(), 0) + 1
                else:
                    dic_country[country.strip()] = "None"
            for city in cities:
                if city:
                    dic_city[city.strip()] = dic_city.get(city.strip(), 0) + 1
                else:
                    dic_city[city.strip()] = "None"
            for org in organisations:
                org = re.sub("^(\s|[0-9])+", "", org)
                org = re.sub("^[\s,A-Z,0-9]+$", "", org)
                if org:
                    dic_org[org.strip()] = dic_org.get(org.strip(), 0) + 1
                else:
                    continue

            affilitation_country_city_org[str(PMID[num])] = [dic_country, dic_city, dic_org]
        else:
            affilitation_country_city_org[str(PMID[num])] = [{}, {}, {}]
        i += 1
        num += 1
    print("\n")

    fw = open("jsonFiles/affiliation/country_org.json", "w")
    json.dump(affilitation_country_city_org, fw)
    print("Wrote countries, cities and organisations of affiliations to jsonFiles/affiliation/country_org.json")
    fw.close()

''' Get the specified entity (country or organisation) from the specified section of the source. 
Arguments to provide: 
- entityType (country or organisation)
- section(title, abstract or affiliation),
- file to read from
- file to write to.'''
def get_entity(entityType, section, fileToRead, fileToWrite):
    print("Retrieving " + entityType + " count in " + section + "...")
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")

    with open(fileToRead, "r") as fr:
        data = json.load(fr)

        if entityType == "country" and section == "title":
            entities = [value[0][0] for key, value in data.items()]
        if entityType == "organisation" and section == "title":
            entities = [value[0][1] for key, value in data.items()]
        if entityType == "country" and section == "abstract":
            entities = [value[1][0] for key, value in data.items()]
        if entityType == "organisation" and section == "abstract":
            entities = [value[1][1] for key, value in data.items()]
        if entityType == "country" and section == "affiliation":
            entities = [value[0] for key, value in data.items()]
        if entityType == "organisation" and section == "affiliation":
            entities = [value[2] for key, value in data.items()]
            
        dic_entity = {}
        if section == "affiliation":
            for i in range(len(PMID)):
                if len(entities) != 0:
                    dic_entity[str(PMID[i])] = entities[i]
                else:
                    dic_entity[str(PMID[i])] = {}
        else:
            for i in range(len(PMID)):
                dic = {}
                if len(entities[i]) != 0:
                    for entity in entities[i]:
                        dic[entity] = dic.get(entity, 0) + 1
                dic_entity[str(PMID[i])] = dic

    with open(fileToWrite, "w") as fw:
        json.dump(dic_entity, fw)
        print("Wrote " + entityType + " count of " + section + " to jsonFiles/title/count_country.json")

###########################################################
########## Calculate f1, precision and recall  ############
###########################################################
"""
input: list
output: list
get_standard_countries(["U.S.","America","Republic of China","Korea","South Korea"])
['UNITED STATES', 'America', 'Republic of China', 'Korea', 'SOUTH KOREA']
"""
def get_standard_countries(a):
    b = []
    for i in a:
        i = i.replace("the ", "")
        s = rangeBuilder.standardizeCountry(i)[0]
        if s != "":
            b.append(s)
        else:
            if i == "America":
                b.append("UNITED STATES")
            elif i == "Republic of China":
                b.append("CHINA")
            elif i == "Korea":
                b.append("SOUTH KOREA")
            else:
                b.append(i)
    return b

"""
input: list
output: list
"""

def get_geocode(a):
    b = []
    for i in a:
        geo = []
        for ii in i.split(","):
            geo.append(str(geocode("nominatim", config, ii)))
        geo = ",".join(geo)
        b.append(geo)
    return b

''' Calculate F1 score, precision score and recall score.
Arguments to provide: 
- entityType (country or organisation)
- section (full text, title, abstract or affiliation),
- jsonFile (file to read from)
- improvement: boolean variable determining if user is calculating improvement.
'''
def cal_f1_pre_recall(entityType, section, jsonFile, improvement):
    if improvement:
        print("Calculating imrpovement score for " + entityType + " in the " + section + "...")
    else:
        print("Calculating precision, recall and f1 score for " + entityType + " in the " + section + "...")

    dataset = pd.read_csv("dataset_org.csv")
    if entityType == "country":
        label = dataset.pop("Data-Location-country")
    else:
        label = dataset.pop("Data-Location-org")

    fr = open(jsonFile, "r")
    data = json.load(fr)
    entity = []
    for value in data.values():
        if value != {}:
            c = [k for k,v in value.items() if v==max(value.values())]
            entity.append(c[0])
        else:
            entity.append("")

    if improvement:
        country_std = get_standard_countries(entity)
        label_std = get_standard_countries(label)
        y_pred = np.array(country_std)
        y_true = np.array(list(label_std))
    else:
        y_pred = np.array(entity)
        y_true = np.array(list(label))

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision score:", precision)
    print("recall score:", recall)
    print("f1 score:", f1)
    fr.close()

###########################################################
################ Cal combination weights ##################
###########################################################

def cal_combination_weights_country():
    print("Calculating combination weights for countries ...")
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")

    fr_fullText = open("jsonFiles/fullText/count_country.json", "r")
    fr_title = open("jsonFiles/title/count_country.json", "r")
    fr_abstract = open("jsonFiles/abstract/count_country.json", "r")
    fr_affiliation = open("jsonFiles/affiliation/count_country.json", "r")

    dataset1 = json.load(fr_fullText)
    dataset2 = json.load(fr_title)
    dataset3 = json.load(fr_abstract)
    dataset4 = json.load(fr_affiliation)

    title = list(dataset2.values())
    abstract = list(dataset3.values())
    affiliation = list(dataset4.values())

    file1 = open("country_weights.csv", "w")
    writer1 = csv.writer(file1)
    writer1.writerow(
        ["affiliation weight", "title weight", "abstract weight", "full text weight", "precision", "recall"])
    for afw in range(1, 10):
        for tw in range(1, 10):
            for aw in range(1, 10):
                if tw + aw + afw == 10:
                    country = cal_combination_weights(PMID, title, abstract, affiliation, tw, aw, afw)
                    country_std = get_standard_countries(country)
                    label_std = get_standard_countries(label_country)
                    y_pred = np.array(country_std)
                    y_true = np.array(list(label_std))

                    precision = precision_score(y_true, y_pred, average="weighted")
                    recall = recall_score(y_true, y_pred, average="weighted")
                    F1_score = f1_score(y_true, y_pred, average="weighted")
                    print(precision)
                    print(recall)
                    writer1.writerow([afw / 10.0, tw / 10.0, aw / 10.0, format(precision, ".3f"),
                                      format(recall, ".3f"), format(F1_score, ".3f")])
    
    print("Finished calculating combination weights for countries.")


def cal_combination_weights(PMID, title, abstract, affiliation, t_weight, a_weight, af_weight):
    result = []
    for i in range(len(PMID)):
        dic = {}
        if not title and not abstract and not affiliation:
            result.append("")
            continue
        if title:
            for key, value in title[i].items():
                dic[key] = int(dic.get(key, 0)) + int(value) * t_weight
        if abstract:
            for key1, value1 in abstract[i].items():
                dic[key1] = int(dic.get(key1, 0)) + int(value1) * a_weight
        if affiliation:
            for key2, value2 in affiliation[i].items():
                if key2:
                    dic[key2] = int(dic.get(key2, 0)) + int(value2) * af_weight
        r = [key4 for key4, value4 in dic.items() if value4 == max(dic.values()) and value4 != 0]
        if r:
            result.append(r[0])
        else:
            result.append("")
    return result

def cal_combination_weights_organisation():
    print("Calculating combination weights for organisations ...")
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_org = dataset.pop("Data-Location-org")

    fr_fullText = open("jsonFiles/fullText/count_org.json", "r")
    fr_title = open("jsonFiles/title/count_org.json", "r")
    fr_abstract = open("jsonFiles/abstract/count_org.json", "r")
    fr_affiliation = open("jsonFiles/affiliation/count_org.json", "r")

    dataset1 = json.load(fr_fullText)
    dataset2 = json.load(fr_title)
    dataset3 = json.load(fr_abstract)
    dataset4 = json.load(fr_affiliation)

    title = list(dataset2.values())
    abstract = list(dataset3.values())
    affiliation = list(dataset4.values())
    
    file1 = open("organisation_weights.csv", "w")
    writer1 = csv.writer(file1)
    writer1.writerow(
        ["title weight", "abstract weight", "affiliation weight", "precision", "recall", "F1-score"])
    for tw in range(1, 10):
        for aw in range(1, 10):
            for afw in range(1, 10):
                if tw + aw + afw == 10:
                    organisation = cal_combination_weights(PMID, title, abstract, affiliation, tw, aw,
                                                           afw)

                    organisation_std = get_standard_countries(organisation)
                    label_std = get_standard_countries(label_org)
                    y_pred = np.array(organisation_std)
                    y_true = np.array(list(label_std))

                    precision = precision_score(y_true, y_pred, average="weighted")
                    recall = recall_score(y_true, y_pred, average="weighted")
                    F1_score = f1_score(y_true, y_pred, average="weighted")
                    print(precision)
                    print(recall)
                    writer1.writerow([tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".3f"),
                                      format(recall, ".3f"), format(F1_score, ".3f")])
    print("Finished calculating combination weights for organisations.")


##############################################
######### Calculate country level  ###########
##############################################

def improvement1(li):
    result = []
    for l in li:
        if l == [] or l.split(",")[0] == "":
            result.append("")
            continue
        if len(l.split(",")) == 2:
            l1, l2 = l.split(",")
        if len(l.split(",")) >= 4:
            l1, l2 = l.split(",")[0], l.split(",")[1]

        if l1.lower() == "republic of china":
            result.append("china"+","+l2)
        elif l1.lower() == "korea":
            result.append("south korea"+","+l2)
        elif l1.lower() == "america":
            result.append("united states"+","+l2)
        else:
            result.append(l1.lower()+","+l2)
    return result

def improvement2(li):
    result = []
    for l in li:
        if l == []:
            result.append("")
            continue
        if l.lower() == "republic of china":
            result.append("china")
        elif l.lower() == "korea":
            result.append("south korea")
        elif l.lower() == "america" or l.lower() == "u.s.":
            result.append("united states")
        else:
            result.append(l.lower())
    return result

def cal_country_level(label):
    print("Calculating country level ...")
    with open("jsonFiles/title_abstract_country_standard.json", "r") as fr:
        data = json.load(fr)
        title_country = improvement1([value[0] for key, value in data.items()])
        abstract_country = improvement1([value[1] for key, value in data.items()])

    geo_label = []
    label_country = get_standard(label)
    for s in label_country:
        geo = get_geocode(s) if not s.isupper() and not s else s
        geo_label.append(geo)
    geo_label = improvement2(geo_label)

    with open("jsonFiles/affiliation/country_city_org.json", "r") as fr1:
        data1 = json.load(fr1)
        affiliation_country = []
        for key, value in data1.items():
            a = [k+","+str(v) for k, v in value[0].items() if value[0][k] == max(value[0].values())]
            affiliation_country.append(",".join(a))
    affiliation_country = improvement1(affiliation_country)

    with open("combination_weights.csv", "w") as fw:
        csv_write = csv.writer(fw)
        csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision", "recall"])
        for tw in range(1, 10):
            for aw in range(1, 10):
                for afw in range(1, 10):
                    if tw + aw + afw == 10:
                        precision, recall = different_weights_combination(geo_label, title_country, abstract_country,
                                                                          affiliation_country, tw, aw, afw)
                        csv_write.writerow(
                            [tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".4f"), format(recall, ".4f")])
    print("Finished calculating country level.")

def different_weights_combination(label_country, title_country, abstract_country, affiliation_country, title_weight,
                                  abstract_weight, affiliation_weight):
    countries = []
    for i in range(len(label_country)):
        dic = {}
        if title_country[i] != "" and title_country[i]:
            a1, a2 = title_country[i].split(",")
            dic[a1] = dic.get(a1, 0) + title_weight * int(a2)
        if abstract_country[i] != "":
            a3, a4 = abstract_country[i].split(",")
            dic[a3] = dic.get(a3, 0) + abstract_weight * int(a4)
        if affiliation_country[i] != "":
            a5, a6 = affiliation_country[i].split(",")
            dic[a5] = dic.get(a5, 0) + affiliation_weight * int(a6)
        c = ",".join([key for key, value in dic.items() if value == max(dic.values()) and value != 0])
        if title_country[i] == "" and abstract_country[i] == "" and affiliation_country[i] == "":
            c = ""
        countries.append(c)

    y_true = np.array(label_country)
    y_pred = np.array(countries)

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    print("precision:", precision)
    print("recall:", recall)
    return precision, recall

##############################################
########### Helper Functions  ################
##############################################

def parse_data_author(data):
    judge_same = True
    dic = {}
    root = ET.fromstring(data)
    for medlineCitation in root.iter("MedlineCitation"):
        affiliations_list = []
        author_affiliation = []
        author_list = []
        PMID = medlineCitation.find("PMID").text
        print(PMID)
        for article in medlineCitation.iter("Article"):
            for AuthorList in article.iter("AuthorList"):
                for author in AuthorList.iter("Author"):
                    last_name = author.find("LastName").text
                    fore_name = author.find("ForeName").text
                    full_name = last_name + " " + fore_name
                    author_list.append(full_name)
                    if author.iter("AffiliationInfo") != None:
                        for affiliationInfo in author.iter("AffiliationInfo"):
                            affiliation = affiliationInfo.find("Affiliation").text
                            affiliations_list.append(affiliation)
                            author_affiliation.append([full_name, affiliation])
            affiliations = "$".join(affiliations_list)
            if affiliations == "":
                affiliations = "None"
            authors = ",".join(author_list)
            if len(author_list) != len(affiliations_list):
                judge_same = False
            dic[PMID] = [authors, affiliations, len(author_list), len(affiliations_list), judge_same]
    return dic

def get_csv(PMID):
    print("Getting csv ...")
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = parse_data_author(data_xml)
        data1.update(data)

    fw = open("author_affiliation.csv", "w", newline="")
    csv_write = csv.writer(fw)
    csv_write.writerow(["PMID", "Author", "Affiliation", "length(Author)", "length(Affiliation)", "equal"])
    for i in list(data1.keys()):
        csv_write.writerow([i, data1[i][0], data1[i][1], data1[i][2], data1[i][3], data1[i][4]])
    print("Finished getting csv.")


def get_standard(a):
    b = []
    for i in a:
        ss = []
        for a in i.split(","):
            a = a.replace("the ","")
            s = rangeBuilder.standardizeCountry(a)[0]
            ss.append(s)
        b.append(",".join(ss))
    return b

##############################################
##############  main function  ###############
##############################################

if __name__ == "__main__":
    startime = time.time()

    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    #### get text and entities
    #get_title_abstract_affiliation_json(PMID)
    #parse_title_abstract_affiliation()
    #get_affiliation_country_city_organization()
    #get_fullText_json(PMCID)
    #parse_fullText()

    #get_fullText_entity("country")
    #get_entity("country", "title", "jsonFiles/title_abstract_entities.json", "jsonFiles/title/count_country.json")
    #get_entity("country", "abstract", "jsonFiles/title_abstract_entities.json", "jsonFiles/abstract/count_country.json")
    #get_entity("country", "affiliation", "jsonFiles/affiliation/country_org.json", "jsonFiles/affiliation/count_country.json")

    #get_fullText_entity("organisation")
    #get_fullText_organisation_disambiguation()
    #get_entity("organisation", "title", "jsonFiles/title_abstract_entities.json", "jsonFiles/title/count_org.json")
    #get_entity("organisation", "abstract", "jsonFiles/title_abstract_entities.json", "jsonFiles/abstract/count_org.json")
    #get_entity("organisation", "affiliation", "jsonFiles/affiliation/country_org.json", "jsonFiles/affiliation/count_org.json")

    #### calculate country baseline
    # cal_f1_pre_recall("country", "full text", "jsonFiles/fullText/count_country.json", False)
    # cal_f1_pre_recall("country", "title", "jsonFiles/title/count_country.json", False)
    # cal_f1_pre_recall("country", "abstract", "jsonFiles/abstract/count_country.json", False)
    # cal_f1_pre_recall("country", "affiliation", "jsonFiles/affiliation/count_country.json", False)

    #### calculate organisation baseline
    # cal_f1_pre_recall("organisation", "full text", "jsonFiles/fullText/count_org_unambiguous.json", False)
    # cal_f1_pre_recall("organisation", "title", "jsonFiles/title/count_org.json", False)
    # cal_f1_pre_recall("organisation", "abstract", "jsonFiles/abstract/count_org.json", False)
    # cal_f1_pre_recall("organisation", "affiliation", "jsonFiles/affiliation/count_org.json", False)

    #### calculate country improvement (last argument changed to True)
    # cal_f1_pre_recall("country", "full text", "jsonFiles/fullText/count_country.json", True)
    # cal_f1_pre_recall("country", "title", "jsonFiles/title/count_country.json", True)
    # cal_f1_pre_recall("country", "abstract", "jsonFiles/abstract/count_country.json", True)
    # cal_f1_pre_recall("country", "affiliation", "jsonFiles/affiliation/count_country.json", True)
    
    #### calculate combination weights
    #cal_combination_weights_country()
    #cal_combination_weights_organisation()

    #cal_country_level(label_country)

    ####
    # get_csv(PMID)

    print("Program finished in " + str(time.time()-startime) + " seconds")
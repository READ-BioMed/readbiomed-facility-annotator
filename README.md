# BIOMED Facility Annotator
#### Discover the possible origin location of a medical source, through extracting both the countries and organisations found in a source's title, abstract, affiliation and full text. NLTK and stanford machine learning libraries are used to assist in this goal.

## PROJECT CONTENTS
- BioPython.py: main project file.
- nltkNER.py, spacyNER.py and stanford.py: extra python files showcasing three methods of extracting organisations and locations. standfordNER.py is the only one used in the main project file.
- json Files: contains json files used in main project
    - folders withing jsonFiles contain the sections of text in which json files relating to that section can be found --> title, abstract, affiliation, full text
- author_affiliation.csv and dataset_org.csv: datasets
- requirements.txt: python library requirements.

## NEW USERS - How to begin
1. Opening the project, all the necessary library imports will cause errors if not previously installed. To install all the requirements run the command `python -m pip install -r requirements.txt`.
    1. In addition the command  `python -m spacy download en_core_web_sm` must be ran.

2. Visit https://stanfordnlp.github.io/CoreNLP/download.html to download stanfordCoreNLP for your device. 
    MAC USERS ONLY: For the standfordCoreNLP packages, you must run the project from root user to import, due to security changes in the recent updates. Simply, run the project from command line as usual but prefixing with `sudo`.

3. Additional rpy2 packages found in BioPython.py will need R studio software to be downloaded. This can be done with the following steps:
    1. Download R from https://cran.r-project.org/.
    2. Download R studio from https://www.rstudio.com/products/rstudio/download/#download.
    3. In R studio, click the "Install" button, and in the white space provided write the name of a required package.
        The required packages are:
        - rangeBuilder
        - terra
        - rgdal
    WINDOW USERS ONLY: If rpy2 fails to run after the above steps, alternatively, the guide from the source https://jianghaochu.github.io/how-to-install-rpy2-in-windows-10.html needs to be followed to successfully download rpy2.
        - Both the variables R_HOME and R_USER can be found in BioPython.py

4. In both BioPython.py and stanfordNER.py there is a commented subheading called "variables to change". Under this subheading new users must change the variables to match certain locations of files in their device. Specifically, this includes the location of standford coreNLP download and R library download. 

## HOW TO USE PROJECT
At the bottom of BioPython.py, in the main function are commented methods which can be uncommented to run. To run the project, type `python BioPython.py` or `sudo python BioPython.py` for MAC OS users in command line. 
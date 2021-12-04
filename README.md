# BIOMED Facility Annotator
#### Discover the possible origin location of a medical source, through extracting both the countries and organisations found in a source's title, abstract, affiliation and full text. NLTK and stanford machine learning libraries are used to assist in this goal.

## PROJECT CONTENTS
- BioPython.py: main project file.
- nltkNER.py, spacyNER.py and stanford.py: extra python files showcasing three methods of extracting organisations and locations. standfordNER.py is the only one used in the main project file.
- json Files: contains json files used in main project
    - folders withing jsonFiles contain the sections of text in which json files relating to that section can be found --> title, abstract, affiliation, full text
- author_affiliation.csv and dataset_org.csv: datasets
- country_weights.csv, organisation_weights.csv and combination_weights.csv: results of methods cal_combination_weights_country(), cal_combination_weights_organisation() and cal_country_level(label) from BioPython.py respectively.

## NEW USERS - How to begin
1. Opening the project, all the necessary library imports will cause errors if not previously installed. As a new user, make sure to go through each of the python files and download the required libraries.
    1. rpy2 packages found in BioPython.py will need R studio software to be downloaded. In R studio, the packages rangeBuilder and terra must be downloaded for the project to run.

2. MAC USERS ONLY: For the standfordCoreNLP packages, you must run the project from root user to import, due to security changes in the recent updates. Simply, run the project from command line as usual but prefixing with "sudo ".

3. In both BioPython.py and stanfordNER.py there is a commented subheading called "variables to change". Under this subheading new users must change the variables to match certain locations of files in their device. Specifically, this includes the location of standford coreNLP download and R library download. 

## HOW TO USE PROJECT
At the bottom of BioPython.py, in the main function are commented methods which can be uncommented to run. To run the project, type "python BioPython.py" or "sudo python BioPython.py" for MAC OS users in command line. 
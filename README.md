# FakeNewsDetection
Project is to create a Fake news classifier by providing it URLs for which:
-- there is at least one tweet, and
-- where the title and content of the URL is available.

We have extracted such a dataset from a Github repo (github.com/several27/FakeNewsCorpus). There were around 2.0 million "Reliable" and 0.9 million "Fake" URLs. We selected 5,500 data points from these two categories to create a dataset of 11,000 URLs for which feature extraction was done.

Features belong to 4 main groups:
Morphological, Psychological (aka Linguistic), Twitter and Readability.

################################################################################################
DATA files uploaded:
1) All_Features_No_Norm.csv -- contains all features
2) M_Features_No_Norm.csv   -- contains only the Morphological features
3) L_Features_No_Norm.csv   -- contains only the Psychological features
4) T_Features_No_Norm.csv   -- contains only the Twitter features
5) R_Features_No_Norm.csv   -- contains only the Readability features
6) MLR_Features_No_Norm.csv -- contains all features EXCEPT Twitter features


################################################################################################
CODE files are uploaded:

1) scriptExtractTweets1.py
Takes an input file for the URLs and extracts all tweets. Uses command line parameters.

2) scriptCreateReadabilityFeatures1.py
Uses a file with URL + Title + Content and processes and creates the Readability features.

3) demoRunFakeNews1.ipynb
Jupyter Notebook file in which one starts with the input file containing the URL, Title and Content. Then executes all the steps to extract tweets, process the data to create the features files for each group. Then compbines the individual feature files into one file containing all the features.
Using the All features file as input now, further subsets of the data are created:
-- using only the non-normalized features
-- using different com

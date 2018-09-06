# Written by Saurabh Kaul
# saurabhkaul.com

#Imports
import pandas as pd
import requests
import re
from html.parser import HTMLParser
import nltk
import string



# Important Variables

#Used to supply a header agent to SEC website

headers={'User-Agent':"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"}

# Stores half the URL and half is fetched from cik_list
url_concat="https://www.sec.gov/Archives/"

# Storing the three Section names as raw string literals


mda=r"MANAGEMENT'S DISCUSSION AND ANALYSIS"

qqdmr=r"QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"


rf=r"RISK FACTORS"


# Helper Functions

# Class with function strip_tags that helps remove all html elements from a string
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()





#This function displays a dataframe in a neat way

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

# Reading Dict's and Lists

cik_list=pd.read_excel('Data Extraction _ Text Analysis/Assignment/cik_list.xlsx')

constrain_dict=pd.read_excel('Data Extraction _ Text Analysis/Assignment/constraining_dictionary.xlsx').values.flatten()

uncertainty_dict=pd.read_excel('Data Extraction _ Text Analysis/Assignment/uncertainty_dictionary.xlsx').values.flatten()

stop_words= pd.read_fwf('Data Extraction _ Text Analysis/Assignment/StopWords_Generic.txt').values.flatten()

master_dict=pd.read_csv('Data Extraction _ Text Analysis/Assignment/LoughranMcDonald_MasterDictionary_2016.csv')

#Creates a list of complex words extracted from the master dict
complex_words=list()

for i in master_dict.index:
    val= master_dict.get_value(i,'Syllables')
    if val>2:
        complex_words.append(master_dict.get_value(i,"Word"))




# Creates positive words and negative words dict from text file
positive_words=open('Data Extraction _ Text Analysis/Assignment/LM_Positive.txt')
positive_words=nltk.word_tokenize(positive_words.read())

negative_words=open('Data Extraction _ Text Analysis/Assignment/LM_Negative.txt')
negative_words=nltk.word_tokenize(negative_words.read())














# Main Class


#The engine that drives this code. Every instance of object Data will store all the variables calculated and scrapped from each report.
#For any data that is missing/could not be calculated it stores ""
class Data:
    'Singular class with all metrics defined that creates an obj holding data of single report'
    def __init__(self,url):
        # Get the url from argument
        self.url=url
        #Stores the report as a string

        self.report=self.get_report(url)
        #Store text under the 3 headings in 3 seperate variables
        self.mda=self.get_mda(self.report)
        self.qqdmr=self.get_qqdmr(self.report)
        self.rf=self.get_rf(self.report)

        #Clean the text under each heading using stop words list, and tokenize it.Store the list of tokens

        self.clean_mda=self.clean_using_swl(self.mda)
        self.clean_qqdmr = self.clean_using_swl(self.qqdmr)
        self.clean_rf = self.clean_using_swl(self.rf)

        # Calculate the positive words score for all three headings
        self.mda_positive_score = self.positive_score(self.clean_mda)
        self.qqdmr_positive_score=self.positive_score(self.clean_qqdmr)
        self.rf_positive_score=self.positive_score(self.clean_rf)
        # Calculate the negative words score for all three headings

        self.mda_negative_score = self.negative_score(self.clean_mda)
        self.qqdmr_negative_score= self.negative_score(self.clean_qqdmr)
        self.rf_negative_score=self.negative_score(self.clean_rf)

        # Calculate and store the polarity score by supplying the values calculated above
        self.mda_polarity_score=self.polarity_score(self.mda_positive_score,self.mda_negative_score)
        self.qqdmr_polarity_score = self.polarity_score(self.qqdmr_positive_score, self.qqdmr_negative_score)
        self.rf_polarity_score = self.polarity_score(self.rf_positive_score, self.rf_negative_score)

        # Store the word count for all three headings
        self.mda_word_count=self.word_count(self.clean_mda)
        self.qqdmr_word_count=self.word_count(self.clean_qqdmr)
        self.rf_word_count=self.word_count(self.clean_rf)

        # Store the sentence count for all three headings
        self.mda_sent=self.sentences(self.clean_mda)
        self.qqdmr_sent = self.sentences(self.clean_qqdmr)
        self.rf_sent = self.sentences(self.clean_rf)

        # Calculate and store Average Sentence Length for all three headings
        self.mda_avg_sent_length= self.avg_sent_len(self.mda_word_count,len(self.mda_sent))
        self.qqdmr_avg_sent_length = self.avg_sent_len(self.qqdmr_word_count, len(self.qqdmr_sent))
        self.rf_avg_sent_length = self.avg_sent_len(self.rf_word_count, len(self.rf_sent))

        # Calculate and store Complex Word Count for all three headings
        self.mda_complex_word_count=self.complex_word_count(self.clean_mda)
        self.qqdmr_complex_word_count = self.complex_word_count(self.clean_qqdmr)
        self.rf_complex_word_count = self.complex_word_count(self.clean_rf)

        # Calculate and store % of complex words
        self.mda_percent_complex=self.percent_complex_words(self.mda_complex_word_count,self.mda_word_count)
        self.qqdmr_percent_complex = self.percent_complex_words(self.qqdmr_complex_word_count, self.qqdmr_word_count)
        self.rf_percent_complex = self.percent_complex_words(self.rf_complex_word_count, self.rf_word_count)

        # Calculate and store fog index
        self.mda_fog_index=self.fog_index(self.mda_avg_sent_length,self.mda_percent_complex)
        self.qqdmr_fog_index = self.fog_index(self.qqdmr_avg_sent_length, self.qqdmr_percent_complex)
        self.rf_fog_index = self.fog_index(self.rf_avg_sent_length, self.rf_percent_complex)

        # Calculate and store Uncertanity word count
        self.mda_uncertain_count=self.dict_count(self.clean_mda,uncertainty_dict)
        self.qqdmr_uncertain_count = self.dict_count(self.clean_qqdmr, uncertainty_dict)
        self.rf_uncertain_count = self.dict_count(self.clean_rf, uncertainty_dict)

        # Calculate and store Constrain word count
        self.mda_constrain_count=self.dict_count(self.clean_mda,constrain_dict)
        self.qqdmr_constrain_count = self.dict_count(self.clean_qqdmr, constrain_dict)
        self.rf_constrain_count = self.dict_count(self.clean_rf, constrain_dict)

        # Calculate and store Word propotion variables positive,negative,constrain and uncertain for all three headings
        self.mda_positive_word_proportion=self.propotion(self.mda_positive_score,self.mda_word_count)
        self.mda_negative_word_proportion = self.propotion(self.mda_negative_score, self.mda_word_count)
        self.mda_uncertainity_word_proportion = self.propotion(self.mda_uncertain_count, self.mda_word_count)
        self.mda_constrain_word_propotion=self.propotion(self.mda_constrain_count,self.mda_word_count)

        self.qqdmr_positive_word_proportion = self.propotion(self.qqdmr_positive_score, self.qqdmr_word_count)
        self.qqdmr_negative_word_proportion = self.propotion(self.qqdmr_negative_score, self.qqdmr_word_count)
        self.qqdmr_uncertainity_word_proportion = self.propotion(self.qqdmr_uncertain_count, self.qqdmr_word_count)
        self.qqdmr_constrain_word_propotion = self.propotion(self.qqdmr_constrain_count, self.qqdmr_word_count)

        self.rf_positive_word_proportion = self.propotion(self.rf_positive_score, self.rf_word_count)
        self.rf_negative_word_proportion = self.propotion(self.rf_negative_score, self.rf_word_count)
        self.rf_uncertainity_word_proportion = self.propotion(self.rf_uncertain_count, self.rf_word_count)
        self.rf_constrain_word_propotion = self.propotion(self.rf_constrain_count, self.rf_word_count)

        # Calculate count of constraining words for whole report
        self.constrain_whole_report= self.dict_whole_report(self.report,constrain_dict)





    # Class methods
    # def __repr__(self):
    #     return self.get_mda(self.get_report)


    # Gets the report using requests module with headers supplied
    def get_report(self,url):
        return requests.get(url,headers=headers).text

    # Function to extract text under the mda heading
    def get_mda(self,report):
        # Check if report is something other than string or MDA heading exists or not
        if(isinstance(report,str)==False or re.search(mda,report)==None):
            return ""
        else:
            # Get an iterable of all instances of MDA in the report
            mda_iter_list=re.finditer(mda,report)

           # Gives us the last instance of MDA from the iterable
            *_,  mda_text_start=mda_iter_list

            # Tuple that stores the starting and ending index of the last instance
            y=mda_text_start.span()
            # We slice the report from the start index of the last instance of MDA to the end of the report
            mda_report_start=report[y[1]:]
            # We now need to look for the index where text under MDA stops and new heading starts. We do this by looking for the next "ITEM" or "PAGE"
            if(re.search(r"ITEM",mda_report_start)!=None):
                # If ITEM is found return its starting index
                closing_index=re.search(r"ITEM",mda_report_start).start()
            elif(re.search(r"PART",mda_report_start)):
                # If ITEM not found check for PART and return its starting index
                closing_index=re.search(r"PART",mda_report_start).start()
            else:
                # If both of them not found return None as MDA could be the last ITEM of the report
                closing_index=None
            # Slice to get the final text under the heading
            final_report= mda_report_start[:closing_index]

            # Return the final report with all HTML removed
            return strip_tags(final_report)


    # Similar to get_mda. Gets us text under the QQDMR heading.
    def get_qqdmr(self,report):
        if(isinstance(report,str)==False or re.search(qqdmr,report)==None):
            return ""

        else:
            qqdmr_iter_list=re.finditer(qqdmr,report)
           # print(qqdmr_iter_list)
            *_,  qqdmr_text_start=qqdmr_iter_list
            # print(qqdmr_text_start)
            y=qqdmr_text_start.span()

            qqdmr_report_start=report[y[1]:]
            if(re.search(r"ITEM",qqdmr_report_start)!=None):
                closing_index=re.search(r"ITEM",qqdmr_report_start).start()
            elif(re.search(r"PART",qqdmr_report_start)):
                closing_index=re.search(r"PART",qqdmr_report_start).start()
            else:
                closing_index=None

            final_report= qqdmr_report_start[:closing_index]
            # print(final_report)

            return strip_tags(final_report)

    # Similar to get_mda. Gets us text under the RF heading.
    def get_rf(self,report):
        if(isinstance(report,str)==False or re.search(rf,report)==None):
            return ""
        else:
            rf_iter_list=re.finditer(rf,report)
           # print(mda_iter_list)
            *_,  rf_text_start=rf_iter_list
            # print(mda_text_start)
            y=rf_text_start.span()

            rf_report_start=report[y[1]:]
            if(re.search(r"ITEM",rf_report_start)!=None):
                closing_index=re.search("ITEM",rf_report_start).start()
            # elif(re.search("PART",rf_report_start)):
            #     closing_index=re.search("PART",rf_report_start).start()
            else:
                closing_index=None

            final_report= rf_report_start[:closing_index]
            # print(final_report)

            return strip_tags(final_report)

    # Tokenizes the report then removes words that are present in the Stop Words list
    def clean_using_swl(self,report):
        if(report==""):
            return ""
        else:
           token_report= nltk.word_tokenize(report)
           for word in token_report:
                if (word.upper() in stop_words):
                    token_report.remove(word)

        return token_report
    # Calculates the positive score of a tokenized report
    def positive_score(selfs,token_report):
        if (token_report == ""):
            return ""
        else:
            counter=0
            for word in token_report:
                if(word.upper() in positive_words):
                    counter+=1
            return counter

    # Calculates the negative score of a tokenized report
    def negative_score(selfs,token_report):
        if (token_report == ""):
            return ""
        else:
            counter=0
            for word in token_report:
                if(word.upper() in negative_words):
                    counter+=1
            return counter
    # Calculates the polarity score from the Positive score and Negative score
    def polarity_score(self,positive_score,negative_score):
        if(positive_score=="" and negative_score==""):
            return ""
        else:
            polarity_score=(positive_score-negative_score)/((positive_score+negative_score)+0.000001)
            return polarity_score
    # Takes a tokenized report, joins it into a string then again tokenizes it to get a list of sentences
    def sentences(self,token_report):
        if(token_report==""):
            return ""
        else:
            # Create a string from list of tokens(Orignal Report)
            token_report=" ".join(token_report)
            # Tokenize it to get the Sentence tokens
            sentence_tokens=nltk.sent_tokenize(token_report)
            return sentence_tokens

    # Returns count of words in a tokenized report
    def word_count(self,token_report):
        if (token_report == ""):
            return ""
        else:
            #Remove all punctuation
            for ix in token_report:
                if(ix in string.punctuation):
                    token_report.remove(ix)

            return len(token_report)

    # Returns the avg sentence length
    def avg_sent_len(self,word_count,sent_count):
        if(word_count=="" or sent_count==""):
            return ""
        else:
            return word_count/sent_count
    # Returns the count of complex words
    def complex_word_count(self,token_report):
        if (token_report == ""):
            return ""
        else:
            counter=0
            for word in token_report:
                if word.upper() in complex_words:
                    counter+=1

            return counter
    # Returns % of complex words
    def percent_complex_words(self,complex_word_count,word_count):
        if (complex_word_count=="" or word_count==""):
            return ""
        else:
            percentage=complex_word_count/word_count
            return percentage

    # Returns fog index of a reprot
    def fog_index(self,avg_sent_len,percent_complex_words):
        if(avg_sent_len=="" or percent_complex_words==""):
            return ""

        else:
            fog_index=0.4*(avg_sent_len+percent_complex_words)
            return fog_index

    # Takes a token report and a dict (which is a list). Counts number of words in report that are in the dict.
    def dict_count(self,token_report,dict):
        if(token_report==""):
            return ""
        else:
            counter=0
            for word in token_report:
                if word.upper() in dict:
                    counter+=1

        return counter

    # Takes a number(score) of a certain metric and returns the propotion of that metric in terms of the total word count that metric was applied to.
    def propotion(self, score_count,word_count):
        if(score_count=="" or word_count==""):
            return ""
        else:
            return score_count/word_count

    # Takes a dict and a full unabridged report we got from get_report. Returns count of words that are in the report that belong to the dict
    def dict_whole_report(self,report,dict):
        # Check if report exists or not
        if(isinstance(report,str)==False or report==None):
            return ""
        else:
            # The SEC filings can have HTML, XBRL and we only need the text that is in the HTML part
            # Therefore we extract all the html using regex
            counter=0
            # All html will be under these two tags
            start=r"<html>\n"
            stop=r"</html>\n"
            # Since not all reports will have HTML we check for html
            if (re.search(start,report)==None or re.search(stop,report)==None):
                # Since no html was found we simply tokenize the report and start counting
                report = nltk.word_tokenize(report)
                for word in report:
                    if (word.upper() in dict):
                        counter += 1
                return counter




            else:
                # We found html and now we need to slice in such a way that we all get the html only
                # HTML can be found a lot of times in the report
                # Creates a list out of the iterable having all the instances of both the html tags
                html_starts = list(re.finditer(start, report))
                html_stops = list(re.finditer(stop, report))
                # An empty string that will hold the extracted html for us
                z=""

                # We iterate over both the list simultaneously
                for x,y in zip(html_starts, html_stops):
                    # We first slice from the last index of <html> and first index of </html>
                    # We keep concating the sliced string
                    z=z+" "+report[x.end():y.start()]
                # we now get concated html from the report that holds the text
                # We now strip it of all html and tokenize it
                report_ext=strip_tags(z)
                report_ext=nltk.word_tokenize(report_ext)


            # Now we count
            for word in report_ext:
                if(word.upper() in dict):
                    counter+=1
            return counter

# --------------------------end of class--------------------------------------------------------------------------------


# Create a list
x=list()

# From the col SECFNAME we create Data objects for each row and append it to a list
for value in cik_list.SECFNAME:
     x.append(Data(url_concat+value))

# List x now holds all the Data objs containing all the metrics for all the reports

# We now create a dataframe that will store all the variables we need to output a file

# First we copy the existing df cik_list

output_df=cik_list.copy()

# Create a list of col names
col_list=[
"mda_negative_score",
"mda_polarity_score",
"mda_average_sentence_length",
"mda_percentage_of_complex_words",
"mda_fog_index",
"mda_complex_word_count",
"mda_word_count",
"mda_uncertainty_score",
"mda_constraining_score",
"mda_positive_word_proportion",
"mda_negative_word_proportion",
"mda_uncertainty_word_proportion",
"mda_constraining_word_proportion",
"qqdmr_positive_score",
"qqdmr_negative_score",
"qqdmr_polarity_score",
"qqdmr_average_sentence_length",
"qqdmr_percentage_of_complex_words",
"qqdmr_fog_index",
"qqdmr_complex_word_count",
"qqdmr_word_count",
"qqdmr_uncertainty_score",
"qqdmr_constraining_score",
"qqdmr_positive_word_proportion",
"qqdmr_negative_word_proportion",
"qqdmr_uncertainty_word_proportion",
"qqdmr_constraining_word_proportion",
"rf_positive_score",
"rf_negative_score",
"rf_polarity_score",
"rf_average_sentence_length",
"rf_percentage_of_complex_words",
"rf_fog_index",
"rf_complex_word_count",
"rf_word_count",
"rf_uncertainty_score",
"rf_constraining_score",
"rf_positive_word_proportion",
"rf_negative_word_proportion",
"rf_uncertainty_word_proportion",
"rf_constraining_word_proportion",
"constraining_words_whole_report"
]


# Add those columns to our output df
output_df.reindex(columns = col_list)

# iterate over x (list of Data objs) and insert values from the objs in the output dataframe

for row_index,Data in enumerate(x):
    output_df.loc[row_index,"mda_negative_score"]=Data.mda_negative_score
    output_df.loc[row_index, "mda_polarity_score"] = Data.mda_polarity_score
    output_df.loc[row_index,"mda_average_sentence_length"]=Data.mda_avg_sent_length
    output_df.loc[row_index,"mda_percentage_of_complex_words"]=Data.mda_percent_complex
    output_df.loc[row_index,"mda_fog_index"]=Data.mda_fog_index
    output_df.loc[row_index,"mda_complex_word_count"]=Data.mda_complex_word_count
    output_df.loc[row_index,"mda_word_count"]=Data.mda_word_count
    output_df.loc[row_index,"mda_uncertainty_score"]=Data.mda_uncertain_count
    output_df.loc[row_index,"mda_constraining_score"]=Data.mda_constrain_count
    output_df.loc[row_index,"mda_positive_word_proportion"]=Data.mda_positive_word_proportion
    output_df.loc[row_index,"mda_negative_word_proportion"]=Data.mda_negative_word_proportion
    output_df.loc[row_index,"mda_uncertainty_word_proportion"]=Data.mda_uncertainity_word_proportion
    output_df.loc[row_index,"mda_constraining_word_proportion"]=Data.mda_constrain_word_propotion
    output_df.loc[row_index,"qqdmr_positive_score"]=Data.qqdmr_positive_score
    output_df.loc[row_index,"qqdmr_negative_score"]=Data.qqdmr_negative_score
    output_df.loc[row_index,"qqdmr_polarity_score"]=Data.qqdmr_polarity_score
    output_df.loc[row_index,"qqdmr_average_sentence_length"]=Data.qqdmr_avg_sent_length
    output_df.loc[row_index,"qqdmr_percentage_of_complex_words"]=Data.qqdmr_percent_complex
    output_df.loc[row_index,"qqdmr_fog_index"]=Data.qqdmr_fog_index
    output_df.loc[row_index,"qqdmr_complex_word_count"]=Data.qqdmr_complex_word_count
    output_df.loc[row_index,"qqdmr_word_count"]=Data.qqdmr_word_count
    output_df.loc[row_index,"qqdmr_uncertainty_score"]=Data.qqdmr_uncertain_count
    output_df.loc[row_index,"qqdmr_constraining_score"]=Data.qqdmr_constrain_count
    output_df.loc[row_index,"qqdmr_positive_word_proportion"]=Data.qqdmr_positive_word_proportion
    output_df.loc[row_index,"qqdmr_negative_word_proportion"]=Data.qqdmr_negative_word_proportion
    output_df.loc[row_index,"qqdmr_uncertainty_word_proportion"]=Data.qqdmr_uncertainity_word_proportion
    output_df.loc[row_index,"qqdmr_constraining_word_proportion"]=Data.qqdmr_constrain_word_propotion
    output_df.loc[row_index,"rf_positive_score"]=Data.rf_positive_score
    output_df.loc[row_index,"rf_negative_score"]=Data.rf_negative_score
    output_df.loc[row_index,"rf_polarity_score"]=Data.rf_polarity_score
    output_df.loc[row_index,"rf_average_sentence_length"]=Data.rf_avg_sent_length
    output_df.loc[row_index,"rf_percentage_of_complex_words"]=Data.rf_percent_complex
    output_df.loc[row_index,"rf_fog_index"]=Data.rf_fog_index
    output_df.loc[row_index,"rf_complex_word_count"]=Data.rf_complex_word_count
    output_df.loc[row_index,"rf_word_count"]=Data.rf_word_count
    output_df.loc[row_index,"rf_uncertainty_score"]=Data.rf_uncertain_count
    output_df.loc[row_index,"rf_constraining_score"]=Data.rf_constrain_count
    output_df.loc[row_index,"rf_positive_word_proportion"]=Data.rf_positive_word_proportion
    output_df.loc[row_index,"rf_negative_word_proportion"]=Data.rf_negative_word_proportion
    output_df.loc[row_index,"rf_uncertainty_word_proportion"]=Data.rf_uncertainity_word_proportion
    output_df.loc[row_index,"rf_constraining_word_proportion"]=Data.rf_constrain_word_propotion
    output_df.loc[row_index,"constraining_words_whole_report"]=Data.constrain_whole_report



output_df.to_csv("output.csv")




# for index,ix in enumerate(x):
#     print(str(index)+"   "+ix.url+"  "+str(ix.constrain_whole_report))



from flask import Flask,request
import yake
import sys

app = Flask(__name__)



# text = """
# A large black mongrel named Rowf and a white terrier named Snitter escape from an animal experiment center in England's Lake District and, aided by a cunning fox, learn to live on their own, until rumors of slaughtered sheep and bubonic plague-carrying dogs transform them into fugitives. Reprint. 15,000 first printing.
# """
def kw_extractor(full_text):    
    print("testing")
    kw = []
    custom_kw_extractor = yake.KeywordExtractor(n=1, top=3)
    keywords = custom_kw_extractor.extract_keywords(full_text)
    kw = list(dict(keywords).keys())
    return kw


@app.route("/testing",methods=[ "GET"])
def testing():
    
    return 'testing working fine'

@app.route("/keywords_extraction",methods=[ "POST"])
def hello():
    desc = request.get_json()['desc']    
    kwy_words = kw_extractor(desc);
    return kwy_words



if __name__ == '__main__':
    app.run(host="localhost", port=8085, debug=True)
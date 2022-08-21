from flask import Flask,request
import yake
import sys
from ML.model import get_tfidf, get_kw
from ML.keyword_extraction import kw_extractor


app = Flask(__name__)


@app.route("/testing",methods=[ "GET"])
def testing():
    return 'testing working fine'


@app.route("/keywords_extraction",methods=[ "POST"])
def keywrod_extraction():
    desc = request.get_json()['desc']    
    kwy_words = kw_extractor(desc);
    return kwy_words


#main
@app.route("/interests",methods=[ "POST"])
def interests():
    desc = request.get_json()['desc']
    name = request.get_json()['name']    
    get_tfidf()
    return kwy_words





@app.route("/recommendation",methods=[ "POST"])
def recommend_by_us():
    desc = request.get_json()['descs']
    name = request.get_json()['name']
    names = request.get_json()['names']
    
    keywords  = request.get_json()['keyWords']    
    get_tfidf(name)
    get_kw
    return kwy_words






if __name__ == '__main__':
    app.run(host="localhost", port=8085, debug=True)
# note:runn this command !st in the terminal: ollama list 
from ollama import Client
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
app=Flask(__name__)
@app.route("/ask",methods=["POST"])
def ask():
    data=request.get_json() 
    message=data.get("message", "")
    url='https://www.austartify.com/'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    text=soup.get_text(separator=' ')
    list=sent_tokenize(text)
    response=(Client()).chat(model='llama3',messages=[
        {"role":"system","content":f"You are a helpful assistant.Use this context to help the user:{list}"},
        {"role":"user","content":message}
    ])
    return jsonify({
        "message": response['message']['content'],
        "list": []  
    })
@app.route("/")
def home():
    return render_template("chatbot.html")  

if __name__ == "__main__":
    app.run(debug=True)
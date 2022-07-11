from flask import Flask,render_template,request
import flask
import json
import word2vec
import pandas as pd
import re

app = Flask(__name__)
data_CBOW = 0
global vocabulary


@app.route("/home")
def home():
   # response.headers['Access-Control-Allow-Origin'] = '*'
    return render_template("cbowNLP.html")

@app.route("/result", methods = ["POST", "GET"])
def result():
    global vocabulary
    output = request.form.to_dict()
    vocabulary  = output["vocabulary"]
    
    getoutput()

    return render_template("cbowNLP.html", vocabulary = vocabulary)

@app.route('/matrix', methods=["GET"]) #endpoint folder /...
def getoutput():
        print("vocabulary " + vocabulary)
        word2vec.start(vocabulary)
        data_CBOW = word2vec.getData()
        print("Outputing data to Matrix JSON file")
        with open("matrix.json", "w") as outfile: #handling json file
            #data = json.load(outfile) #reading json file, output list of dictionaries

            v = [word.lower() for word in re.compile('\w+').findall(vocabulary)]
            v = ["UNK"] + v

            data_CBOW["words"] = v

            d = data_CBOW
       
            json.dump(d, outfile, indent=4)

            #data.append(data_CBOW)

            #response = requests.post(url, data=data.encode())
            return flask.jsonify(d) #sending a valid response, navigate to http://localhost:6969/matrix to see response

if __name__ == "__main__":
    
    app.run("localhost", 6969)
    #app.run(debug=True)




'''References: 
	- https://tms-dev-blog.com/python-backend-with-javascript-frontend-how-to/

'''
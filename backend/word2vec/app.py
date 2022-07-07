from flask import Flask
import flask
import json
import word2vec
import pandas as pd

app = Flask(__name__)
data_CBOW = 0


@app.route("/")
def hello():
    return "Hello, World!"
    



@app.route('/matrix', methods=["GET"]) #endpoint folder /...
def users():

    print("Outputing data to Matrix JSON file")
    with open("matrix.json", "w") as outfile: #handling json file
        #data = json.load(outfile) #reading json file, output list of dictionaries

        d = data_CBOW
   
        json.dump(d, outfile, indent=4)

        #data.append(data_CBOW)

        #response = requests.post(url, data=data.encode())
        return flask.jsonify(d) #sending a valid response, navigate to http://localhost:6969/matrix to see response


if __name__ == "__main__":
    word2vec.start()
    data_CBOW = word2vec.getData()
    app.run("localhost", 6969)




'''References: 
	- https://tms-dev-blog.com/python-backend-with-javascript-frontend-how-to/

'''
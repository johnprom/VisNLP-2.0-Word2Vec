from flask import Flask
import flask
import json

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/matrix', methods=["GET"]) #endpoint folder /...
def users():
    print("Outputing data to Matrix JSON file")
    with open("matrix.json", "r") as f: #handling json file
        data = json.load(f) #reading json file, output list of dictionaries
        #TODO: 
        #	- Extract python Data
        #	- make data append data from python

        data.append({
            "epoch": "2",
            "weights": [0.45, 0.86, 0.95, 0.87, 0.98]
            
        })
        return flask.jsonify(data) #sending a valid response, navigate to http://localhost:6969/users to see response


if __name__ == "__main__":
    app.run("localhost", 6969)




'''References: 
	- https://tms-dev-blog.com/python-backend-with-javascript-frontend-how-to/

'''
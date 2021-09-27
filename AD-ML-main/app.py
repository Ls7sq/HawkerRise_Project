import pandas as pd
from flask import Flask, request, jsonify
import mysql.connector
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
import time
import json

app = Flask(__name__)

@app.route('/')
def hello():
    return 'This is Team 5 ADProj ML App.'

@app.route('/highestRatedStalls', methods=['GET'])
def callModelHighestRated():
    
    # get start time
    start_time = time.time()
    
    # connect to MySQL Database
    database = mysql.connector.connect(
      host="35.197.132.116",
      user="root",
      password="adproject",
      database="hawkerise")
    
    mycursor = database.cursor()
    
    mycursor.execute("SELECT * FROM rating")
    
    myresult = mycursor.fetchall()
    
    # put data into DataFrame
    df = pd.DataFrame(myresult, columns = ["ID", "UserID", "Rating", "StallID"], index=None)
    
    # select only rows with rating > 3 
    df.loc[df.Rating > 3, :]
    
    # group DataFrame by StallID
    df = df.groupby(["StallID"]).sum()
    
    # sort DataFrame by highest ratings
    df.sort_values("Rating", axis = 0, ascending = False, inplace = True)
    
    # return top 5 highest rated stalls as JSON array
    df_stall = pd.DataFrame(columns = ["id", "close_hours", "contact_number", "first_name", "hawker_img", "last_name", "operating_hours", "password", "photo", "stall_name", "status", "tags", "unit_number", "user_name", "centre_id"], index=None)
    
    for y in df.head(5).index.tolist():
        mycursor.execute("SELECT * FROM hawker WHERE id=" + str(y))
        stall = mycursor.fetchall()
        add_stall = pd.DataFrame(stall, columns = ["id", "close_hours", "contact_number", "first_name", "hawker_img", "last_name", "operating_hours", "password", "photo", "stall_name", "status", "tags", "unit_number", "user_name", "centre_id"], index=None)
        df_stall = df_stall.append(add_stall)
        
    df_stall.drop(['tags'], axis=1, inplace=True)
    
    # get end time
    end_time = time.time()
    print("Time Taken:" + str(end_time-start_time))
    
    return json.dumps(json.loads(df_stall.reset_index().to_json(orient='records')), indent=2)
    
@app.route('/recommendStalls', methods=['GET'])
def callModelRecommender():
    uid = request.args.get('uid', type = str)
    
    # get start time
    start_time = time.time()
    
    # connect to MySQL Database
    database = mysql.connector.connect(
      host="35.197.132.116",
      user="root",
      password="adproject",
      database="hawkerise")
    
    mycursor = database.cursor()
    
    mycursor.execute("SELECT * FROM rating")
    
    myresult = mycursor.fetchall()
    
    # put data into DataFrame
    df = pd.DataFrame(myresult, columns = ["ID", "UserID", "Rating", "StallID"], index=None)
    
    # read data into Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["UserID", "StallID", "Rating"]], reader)
    
    # train model
    sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
    }
    algo = KNNWithMeans(sim_options=sim_options)
    
    trainingSet = data.build_full_trainset()
    
    algo.fit(trainingSet)
    
    # find top 5 highest rated stalls
    computations = pd.DataFrame(columns=["PredictedRating"], index=pd.RangeIndex(1,101))
    for x in range(1,101):
        computations.loc[x] = algo.predict(uid,x).est
    computations.drop(computations[computations.PredictedRating == 1].index, inplace=True)
    computations.sort_values(by=["PredictedRating"],inplace=True,ascending=False)
    
    # return top 5 highest reccomended stalls as JSON array
    df_stall = pd.DataFrame(columns = ["id", "close_hours", "contact_number", "first_name", "hawker_img", "last_name", "operating_hours", "password", "photo", "stall_name", "status", "tags", "unit_number", "user_name", "centre_id"], index=None)
    
    for y in computations.head(5).index.tolist():
        mycursor.execute("SELECT * FROM hawker WHERE id=" + str(y))
        stall = mycursor.fetchall()
        add_stall = pd.DataFrame(stall, columns = ["id", "close_hours", "contact_number", "first_name", "hawker_img", "last_name", "operating_hours", "password", "photo", "stall_name", "status", "tags", "unit_number", "user_name", "centre_id"], index=None)
        df_stall = df_stall.append(add_stall)
        
    df_stall.drop(['tags'], axis=1, inplace=True)
    
    # get end time
    end_time = time.time()
    print("Time Taken:" + str(end_time-start_time))
    
    return json.dumps(json.loads(df_stall.reset_index().to_json(orient='records')), indent=2)
    
# run the server
if __name__ == '__main__':
    app.run(port=5000, debug=True)

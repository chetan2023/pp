import pickle
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict
import random
from fastapi.exceptions import HTTPException

app = FastAPI()
handler = Mangum(app)

traits = pd.read_csv(r"traits/clusterswithtraits.csv")
question_set = pd.read_csv(r"question_sets/4set_questions.csv")



#############################################
from pydantic import BaseModel
from typing import Dict, List

class PredictReq(BaseModel):
    set_number: int
    score: Dict[str, float]

class PredictResponse(BaseModel):
    cluster: int
    personality: str
    traits: str
    overallscore: int
    results: Dict[str, float]

class Question(BaseModel):
    bigfive: str
    question: str
    question_code: str

class QuestionResp(BaseModel):
    set_number: int
    questions: List[Question]

####################################################

# from schema import Question, QuestionResp

@app.get("/questions", response_model=QuestionResp)
async def read_questions():
    question_resp = []
    set_list = [1, 2, 3, 4]
    SET = random.choice(set_list)
    setwise_question_df = question_set[question_set["SET"] == int(SET)]
    for index in setwise_question_df.index:
        question_resp.append(
            Question(
                bigfive=setwise_question_df["BigFive"][index],
                question_code=setwise_question_df["Question Code"][index],
                question=setwise_question_df["Question"][index],
            )
        )
    return QuestionResp(set_number=int(SET), questions=question_resp)

# from schema import PredictReq, PredictResponse
# from validate_score import validate_predict_req

#########################################################
# TODO: recheck
set1_question_code = [
    "EXT3",
    "EXT4",
    "EXT7",
    "EXT8",
    "EST1",
    "EST2",
    "EST3",
    "EST10",
    "AGR1",
    "AGR2",
    "AGR5",
    "AGR8",
    "CSN3",
    "CSN4",
    "CSN5",
    "CSN8",
    "OPN5",
    "OPN6",
    "OPN7",
    "OPN10",
]
# set1_question_code = set1_question_code.sort()
# TODO: recheck
set2_question_code = [
    "EXT1",
    "EXT4",
    "EXT6",
    "EXT8",
    "EST2",
    "EST4",
    "EST7",
    "EST8",
    "AGR5",
    "AGR6",
    "AGR9",
    "AGR10",
    "CSN1",
    "CSN2",
    "CSN6",
    "CSN7",
    "OPN1",
    "OPN3",
    "OPN4",
    "OPN8",
]
# set2_question_code = set2_question_code.sort()
# TODO: recheck
set3_question_code = [
    "EXT1",
    "EXT2",
    "EXT7",
    "EXT8",
    "EST2",
    "EST5",
    "EST8",
    "EST10",
    "AGR2",
    "AGR5",
    "AGR7",
    "AGR10",
    "CSN1",
    "CSN3",
    "CSN5",
    "CSN10",
    "OPN2",
    "OPN3",
    "OPN5",
    "OPN6",
]
# set3_question_code = set3_question_code.sort()
# TODO: recheck
set4_question_code = [
    "EXT5",
    "EXT2",
    "EXT4",
    "EXT9",
    "EST9",
    "EST2",
    "EST6",
    "EST4",
    "AGR9",
    "AGR4",
    "AGR7",
    "AGR10",
    "CSN3",
    "CSN9",
    "CSN6",
    "CSN7",
    "OPN2",
    "OPN10",
    "OPN7",
    "OPN1",
]
# set4_question_code = set4_question_code.sort()
def error_string(set_number : int, question_code: str):
    return Exception("keys in dict should be in " + str(question_code) + " for set number " + str(set_number))

# from schema import PredictReq
def validate_predict_req(data: PredictReq):
    if data.set_number == 1 and list(data.score.keys()) != set1_question_code:
        raise error_string(data.set_number, str(set1_question_code))
    if data.set_number == 2 and list(data.score.keys()) != set2_question_code:
        raise error_string(data.set_number, str(set2_question_code))
    if data.set_number == 3 and list(data.score.keys()) != set3_question_code:
        raise error_string(data.set_number, str(set3_question_code))
    if data.set_number == 4 and list(data.score.keys()) != set4_question_code:
        raise error_string(data.set_number, str(set4_question_code))

##################################################################

@app.post("/predict", response_model=PredictResponse)
def predict_cluster(data: PredictReq):
    """
    Score will be question code and values
    """

    try:
      validate_predict_req(data)
    except Exception as e:
      raise HTTPException(
         status_code=400,
         detail=str(e)
      )
    print("##########HeRE PREDICT THE API######")
    df = pd.DataFrame([data.score], columns=data.score.keys())
    print("dataframe created", df)
    # Load the saved model
    model_file = str(data.set_number) + "pp.pkl"
    model = "models/" + model_file
    with open(model, "rb") as file:
      classifier = pickle.load(file)
    print("Loaded model", model)
    print(type(classifier))
    # df = df.sort_index(axis=1)
    y_pred = classifier.predict(df)
    my_cluster = y_pred[0]
    print("My Cluster", my_cluster)
    my_df = traits[traits["SET"] == data.set_number]
    my_df = my_df[my_df["Clusters"] == my_cluster]
    my_df = my_df[
        [
            "SET",
            "Clusters",
            "Extroversion",
            "Neurotic",
            "Agreeable",
            "Conscientious",
            "Openness",
            "Label",
            "Traits",
        ]
    ]
    personality = my_df["Label"].to_string(index=False)
    my_df = my_df.reset_index(drop=True)
    overall_score = (
        my_df["Openness"]
        + my_df["Conscientious"]
        + my_df["Extroversion"]
        + my_df["Agreeable"]
        + my_df["Neurotic"]
    ) / 5
    my_sums = pd.DataFrame()
    my_sums["Extroversion"] = my_df["Extroversion"]
    my_sums["Agreeable"] = my_df["Agreeable"]
    my_sums["Conscientious"] = my_df["Conscientious"]
    my_sums["Openness"] = my_df["Openness"]
    my_sums["Neurotic"] = my_df["Neurotic"]
    my_sums = my_sums[
        ["Agreeable", "Conscientious", "Openness", "Extroversion", "Neurotic"]
    ]
    my_results = my_sums.T
    my_results = my_results.reset_index().rename(
        columns={"index": "Personality", int(0): "Score"}
    )
    my_results = my_results.set_index("Personality")["Score"].to_dict()
    mytraits = " ".join(str(i) for i in my_df["Traits"])
    return PredictResponse(
        cluster=my_cluster,
        personality=personality,
        traits=mytraits,
        overallscore=overall_score,
        results=my_results,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999, reload=True)
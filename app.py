from flask import Flask,render_template,request,jsonify
'''rednder template -used to call html file'''

from src.pipeline.prediction_pipeline import PredictionPileline,CustomClass


app=Flask(__name__)

@app.route("/",methods=["GET","POST"])       # where we want to run our appluication
def prediction_data():
    if request.method=="GET":
        return  render_template("home.html")
    else:
        data=CustomClass(
            age=int(request.form.get("age")),
            workclass=int(request.form.get("workclass")),
            education_num=int(request.form.get("education_num")),
            marital_status=int(request.form.get("marital_status")),
            occupation=int(request.form.get("occupation")),
            relationship=int(request.form.get("relationship")),
            race=int(request.form.get("race")),
            sex=int(request.form.get("sex")),
            capital_gain=int(request.form.get("capital_gain")),
            capital_loss=int(request.form.get("capital_loss")),
            hours_per_week=int(request.form.get("hours_per_week")),
            native_country=int(request.form.get("native_country"))


        )


    final_data=data.get_data_frame()
    pipeline_prediction=PredictionPileline()
    pred=pipeline_prediction.predict(final_data)

    result=pred

    if result==0:
        return render_template("results.html",final_result="Your Yearly Income is less than Equal to 50k: {}".format(result))
    elif result==1:
        return render_template("results.html",final_result="Your Yearly Income is More than Equal to 50k: {}".format(result))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)





























# import os,sys 
# from src.logger import logging
# from src.exception import CustomeException

# from src.pipeline.prediction_pipeline import 

# app=Flask(__name__)

# @app.route('/',methods=['GET','POST'])



# def index():
#     try:
#         raise Exception("We are testing")
#     except Exception as e:
#         abc=CustomeException(e,sys)
#         logging.info(abc.error_message)
#         return "Welcome here"
    


# if __name__=="__main__":
#     app.run(debug=True)



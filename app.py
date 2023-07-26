from flask import Flask
import os,sys 
from src.logger import logging
from src.exception import CustomeException

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])



def index():
    try:
        raise Exception("We are testing")
    except Exception as e:
        abc=CustomeException(e,sys)
        logging.info(abc.error_message)
        return "Welcome here"
    


if __name__=="__main__":
    app.run(debug=True)



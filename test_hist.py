from flask import Flask, render_template, request, redirect, make_response, send_file, session
from flask_bootstrap import Bootstrap
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
app = Flask(__name__)

bootstrap = Bootstrap(app)

@app.route('/',methods=['GET', 'POST'])  
def Text(): 
    data = pd.read_csv("insurance_train.csv")
    columns=data.columns
    for e in data.columns:
        ax = data[e].plot.hist()
        ax.set_xlabel(e)
        ax.set_ylabel('Count')
        
        plt.savefig("static\\"+e+".png")
        plt.show()
    return render_template("test_hist.html",columns=columns) 

if __name__ == '__main__':
    app.run(port=5000,debug=True)
from distutils.log import debug 
from fileinput import filename 
from flask import Flask, render_template, request  
import os
import mom

app = Flask(__name__)   
  
@app.route('/')   
def main():   
    return render_template("index.html")   
  
@app.route('/uploadphoto', methods = ['POST'])   
def uploadphoto():   
    if request.method == 'POST':
        flist = request.files.getlist('file')
        if not flist:
            return render_template("confirmation.html")
        for f in flist:
            if not "mov" in f.filename.lower():
                f.save("../shrockers/" + f.filename)   
            else:
                f.save("../MOMents/" + f.filename)
        
        print("here")
        
        mom.generate_video()
        # with open("mom.py", 'r') as file:
        #     python_code = file.read()
        #     exec(python_code)
        # os.system("python mom.py")
        return render_template("confirmation.html")  
if __name__ == '__main__':   
    app.run(debug=True)
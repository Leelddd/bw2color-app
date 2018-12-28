from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)


# todo get result image from /static/upload/img to /static/result/img by dataset model
def get_result(dataset, img):
    pass


@app.route('/')
@app.route('/basic')
def basic():
    return render_template('index.html')


@app.route('/upload/image', methods=['POST'])
def upload_image():
    dataset = request.args.get('dataset')
    print(dataset)
    f = request.files['file']
    filename = 'static/upload/' + f.filename
    f.save(filename)

    get_result(dataset, f.filename)
    return f.filename


if __name__ == '__main__':
    app.run()

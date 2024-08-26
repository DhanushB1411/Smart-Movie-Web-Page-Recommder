import os
import pickle
import numpy as np

from flask import Flask,render_template,request,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Movies(db.Model):
    movieid = db.Column(db.Integer,primary_key = True)
    moivename = db.Column(db.String(100), nullable=False)
    review = db.Column(db.String(100), nullable=False)
    emotion = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())

    def __repr__(self):
        return f'<Movie {self.moivename}>'


model = pickle.load(open('model.pkl','rb'))

# Load the vectorizer along with the model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

from nltk.tokenize import RegexpTokenizer
#for tokexizing the data into
# "my name is"  => ["my","name","is"]
from nltk.stem.porter import PorterStemmer
#cleaning the data like "liking " -> "like"
from nltk.corpus import  stopwords
# to remove the unwanted data like the is
import nltk
nltk.download('stopwords')
# Downloading the stopwords
#tokenizer with spaceblank
tokenizer = RegexpTokenizer(r"\w+")

en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
def getCleanedText(text):
  text = text.lower()
  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text



@app.route("/")
def hello_world():
    #moviesall = db.session.query(Movies.moivename).distinct().all()
    #dict = {}
    # for movies in moviesall:
    #     positive_count = Movies.query.filter_by(moivename = movies,emotion='Positive').count()
    #     negative_count = Movies.query.filter_by(moivename = movies,emotion='Negative').count()
    #     dict["movie"] = movies
    #     dict["positive_count"]= positive_count
    #     dict["negative_count"] = negative_count
    reviews = Movies.query.all()
    return render_template('index.html',Reviews = reviews)#movies = moviesall)


@app.route("/predict", methods=['POST'])
def predict():
    try:
        review = request.form['review']
        cleaned_review = getCleanedText(review)
        rev = vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(rev)
        pred = 'Positive' if prediction[0] == 1 else 'Negative'
        prediction_text = 'POSITIVE RESPONSE' if prediction[0] == 1 else 'NEGATIVE RESPONSE'
        
        new_review = Movies(moivename='fight Club', review=review, emotion=pred)
        db.session.add(new_review)
        db.session.commit()
        
        reviews = Movies.query.all()
        return render_template('movie.html', prediction_text=prediction_text, Reviews=reviews)
    except Exception as e:
        return str(e)



@app.route("/film")
def home():
        reviews = Movies.query.filter_by(moivename = 'Fight Club')
        return render_template("movie.html",Reviews = reviews)


@app.route("/batman",methods = ['POST','GET'])
def batman():
    movie_name = request.args.get('movie', 'Batman')
    if request.method == 'POST':
        try:
            review = request.form['review']
            movie_name = request.form['movie']
            cleaned_review = getCleanedText(review)
            rev = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(rev)

            pred = 'Positive' if prediction[0] == 1 else 'Negative'
            prediction_text = 'POSITIVE RESPONSE' if prediction[0] == 1 else 'NEGATIVE RESPONSE'

            new_review = Movies(moivename=movie_name, review=review, emotion=pred)
            db.session.add(new_review)
            db.session.commit()

            reviews = Movies.query.filter_by(moivename = movie_name)
            positive_count = Movies.query.filter_by(moivename = movie_name,emotion='Positive').count()
            negative_count = Movies.query.filter_by(moivename = movie_name,emotion='Negative').count()
            return render_template('check.html', prediction_text=prediction_text, Reviews=reviews,positive_count=positive_count, negative_count=negative_count)
        except Exception as e:
            return str(e)
    else:
        reviews = Movies.query.filter_by(moivename = movie_name)
        positive_count = Movies.query.filter_by(moivename = movie_name,emotion='Positive').count()
        negative_count = Movies.query.filter_by(moivename = movie_name,emotion='Negative').count()
        return render_template('check.html',Reviews=reviews,movie_name = movie_name,positive_count=positive_count, negative_count=negative_count)

@app.route("/count_reviews")
def count_reviews():
    positive_count = Movies.query.filter_by(moivename = 'fight Club',emotion='Positive').count()
    negative_count = Movies.query.filter_by(moivename = 'fight Club',emotion='Negative').count()
    return render_template('count.html', positive_count=positive_count, negative_count=negative_count)

    
if __name__ == "__main__":
    app.run()


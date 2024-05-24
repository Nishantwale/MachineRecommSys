from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response
from flask_sqlalchemy import SQLAlchemy
import hashlib
import pickle
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import base64
import cv2
import json
import face_recognition
from werkzeug.utils import secure_filename

import face_recognition_models
from sklearn.linear_model import LinearRegression
from functools import wraps

# Load pickled data
popular_df = pickle.load(open('popular_restaurant.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
restaurant = pickle.load(open('restaurant.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
plt.switch_backend('agg')

app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

def load_reviews():
    if os.path.exists('reviews.json'):
        try:
            with open('reviews.json', 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            return []
    return []

def save_reviews(reviews):
    with open('reviews.json', 'w') as file:
        json.dump(reviews, file)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=True)  # Allow NULL for face recognition users
    face_encoding = db.Column(db.PickleType, nullable=True)  # Store face encoding
    favorites = db.relationship('Favorite', backref='user', lazy=True)

# Define Favorite model
class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    restaurant_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    image = db.Column(db.String(100), nullable=False)
    cost = db.Column(db.String(100), nullable=False)
    rate = db.Column(db.Float, nullable=False)
    url = db.Column(db.String(100), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('', 'error')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

def no_cache(view):
    @wraps(view)
    def no_cache_response(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return no_cache_response

@app.route('/')
@no_cache
def index():
    latest_reviews = load_reviews()
    return render_template('index.html', latest_reviews=latest_reviews)

@app.route('/home')
@login_required
@no_cache
def home():
    # Render template with restaurant data
    restaurant_data = popular_df.head(50)
    # Prepare data for regression
    X = restaurant_data['cost'].values.reshape(-1, 1)
    y = restaurant_data['avg_votes'].values.reshape(-1, 1)
    # Fit linear regression model
    regression_model = LinearRegression()
    regression_model.fit(X, y)
    predicted_avg_votes = regression_model.predict(X)
    # Plot the linear regression
    plt.figure(figsize=(10, 8))
    plt.scatter(X.flatten(), y.flatten(), color='blue', label='Actual data')
    plt.plot(X.flatten(), predicted_avg_votes.flatten(), color='red', linewidth=2, label='Linear regression')
    plt.title('Linear Regression: Average Votes vs Cost for Top 50 Restaurants', fontsize=20)
    plt.xlabel('Cost (for two people)')
    plt.ylabel('Average Votes')
    plt.legend()
    plt.grid(True)
    regression_plot_path = "static/regression_plot.png"
    plt.savefig(regression_plot_path)
    plt.clf()
    with open(regression_plot_path, "rb") as img_file:
        encoded_plot = base64.b64encode(img_file.read()).decode('utf-8')
    os.remove(regression_plot_path)
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'purple', 'pink']
    plt.bar(range(50), predicted_avg_votes.flatten()[:50], color=colors)
    plt.xlabel('Restaurant Index')
    plt.ylabel('Predicted Average Votes')
    plt.title('Bar Regression Graph: Predicted Average Votes for Top 50 Restaurants', fontsize=20)
    plt.grid(True)
    bar_plot_path = "static/bar_regression_plot.png"
    plt.savefig(bar_plot_path)
    plt.clf()
    with open(bar_plot_path, "rb") as img_file:
        encoded_bar_plot = base64.b64encode(img_file.read()).decode('utf-8')
    os.remove(bar_plot_path)
    return render_template('home.html',
                           restaurant_name=list(popular_df['name'].values),
                           Location=list(popular_df['location'].values),
                           Image=list(popular_df['Image'].values),
                           cost=list(popular_df['cost'].values),
                           votes=list(popular_df['num_votes'].values),
                           Ratings=list(popular_df['avg_votes'].values),
                           URL=list(popular_df['url'].values),
                           restaurant_data=restaurant_data.to_dict(orient='records'),
                           regression_plot=encoded_plot,
                           bar_regression_plot=encoded_bar_plot)

@app.route('/profile')
@login_required
@no_cache
def profile():
    user = User.query.filter_by(username=session['username']).first()
    if user:
        return render_template('profile.html', username=user.username, email=user.email)
    else:
        flash('User not found.', 'error')
        return redirect(url_for('signin'))

@app.route('/recommend')
@login_required
@no_cache
def recommend_restaurant():
    return render_template('recommend_restaurant.html')

@app.route('/recommend_restaurant', methods=['POST'])
@login_required
@no_cache
def recommend():
    user_input = request.form.get('user_input')
    restaurant_name_lower = user_input.lower()
    pt_index_lower = [name.lower() for name in pt.index]
    if restaurant_name_lower not in pt_index_lower:
        message = f"Restaurant '{user_input}' not found."
        explore_message = "Explore these Restaurants instead:"
        random_restaurants = random.sample(pt.index.tolist(), 5)
        recommendations = []
        for name in random_restaurants:
            temp_df = restaurant[restaurant['name'] == name]
            item = [
                name,
                temp_df['location'].iloc[0],
                temp_df['Image'].iloc[0],
                temp_df['cost'].iloc[0],
                temp_df['rate'].iloc[0] if not pd.isnull(temp_df['rate'].iloc[0]) else 4.2,
                temp_df['url'].iloc[0]
            ]
            recommendations.append(item)
        recommendation_data = []
        for restaurant_info in recommendations:
            recommendation_data.append((float(restaurant_info[3].replace(',', '')), float(restaurant_info[4])))
        X_recommend = np.array([info[0] for info in recommendation_data]).reshape(-1, 1)
        y_recommend = np.array([info[1] for info in recommendation_data])
        model_recommend = LinearRegression()
        model_recommend.fit(X_recommend, y_recommend)
        predicted_ratings_recommend = model_recommend.predict(X_recommend)
        plt.clf()
        plt.figure(figsize=(10, 8))
        plt.scatter(X_recommend, y_recommend, label='Data Points')
        plt.plot(X_recommend, model_recommend.predict(X_recommend), color='red', label='Linear Regression')
        plt.xlabel('Cost')
        plt.ylabel('Rating')
        plt.title('Regression Graph For Recommendations', fontsize=20)
        plt.legend()
        plt.grid(True)
        plot_path = "static/regression_plot.png"
        plt.savefig(plot_path)
        plt.close()
        with open(plot_path, "rb") as img_file:
            encoded_plot = base64.b64encode(img_file.read()).decode('utf-8')
        os.remove(plot_path)
        plt.clf()
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'orange', 'purple', 'pink']
        plt.bar(range(len(recommendations)), [info[1] for info in recommendation_data], color=colors)
        plt.xlabel('Recommendation Index')
        plt.ylabel('Actual Rating')
        plt.title('Bar Graph For the Recommendations', fontsize=20)
        plt.tight_layout()
        bar_plot_path = "static/bar_plot.png"
        plt.savefig(bar_plot_path)
        plt.close()
        with open(bar_plot_path, "rb") as img_file:
            encoded_bar_plot = base64.b64encode(img_file.read()).decode('utf-8')
        os.remove(bar_plot_path)
        return render_template('recommend_restaurant.html', message=message, explore_message=explore_message,
                               recommendations=recommendations, regression_plot=encoded_plot,
                               bar_plot=encoded_bar_plot)
    index = pt_index_lower.index(restaurant_name_lower)
    restaurant_name_actual = pt.index[index]
    similar = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    for i in similar:
        item = []
        temp_df = restaurant[restaurant['name'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('name')['name'].values))
        item.extend(list(temp_df.drop_duplicates('name')['location'].values))
        item.extend(list(temp_df.drop_duplicates('name')['Image'].values))
        item.extend(list(temp_df.drop_duplicates('name')['cost'].values))
        item.extend(list(temp_df.drop_duplicates('name')['rate'].values))
        item.extend(list(temp_df.drop_duplicates('name')['url'].values))
        data.append(item)
    recommendation_data = []
    for restaurant_info in data:
        recommendation_data.append((float(restaurant_info[3].replace(',', '')), float(restaurant_info[4])))
    X = np.array([info[0] for info in recommendation_data]).reshape(-1, 1)
    y = np.array([info[1] for info in recommendation_data])
    model = LinearRegression()
    model.fit(X, y)
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y, label='Data Points')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression')
    plt.xlabel('Cost')
    plt.ylabel('Rating')
    plt.title('Regression Graph For Recommendations', fontsize=20)
    plt.legend()
    plt.grid(True)
    plot_path = "static/regression_plot.png"
    plt.savefig(plot_path)
    plt.close()
    with open(plot_path, "rb") as img_file:
        encoded_plot = base64.b64encode(img_file.read()).decode('utf-8')
    os.remove(plot_path)
    plt.clf()
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'purple', 'pink']
    plt.bar(range(len(data)), [info[1] for info in recommendation_data], color=colors)
    plt.xlabel('Recommendation Index')
    plt.ylabel('Actual Rating')
    plt.title('Bar Graph For the Recommendations', fontsize=20)
    plt.tight_layout()
    bar_plot_path = "static/bar_plot.png"
    plt.savefig(bar_plot_path)
    plt.close()
    with open(bar_plot_path, "rb") as img_file:
        encoded_bar_plot = base64.b64encode(img_file.read()).decode('utf-8')
    os.remove(bar_plot_path)
    return render_template('recommend_restaurant.html', data=data, regression_plot=encoded_plot,
                           bar_plot=encoded_bar_plot)

@app.route('/update_rating', methods=['POST'])
@login_required
@no_cache
def update_rating():
    data = request.json
    restaurant_name = data['name']
    rating = data['rating']
    restaurant.loc[restaurant['name'] == restaurant_name, 'rate'] = rating
    restaurant.to_csv('zomato.csv', index=False)

@app.route('/contact')
@no_cache
def contact():
    return render_template('contact.html')

@app.route('/signup', methods=['GET', 'POST'])
@no_cache
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        image_data = request.form.get('image')  # Base64 encoded image
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))
        elif existing_email:
            flash('Email already exists.', 'error')
            return redirect(url_for('signup'))
        else:
            # Decode the image
            image_data = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Get face encoding
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                face_encoding = face_encodings[0]
                new_user = User(username=username, email=email, face_encoding=face_encoding)
                db.session.add(new_user)
                db.session.commit()
                flash('Registered successfully.', 'success')
                return redirect(url_for('signup'))
            else:
                flash('No face detected. Please try again.', 'error')
                return redirect(url_for('signup'))
    else:
        return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
@no_cache
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        image_data = request.form.get('image')  # Base64 encoded image
        user = User.query.filter_by(username=username).first()
        if user:
            # Decode the image
            image_data = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Get face encoding
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                face_encoding = face_encodings[0]
                # Compare face encodings
                match = face_recognition.compare_faces([user.face_encoding], face_encoding)[0]
                if match:
                    session['username'] = username
                    flash('Logged in successfully.', 'success')
                    return redirect(url_for('profile'))
                else:
                    flash('Invalid username or image.', 'error')
                    return redirect(url_for('signin'))
            else:
                flash('No face detected. Please try again.', 'error')
                return redirect(url_for('signin'))
        else:
            flash('Invalid username or image.', 'error')
            return redirect(url_for('signin'))
    else:
        return render_template('signin.html')

@app.route('/signout')
@no_cache
def signout():
    session.pop('username', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/add_favorite', methods=['POST'])
@login_required
@no_cache
def add_favorite():
    data = request.json
    restaurant_name = data['name']
    location = data['location']
    image = data['image']
    cost = data['cost']
    rate = data['rate']
    url = data['url']
    existing_favorite = Favorite.query.filter_by(user_id=session['username'], restaurant_name=restaurant_name).first()
    if existing_favorite:
        flash('Restaurant already in favorites.', 'error')
    else:
        new_favorite = Favorite(user_id=session['username'], restaurant_name=restaurant_name, location=location, image=image, cost=cost, rate=rate, url=url)
        db.session.add(new_favorite)
        db.session.commit()
        flash('Restaurant added to favorites successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/favorites')
@login_required
@no_cache
def favorites():
    user_favorites = Favorite.query.filter_by(user_id=session['username']).all()
    return render_template('favorites.html', favorites=user_favorites)


@app.route('/feedback')
@no_cache
def feedback():
    return render_template('feedback.html')


latest_reviews = []

@app.route('/submit_feedback', methods=['POST'])
@no_cache
def submit_feedback():
    global latest_reviews
    if request.method == 'POST':
        name = request.form['name']
        qualification = request.form['qualification']
        review = request.form['review']
        image = request.files['image']

        if image:
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            latest_reviews = load_reviews()
            latest_reviews.insert(0, {
                'review': review,
                'image': url_for('static', filename=f'uploads/{filename}'),
                'name': name,
                'qualification': qualification
            })
            # Keep only the latest three reviews
            if len(latest_reviews) > 3:
                latest_reviews.pop()
            save_reviews(latest_reviews)

    return render_template('index.html', latest_reviews=latest_reviews)

if __name__ == '__main__':
    app.run(debug=True)

# Restaurant Recommendation System
## Overview
This project is an advanced Restaurant Recommendation System deployed on Heroku, designed to enhance user experience through personalized recommendations and interactive features. It leverages Flask for backend functionality and integrates HTML, CSS, and Bootstrap for the frontend. The system utilizes machine learning algorithms for providing recommendations and incorporates face recognition for secure user authentication.

## Features
### Before Login
#### Index Page
   - Description: The landing page for users before they log in.
##### Features:
  - Displays the project title, a brief description, and a random image.
  - Navigation bar with menus: Home, About, Top Picks, Explore, Discover, Reviews, Contact, Feedback, SignUp, and SignIn.
  - Footer with additional links and information.

##### Contact Page
  - Description: Allows users to send messages directly.
##### Features:
  - Contact form that collects email, name, and message.
  - Sends the collected information to the specified email.
      
#### Feedback Page
  - Description: Collects feedback from users.
##### Features:
  - Form with fields for name, qualification, image upload, and message.
  - Displays the latest three feedbacks dynamically on the index page.
    
#### SignUp Page
  - Description: Facilitates user registration.
##### Features:
  - Advanced signup form with username, email, and a Capture Image button.
  - Integrated with the face_recognition library for face authentication.
  - Captured images are stored in an SQLite database in encrypted format.
    
#### SignIn Page
  - Description: Allows users to log in securely.
##### Features:
  - Login form with username and face capture.
  - Matches captured image with the stored image for authentication.
    
### After Login

#### Home Page
  - Description: Personalized home page for logged-in users.
##### Features:
  - Displays user profile with username, email, and logout button.
  - Navigation bar with menus: Home, Explore, Recommend, and Favorite.
  - Footer with additional links and information.
    
#### Explore
  - Description: Showcases the top 50 restaurants.
##### Features:
  - Uses Zomato dataset from Kaggle.
  - Restaurants ranked by average rating, with details like image, location, cost, votes, and rating.
  - Linear regression and bar graphs for visual representation.
    
#### Recommend Section
  - Description: Provides personalized restaurant recommendations.
##### Features:
  - Content-based filtering to recommend restaurants based on user search.
  - Displays recommended restaurants with details like image, cost, location, and rating.
  - Option to add recommended restaurants to the favorite section.
  - Linear regression and bar graphs for visual insights.
    
#### Favorite Section
  - Description: Shows user’s favorite restaurants.
##### Features:
  - Displays a list of restaurants marked as favorites by the user.
  - Facilitates quick access to preferred dining options.

## Technical Details

### Frontend
#### Technologies Used: HTML, CSS, Bootstrap.
#### Functionality: User-friendly interface for seamless interaction.

### Backend
#### Technologies Used: Python, Flask, SQLite.
#### Functionality: Handles user authentication, data processing, and recommendation algorithms.

### Machine Learning
#### Libraries Used: Pandas, NumPy, Matplotlib, Scikit-learn.
#### Algorithms: Popularity-based and content-based filtering.

## Installation
  - To run this project locally:

1. Clone the repository:
   git clone https://github.com/yourusername/restaurant-recommendation-system.git
   
2. Navigate to the project directory:
   cd restaurant-recommendation-system
   
3. Create a virtual environment:
   python -m venv venv
   
4. Activate the virtual environment:
   - On Windows:
     venv\Scripts\activate
   - On MacOS/Linux:
     source venv/bin/activate
     
5. Install the required dependencies:
   pip install -r requirements.txt
   
6. Set up the database:
   flask db init
   flask db migrate
   flask db upgrade

7. Run the application:
   flask run

8. Open your browser and navigate to:
   http://127.0.0.1:5000/


## Usage
  1. Signup: Register by capturing your image and entering your username and email.
  2. Login: Authenticate using face recognition.
  3. Explore: Browse top 50 restaurants.
  4. Recommend: Get personalized restaurant recommendations.
  5. Favorite: Save preferred restaurants to your favorites list.

   
## Contributing
  - Feel free to fork this repository and contribute by submitting a pull request. Please make sure to update tests as appropriate.


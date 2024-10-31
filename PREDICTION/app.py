import pandas as pd
from flask import Flask, render_template, request, send_file
from main import forecast_sales, generate_plot
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('start.html')

# Home route to display form and results
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        # Get user inputs from the form
        category = request.form['category']
        date = request.form['date']
        input_date = pd.to_datetime(date)

        # Generate forecast value and plot based on inputs
        forecast_value, closest_prediction_date, plot_file = forecast_sales(category, date)

        # Render results back to the result template
        return render_template('result.html',
                               forecast_value=forecast_value,
                               closest_prediction_date=closest_prediction_date,
                               category=category,
                               user_input_date=input_date,
                               plot_url=f'/plot/{plot_file}')  # Point to the route serving the plot

    return render_template('index.html')

# Route to serve plot images from the static folder
@app.route('/plot/<filename>')
def plot_image(filename):
    plot_path = os.path.join('static', filename)
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return "Plot not found", 404

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

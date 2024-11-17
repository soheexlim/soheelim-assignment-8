from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from logistic_regression import do_experiments

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure 'index.html' is in the templates folder

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        # Get inputs from the request
        start = float(request.json['start'])
        end = float(request.json['end'])
        step_num = int(request.json['step_num'])

        # Run the experiment
        do_experiments(start, end, step_num)

        # Prepare paths to the result images
        dataset_img = "results/dataset.png"
        parameters_img = "results/parameters_vs_shift_distance.png"

        # Append timestamps to force the browser to reload images
        return jsonify({
            "dataset_img": f"{dataset_img}?t={os.path.getmtime(dataset_img)}",
            "parameters_img": f"{parameters_img}?t={os.path.getmtime(parameters_img)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)

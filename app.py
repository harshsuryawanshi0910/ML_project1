from flask import Flask, request, render_template
import pandas as pd
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pipeline.predict_pipeline import PredictPipeline, CustomData
    print("Successfully imported from src.pipeline.predict_pipeline")
except ImportError as e:
    print(f"Error importing: {e}")
    # Try alternative import
    try:
        from src.pipeline.predict_pipeline import PredictPipeline, CustomData
        print("Successfully imported from src.pipeline.predict_pipeline")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in src/pipeline: {os.listdir('src/pipeline') if os.path.exists('src/pipeline') else 'src/pipeline not found'}")
        sys.exit(1)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        # Get form data
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity')
        parental_education = request.form.get('parental_education')
        lunch = request.form.get('lunch')
        test_course = request.form.get('test_course')
        reading_score = int(request.form.get('reading_score'))
        writing_score = int(request.form.get('writing_score'))
        
        print(f"Received data: gender={gender}, race={race_ethnicity}")
        
        # Create CustomData object
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        # Get dataframe
        pred_df = data.get_data_as_data_frame()
        print("Input dataframe:", pred_df.to_dict())
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction results:", results)
        
        return render_template('home.html', results=round(float(results[0]), 2))
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
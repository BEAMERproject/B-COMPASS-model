from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import warnings
import joblib
import logging

model = joblib.load('models/BEAMER_pred_model_binary28_5varsAge.joblib')
scaler = joblib.load("models/min_max_scaler.pkl")

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

Acceptance_Cutoff = 4.666667
Control_Cutoff = 4.333333

Health_Consciousness_Cutoff_Segment_1 = 4.2
Health_Consciousness_Cutoff_Segment_2 = 3.8
Health_Consciousness_Cutoff_Segment_3 = 4
Health_Consciousness_Cutoff_Segment_4 = 3.6

Concern_Cutoff_Segment_1 = 3.64
Concern_Cutoff_Segment_2 = 3.64
Health_Priority_Cutoff_Segment_3 = 5.32
Health_Priority_Cutoff_Segment_4 = 5.04

#SEH Function
def get_SEH_segments(Acceptance, Control):
    segment_1 = 1 if (Acceptance > Acceptance_Cutoff and Control > Control_Cutoff) else 0
    segment_2 = 1 if (Acceptance > Acceptance_Cutoff and Control <= Control_Cutoff) else 0
    segment_3 = 1 if (Acceptance <= Acceptance_Cutoff and Control > Control_Cutoff) else 0
    segment_4 = 1 if (Acceptance <= Acceptance_Cutoff and Control <= Control_Cutoff) else 0
    
    segment_number = None
    if segment_1:
        segment_number = 1
    elif segment_2:
        segment_number = 2
    elif segment_3:
        segment_number = 3
    elif segment_4:
        segment_number = 4
    
    return segment_1, segment_2, segment_3, segment_4, segment_number

def get_Final_segments(Acceptance, Control, Health_Consciousness, Health_Priority, Concern, Age, segment_number, Prediction):
    if segment_number == 1:
        if Health_Consciousness > Health_Consciousness_Cutoff_Segment_1:
            if Concern > Concern_Cutoff_Segment_1:
                if Prediction == 1:
                    Final_Segment = "1AA"
                else:
                    Final_Segment = "1AM"
            else:
                if Prediction == 1:
                    Final_Segment = "1BA"
                else:
                    Final_Segment = "1BM"
        else:
            if Concern > Concern_Cutoff_Segment_1:
                if Prediction == 1:
                    Final_Segment = "1CA"
                else:
                    Final_Segment = "1CM"
            else:
                if Prediction == 1:
                    Final_Segment = "1DA"
                else:
                    Final_Segment = "1DM"

    elif segment_number == 2:
        if Health_Consciousness > Health_Consciousness_Cutoff_Segment_2:
            if Concern > Concern_Cutoff_Segment_2:
                if Prediction == 1:
                    Final_Segment = "2AA"
                else:
                    Final_Segment = "2AM"
            else:
                if Prediction == 1:
                    Final_Segment = "2BA"
                else:
                    Final_Segment = "2BM"
        else:
            if Concern > Concern_Cutoff_Segment_2:
                if Prediction == 1:
                    Final_Segment = "2CA"
                else:
                    Final_Segment = "2CM"
            else:
                if Prediction == 1:
                    Final_Segment = "2DA"
                else:
                    Final_Segment = "2DM"
    
    elif segment_number == 3:
        if Health_Consciousness > Health_Consciousness_Cutoff_Segment_3:
            if Health_Priority > Health_Priority_Cutoff_Segment_3:
                if Prediction == 1:
                    Final_Segment = "3AA"
                else:
                    Final_Segment = "3AM"
            else:
                if Prediction == 1:
                    Final_Segment = "3BA"
                else:
                    Final_Segment = "3BM"
        else:
            if Health_Priority > Health_Priority_Cutoff_Segment_3:
                if Prediction == 1:
                    Final_Segment = "3CA"
                else:
                    Final_Segment = "3CM"
            else:
                if Prediction == 1:
                    Final_Segment = "3DA"
                else:
                    Final_Segment = "3DM"

    elif segment_number == 4:
        if Health_Consciousness > Health_Consciousness_Cutoff_Segment_4:
            if Health_Priority > Health_Priority_Cutoff_Segment_4:
                if Prediction == 1:
                    Final_Segment = "4AA"
                else:
                    Final_Segment = "4AM"
            else:
                if Prediction == 1:
                    Final_Segment = "4BA"
                else:
                    Final_Segment = "4BM"
        else:
            if Health_Priority > Health_Priority_Cutoff_Segment_4:
                if Prediction == 1:
                    Final_Segment = "4CA"
                else:
                    Final_Segment = "4CM"
            else:
                if Prediction == 1:
                    Final_Segment = "4DA"
                else:
                    Final_Segment = "4DM"

    return Final_Segment

def get_Groups(Final_Segment):
    segment_to_group = {
        "1AA": "1", "1CA": "1", "1DA": "1", "1CM": "1",
        "2AA": "2", "2CA": "2", "2DA": "2", "2CM": "2",
        "3BA": "3", "3CA": "3", "3DA": "3", "3DM": "3",
        "4BA": "4", "4CA": "4", "4DA": "4", "4DM": "4",
        "1BA": "5", "1AM": "5", "1BM": "5", "1DM": "5",
        "2BA": "6", "2AM": "6", "2BM": "6", "2DM": "6",
        "3AA": "7", "3AM": "7", "3BM": "7", "3CM": "7",
        "4AA": "8", "4AM": "8", "4BM": "8", "4CM": "8"
    }

    return segment_to_group.get(Final_Segment)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/compassmodel', methods=['POST'])
def inference():

    cli_input = request.json
    app.logger.info("POST /compassmodel body: %s", cli_input)

    Acceptance = float(cli_input['Acceptance'])
    Control = float(cli_input['Control'])
    Health_Consciousness = float(cli_input['Health_Consciousness'])
    Health_Priority = float(cli_input['Health_Priority'])
    Concern = float(cli_input['Concern'])
    Age = int(cli_input['Age'])

    '''
    req_json = 
    {
        "Acceptance": 4.5,
        "Control": 6,
        "Health_Consciousness": 4.5,
        "Health_Priority": 6,
        "Concern": 6, 
        "Age": 56 
    }
    '''

    #1º SEH

    Segment_1, Segment_2, Segment_3, Segment_4, SEH_Segment = get_SEH_segments(Acceptance, Control)

    #2º ML Prediction

    input_df = pd.DataFrame({'Acceptance': [Acceptance],
                         'Age_c': [Age], 
                         'Perceived control': [Control], 
                         'Health consciousness': [Health_Consciousness], 
                         'Necessity': [Health_Priority],
                         'Concerns': [Concern] })

    scaled_input_df = pd.DataFrame(scaler.transform(input_df), index=input_df.index, columns=input_df.columns)

    scaled_values = scaled_input_df.to_dict('records')[0]

    model_df = pd.DataFrame({
    'Acceptance': [scaled_values['Acceptance']],
    'Age_c': [scaled_values['Age_c']],
    'Perceived control': [scaled_values['Perceived control']],
    'Health consciousness': [scaled_values['Health consciousness']],
    'Necessity': [scaled_values['Necessity']],
    'Concerns': [scaled_values['Concerns']],
    'Segment_1': [Segment_1],
    'Segment_2': [Segment_2],
    'Segment_3': [Segment_3],
    'Acceptance_Segment_1': [scaled_values['Acceptance'] * Segment_1],
    'Acceptance_Segment_2': [scaled_values['Acceptance'] * Segment_2],
    'Acceptance_Segment_3': [scaled_values['Acceptance'] * Segment_3],
    'Perceived control_Segment_1': [scaled_values['Perceived control'] * Segment_1],
    'Perceived control_Segment_2': [scaled_values['Perceived control'] * Segment_2],
    'Perceived control_Segment_3': [scaled_values['Perceived control'] * Segment_3],
    'Necessity_Segment_1': [scaled_values['Necessity'] * Segment_1],
    'Necessity_Segment_2': [scaled_values['Necessity'] * Segment_2],
    'Necessity_Segment_3': [scaled_values['Necessity'] * Segment_3],
    'Concerns_Segment_1': [scaled_values['Concerns'] * Segment_1],
    'Concerns_Segment_2': [scaled_values['Concerns'] * Segment_2],
    'Concerns_Segment_3': [scaled_values['Concerns'] * Segment_3],
    'Health consciousness_Segment_1': [scaled_values['Health consciousness'] * Segment_1],
    'Health consciousness_Segment_2': [scaled_values['Health consciousness'] * Segment_2],
    'Health consciousness_Segment_3': [scaled_values['Health consciousness'] * Segment_3],
    'Age_c_Segment_1': [scaled_values['Age_c'] * Segment_1],
    'Age_c_Segment_2': [scaled_values['Age_c'] * Segment_2],
    'Age_c_Segment_3': [scaled_values['Age_c'] * Segment_3]
    })

    prediction=model.predict(model_df)
    predicted_label = 1 if prediction > 0.5 else 0

    #3º Final Segmentation

    Final_Segment = get_Final_segments(Acceptance, Control, Health_Consciousness, Health_Priority, Concern, Age, SEH_Segment, predicted_label)
    
    #4º Groups

    Group = get_Groups(Final_Segment)
    
    response = {
        "1st-level": Final_Segment[0],
        "2nd-level": Final_Segment[1],
        "3rd-level": Final_Segment[2],
        "Final_Segment": Final_Segment,
        "Group": Group
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

model_path = "estabasmodel"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def interpret_label(predicted_label):
    if predicted_label == 0:
        return "Negative"
    elif predicted_label == 1:
        return "Positive"
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tokenized_text = tokenizer(data["text"], truncation=True, return_tensors="pt")
    outputs = model(**tokenized_text)
    predictions = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
    negative_percentage = predictions[0] * 100
    positive_percentage = predictions[1] * 100
    predicted_label = outputs.logits.argmax(-1)
    interpretation = interpret_label(predicted_label.item())
    output = {
        'prediction': interpretation,
        'negative_confidence': negative_percentage,
        'positive_confidence': positive_percentage
    }
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=False)

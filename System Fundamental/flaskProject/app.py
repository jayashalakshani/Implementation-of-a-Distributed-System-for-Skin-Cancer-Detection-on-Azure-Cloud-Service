from flask import Flask, render_template, request
from plot_pred import pred_and_plot_image
from model.modelClass import ResNet18
from torchvision import transforms
import torch
from pathlib import Path


app = Flask(__name__)


@app.route('/')
def home():

    return render_template('System_Fundamental.html', result=None)



@app.route('/predict', methods=['POST'])
def predict():
    model_path = Path("model/Model2.pth")

    # Define the class names for predictions
    class_names = ['Actinic keratosis',
                   'Basal cell carcinoma',
                   'Benign keratosis',
                   'Dermatofibroma',
                   'Melanocytic nevus',
                   'Melanoma',
                   'Squamous cell carcinoma',
                   'Vascular lesion']

    # Load your deep learning model
    model = ResNet18()  # Replace with the actual class of your model
    # model.load_state_dict(torch.load(f="/content/Model2.pth"))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Get the uploaded image file
    image = request.files['image']
    img_path = "static/" + image.filename
    image.save(img_path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # resize image to 224, 224 (height x width)
        transforms.ToTensor(),  # get images into range [0, 1]
        normalize])

    res_label, res_prob = pred_and_plot_image(class_names=class_names,
                                              model=model,
                                              image_path=img_path,
                                              transform=manual_transforms)


    return render_template('testinghtml.html', label=res_label, prob=res_prob.detach().numpy(), image_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)

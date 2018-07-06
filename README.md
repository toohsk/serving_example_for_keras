# Example codes for serving Keras model

This repository is example of serving Keras model.<br>
Trained Keras model's predict result will be returned via web api.
Web api is served by Flask.<br>

# Explain

`app.py` is a Flask app.<br>
Api will predict a number from image which is included in request and return the predict result.
Also `app.py` includes web app which has canvas, and you can write a number 0 to 9.
`static` and `templates` directories are for web app.

`train.py` is a training keras model script.
Training task is MNIST.
After training, parameter's are saved to model directory.

`load.py` is for loading trained model.
It will load model from `model` directory.

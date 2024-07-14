# MobileNet Deployment with Flask

This is an application that deploys a MobileNet model with a web server using Flask. 
It is designed to make predictions from user input, employs data normalization, and 
uses a pre-trained model that doesn't require fine-tuning.

## Installation

To install the dependencies, run:
```bash
pip install -r requirements.txt
```
To start the application, use the following command:

```bash
FLASK_APP=flask_deployment.py flask run
```
To make your own predictions and interact with the server, open a new terminal and run the following command:
```bash
curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@kitten.jpg"
```
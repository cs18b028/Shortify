# Shortify
Keywords-based Summarizer
<br>
<br>
Given a query, this tool gets relevant questions from stack overflow and generates summaries from a ranked list of answers for the retrieved questions, based on keywords along with one full crisp and informative summarized answer. For simplicity, we are restricting to queries in the python domain.

## System/Software requirements
The entire tool has been developed on windows 10 and has been implemented as a webapp (react+flask framework). Following are the important packages used in the webapp.

For flask backend
- python 3.8.5
- pip 21.0.1
For react frontend
- node v12.18.2
- npm 6.14.5

Jupyter Notebook(version 6.2.0) has been used for initial development.

- Running the code in Jupyter Notebook also needs you to install all the python dependencies and libraries. To install them navigate to the WebApp/backend directory in cmd (windows) or terminal (linux) and then execute ```pip install -r requirements.txt```

## Download and run the webapp on your local system
- Prequisites: technologies used and their versions are specified in [System/Software requirements](#systemsoftware-requirements). It is better to get those versions to avoid compatibility issues.
- Clone the repo or download the zip and unzip into a folder of you preference
- Navigate to Shortify-main directory in cmd or bash and execute the following commands in this directory
- To up the Flask backend (runs on localhost:5000)
```
cd WebApp/backend
pip install -r requirements.txt
flask run
```
- To start the React frontend (runs on localhost:3000)
```
cd WebApp/frontend
npm install
npm start
```

## The working app

## Implemenataion details

<p align="center"><img src="images/shortify.png" height="80%" width="80%"/></p>
<p align="center">Design of the tool</p>

## File guide

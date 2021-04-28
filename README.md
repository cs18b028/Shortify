# Shortify
Keywords-based Summarizer
<br>
<br>
Given a query, this tool gets relevant questions from stack overflow and generates summaries from a ranked list of answers for the retrieved questions, based on keywords along with one full crisp and informative summarized answer. For simplicity, we are restricting to queries in the python domain.

## Download and run it on your local system
<br>
- Prequisites: Node.js, npm, python and pip (latest versions) 
- Download the zip file and unzip into a folder of you preference
- Navigate to Shortify-main directory in cmd or bash and execute the following commands in this directory
- To up the Flask backend (runs on localhost:5000)
```
cd WebApp/backend
pip install -r > requirements.txt
flask run
```
- To start the React frontend (runs on localhost:3000)
```
cd WebApp/frontend
npm install
npm start
```

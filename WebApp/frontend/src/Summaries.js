// Summaries.js - displays the summaries, makes get req to the api route /summary in the backend

import React, { useState } from 'react';
import axios from 'axios';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import FormControl from 'react-bootstrap/FormControl';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Card from 'react-bootstrap/Card';
import Spinner from 'react-bootstrap/Spinner';
import ReactHtmlParser from 'react-html-parser';
import {NavLink} from 'react-router-dom'

function Summaries() {

  const [query, setQuery] = useState(''); // user query
  const [submitted, setSubmitted] = useState(false); // submitted check
  const [searchresults, setSearchResults] = useState([]); // results

  // fetchdata - function to make the get request and set the searchresults

  const fetchdata = () => {
    try {
      axios.get('http://127.0.0.1:5000/summary', {
        params: {
          "query": query
        }
      })
        .then((response) => {
          setSearchResults(response.data.results);
          setSubmitted(false);
        });
    }
    catch (e) {
      setSubmitted(false);
      console.log('Error: ' + e)
    }
  }

  // handle form submission

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
    setSearchResults([])
    fetchdata();
  }

  // results components

  var renderResults = searchresults.map((result, i) => {
    return (
      <div style={{padding: '10px 40px'}}>
          <h4>{result.topic}</h4>
          <p align="justify">{ReactHtmlParser(result.summary)}</p>
      </div>
    );
  });

  return (
    <div>
      <Navbar bg="light" variant="light" style={{ margin: '5vh 10vw 0 10vw', borderRadius: '1rem' }}>
        <Navbar.Brand style={{ margin: '0 2vw 0 3vw' }}><h3>Shortify</h3></Navbar.Brand>
        <Nav style={{ marginRight: '3vw' }}>
          <Nav.Link><NavLink to="/questions">Questions</NavLink></Nav.Link>
          <Nav.Link><NavLink to="/summaries"><b>Summaries</b></NavLink></Nav.Link>
        </Nav>
        <Form inline onSubmit={handleSubmit}>
          <FormControl type="text" autoFocus placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: '40vw', marginRight: '1vw'  }} />
          <Button variant="outline-info" type="submit">Search</Button>
        </Form>
      </Navbar>

      <div style={{ margin: '5vh 10vw 5vh 10vw', padding: '0 30px'}}>
        <h6>What happens here?</h6>
        <ul>
          <li>Stackoverflow will be searched for questions most relevant to your query</li>
          <li>Those questions are categorized into different topics indicated by the keywords in those questions</li>
          <li>Answers to the questions under each topic are extracted and summarized</li>
        </ul>
        <h6>How to use?</h6>
        <ul>
          <li>Summaries under each topic contain sentences from answers to the questions that are most relevent to your query from stackoverflow</li>
          <li>Each topic is characterized by its keywords. The keywords tell the context of the answers used to build the summaries </li>
          <li>Choose which summary to read depending on the context you want</li>
          <li>If a sentence in a summary makes sense to you hover your mouse over it to see the related stackoverflow post</li>
          <li>Clicking on the sentence will open the stackoverflow post in a newtab</li>
        </ul>
      </div>

      <Card style={{ margin: '5vh 10vw 5vh 10vw', borderRadius: '1rem'}}>
        {(!submitted && renderResults!==0)? renderResults : 
          <div style={{padding:'20px', margin: 'auto'}}>
            <p align="center"><Spinner animation="border" /></p>
            <p align="center" style={{fontWeight: 'lighter'}}>Shouldn't take more than 30 seconds</p>
          </div>}
      </Card>

    </div>
  );
}

export default Summaries;
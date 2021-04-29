// Questions.js - displays the relevant questions, makes get req to the api route /question in the backend

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
import { NavLink } from "react-router-dom";

function Questions() {

  const [query, setQuery] = useState(''); // user query
  const [submitted, setSubmitted] = useState(false); // submitted check
  const [searchresults, setSearchResults] = useState([]); // results

  // fetchdata - function to make the get request and set the searchresults

  const fetchdata = () => {
    try {
      axios.get('http://127.0.0.1:5000/question', {
        params: {
          "query": query
        }
      })
        .then((response) => {
          setSearchResults(response.data.results);
          setSubmitted(false)
        });
    }
    catch (e) {
      setSubmitted(false)
      console.log('Error: ' + e)
    }
  }

  // handle form submission

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true)
    setSearchResults([])
    fetchdata();
  }

  // results components
  
  var renderResults = searchresults.map((result, i) => {
    return (
      <div  style={{padding: '10px'}}>
        <div className="col-12 col-md-10">
          <h4>{ReactHtmlParser(result.question)}</h4>
          <p>score: {result.score} out of 10</p>
        </div>
      </div>
    );
  });

  return (
    <div>
      <Navbar bg="light" variant="light" style={{ margin: '5vh 10vw 0 10vw', borderRadius: '1rem' }}>
        <Navbar.Brand style={{ margin: '0 2vw 0 3vw' }}><h3>Shortify</h3></Navbar.Brand>
        <Nav style={{ marginRight: '3vw' }}>
          <Nav.Link><NavLink to="/questions"><b>Questions</b></NavLink></Nav.Link>
          <Nav.Link><NavLink to="/summaries">Summaries</NavLink></Nav.Link>
        </Nav>
        <Form inline onSubmit={handleSubmit}>
          <FormControl type="text" autoFocus placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: '40vw', marginRight: '1vw' }} />
          <Button variant="outline-info" type="submit">Search</Button>
        </Form>
      </Navbar>
      <div style={{ margin: '5vh 10vw 5vh 10vw'}}>
        <h6 align='center'>Top 20 stackoverflow questions that are related to your query will be displayed here (with links). For summaries click the link in the navbar</h6>
      </div>
      <Card style={{ margin: '5vh 10vw 5vh 10vw', borderRadius: '1rem' }}>
        {(!submitted && renderResults!==0)? renderResults :
          <div style={{padding:'20px', margin: 'auto'}}>
            <p align="center"><Spinner animation="border" /></p>
            <p align="center" style={{fontWeight: 'lighter'}}>Shouldn't take more than 15 seconds</p>
          </div>}
      </Card>
    </div>
  );
}

export default Questions;
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

  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [searchresults, setSearchResults] = useState([]);

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

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true)
    setSearchResults([])
    fetchdata();
  }

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
          <Nav.Link><NavLink to="/questions">Questions</NavLink></Nav.Link>
          <Nav.Link><NavLink to="/summaries">Summaries</NavLink></Nav.Link>
        </Nav>
        <Form inline onSubmit={handleSubmit}>
          <FormControl type="text" placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: '40vw', marginRight: '1vw' }} />
          <Button variant="outline-info" type="submit">Search</Button>
        </Form>
      </Navbar>
      <Card style={{ margin: '5vh 10vw 5vh 10vw', borderRadius: '1rem' }}>
        {(!submitted && renderResults!==0)? renderResults : <div style={{padding:'20px', margin: 'auto'}}><Spinner animation="border" /></div>}
      </Card>
    </div>
  );
}

export default Questions;
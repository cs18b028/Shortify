import React, { useState } from 'react';
import axios from 'axios';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import FormControl from 'react-bootstrap/FormControl';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import Card from 'react-bootstrap/Card';
import ReactHtmlParser from 'react-html-parser';
import {NavLink} from 'react-router-dom'

function Summaries() {

  const [query, setQuery] = useState('');
  const [searchresults, setSearchResults] = useState([]);

  const fetchdata = () => {
    try {
      axios.get('http://127.0.0.1:5000/summary', {
        params: {
          "query": query
        }
      })
        .then((response) => {
          setSearchResults(response.data.results);
        });
    }
    catch (e) {
      console.log('Error: ' + e)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    setSearchResults([])
    fetchdata();
  }

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
          <Nav.Link><NavLink to="/summaries">Summaries</NavLink></Nav.Link>
        </Nav>
        <Form inline onSubmit={handleSubmit}>
          <FormControl type="text" placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: '40vw', marginRight: '1vw'  }} />
          <Button variant="outline-info" type="submit">Search</Button>
        </Form>
      </Navbar>
      <Card style={{ margin: '5vh 10vw 5vh 10vw', borderRadius: '1rem'}}>
        {renderResults}
      </Card>
    </div>
  );
}

export default Summaries;
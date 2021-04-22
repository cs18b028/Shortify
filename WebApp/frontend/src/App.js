import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import FormControl from 'react-bootstrap/FormControl';
import Navbar from 'react-bootstrap/Navbar';
import Card from 'react-bootstrap/Card';

function App() {

  const [query, setQuery] = useState('');
  const [searchresults, setSearchResults] = useState([]);

  const fetchdata = () => {
    try {
      axios.get('http://127.0.0.1:5000/api', {
        params: {
          "query": query
        }
      })
        .then((response) => {
          setSearchResults(response.data.results)
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
      <div key={result.index}>
        <div className="col-12 col-md-10">
          <h1>{result.title}</h1>
          <h4>Similarity Score: {result.similarity_score}</h4>
        </div>
      </div>
    );
  });

  return (
    <div>
        <Navbar bg="light" variant="light" style={{ margin: '5vh 10vw 0 10vw', borderRadius: '1rem' }}>
          <Navbar.Brand style={{ marginLeft: '1vw' }}><h3>Shortify</h3></Navbar.Brand>
          <Form inline onSubmit={handleSubmit}>
            <FormControl type="text" placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: '60vw', margin: '0 1vw' }}/>
            <Button variant="outline-info">Search</Button>
          </Form>
        </Navbar>
        <Card style={{ margin: '5vh 10vw 0 10vw', borderRadius: '1rem' }}>
          {renderResults}
        </Card>
    </div>
  );
}

export default App;
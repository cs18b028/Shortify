import React, { useState} from 'react';
import axios from 'axios';
import './App.css';
import { Card } from 'react-bootstrap';

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
      <div>
        <div className="col-12 col-md-10">
          <h1>{result.title}</h1>
          <h4>Similarity Score: {result.similarity_score}</h4>
        </div>
      </div>
    );
  });

  return (
    <div>
      <div className="container">
        <Card>
          <Card.Body>
            <form className="form-group" onSubmit={handleSubmit}>
              <input type="text" className="form-control" value={query} placeholder="Search Query" onChange={(e) => setQuery(e.target.value)} />
            </form>
            {renderResults}
          </Card.Body>
        </Card>
      </div>
    </div>
  );
}

export default App;
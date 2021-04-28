// App.js

import React from 'react';
import { BrowserRouter, Route, Switch, Redirect } from "react-router-dom";
import Summaries from "./Summaries";
import Questions from "./Questions";

// App Component

function App() {
  return (
    <div>
      <BrowserRouter>
          <Switch>
            <Route path="/questions" component={Questions} />
            <Route path="/summaries" component={Summaries} />
            <Route path="/"><Redirect to="/questions"/></Route>
          </Switch>
      </BrowserRouter>
    </div>
  )
}

export default App;
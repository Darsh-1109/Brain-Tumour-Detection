import "./App.css";
import Home from "./components/Home";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React from "react";

function App() {
  return (
    <div className="App">
      <header className="App-header">
      <Router>
          <Routes>
            <Route path="/" element={<Home />} />
          </Routes>
        </Router>
      </header>
    </div>
  );
}

export default App;

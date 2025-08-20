// 📄 src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css"; // 없다면 생략 가능

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);

import React, { useState } from "react";
import axios from "axios";

function App() {
  const [model, setModel] = useState("unet");
  const [epoch, setEpoch] = useState(5);
  const [prediction, setPrediction] = useState(null);

  const fetchPrediction = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/compare/${model}/${epoch}`);
      setPrediction(`data:image/png;base64,${response.data.prediction}`);
    } catch (error) {
      console.error("Error fetching predictions", error);
    }
  };

  return (
    <div>
      <h1>Segmentation Model Comparison</h1>

      <label>Select Model:</label>
      <select value={model} onChange={(e) => setModel(e.target.value)}>
        <option value="unet">UNet</option>
        <option value="deeplabv3">DeepLabV3</option>
        <option value="pspnet">PSPNet</option>
      </select>

      <label>Select Epoch:</label>
      <select value={epoch} onChange={(e) => setEpoch(e.target.value)}>
        <option value={5}>Epoch 5</option>
        <option value={10}>Epoch 10</option>
        <option value={20}>Epoch 20</option>
        <option value={50}>Epoch 50</option>
      </select>

      <button onClick={fetchPrediction}>Compare Prediction</button>

      {prediction && <img src={prediction} alt="Prediction Comparison" />}
    </div>
  );
}

export default App;
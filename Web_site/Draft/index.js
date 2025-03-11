// 🟢 Text-based diagnosis prediction
document.querySelector('.sub-btn').addEventListener('click', async () => {
    const symptoms = {
        "symptoms": [
            document.getElementById('symptom1').checked ? 1 : 0,
            document.getElementById('symptom2').checked ? 1 : 0,
            document.getElementById('symptom3').checked ? 1 : 0,
            document.getElementById('symptom4').checked ? 1 : 0
        ]
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/predict-text/', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(symptoms)
        });

        const data = await response.json();
        alert(`Diagnosis: ${data.risk_percentage.toFixed(2)}% risk of breast cancer`);
    } catch (error) {
        console.error("Error fetching text prediction:", error);
    }
});

// 🟢 Image-based diagnosis prediction
document.getElementById('uploadBtn').addEventListener('change', async (event) => {
    let formData = new FormData();
    formData.append("file", event.target.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict-image/', { 
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        alert(`Image Diagnosis: ${data.prediction}`);
    } catch (error) {
        console.error("Error fetching image prediction:", error);
    }
});

// 🟢 Chatbot function
document.getElementById('chatBtn').addEventListener('click', async () => {
    const message = document.getElementById('chatInput').value.trim(); 

    if (message === "") {
        alert("Please enter a message!");
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/chatbot/?user_input=' + encodeURIComponent(message), { 
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        // Display chatbot response in chat UI
        const chatResponseElement = document.getElementById('chatResponse');
        chatResponseElement.innerText = data.response;

        // Add response to list
        const listItem = document.createElement("li");
        listItem.textContent = data.response;
        document.querySelector(".answer").appendChild(listItem);

    } catch (error) {
        console.error("Error fetching chatbot response:", error);
    }
});

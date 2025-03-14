document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("submitBtn").addEventListener("click", function (event) {
        event.preventDefault(); // Prevent form from reloading the page

        let age = document.getElementById("age").value;
        let symptoms = [];

        // Collect all 9 symptoms from checkboxes
        document.querySelectorAll(".symptom-checkbox").forEach((checkbox) => {
            symptoms.push(checkbox.checked ? 1.0 : 0.0);
        });

        if (symptoms.length !== 8) {
            alert(`Error: Expected 9 symptoms, but found ${symptoms.length}`);
            return;
        }

        if (!age || isNaN(parseFloat(age))) {
            alert("Please enter a valid age.");
            return;
        }

        // ✅ Send age as the first feature, then symptoms
        const data = {
            age: parseFloat(age),
            symptoms: symptoms
        };

        console.log("📤 Sending Data:", JSON.stringify(data));

        fetch("http://127.0.0.1:8000/predict/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            console.log("✅ API Response:", result);
            if (result.diagnosis && result.confidence) {
                localStorage.setItem("predictionResult", JSON.stringify(result));
                window.location.href = "3rd_page.html"; // Redirect to results page
            } else {
                alert("Failed to get a valid prediction.");
            }
        })
        .catch(error => console.error("❌ Error:", error));
    });
});

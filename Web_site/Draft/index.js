document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('symptomForm');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const age = document.getElementById('age').value;
        const symptoms = [];
        document.querySelectorAll('.symptom-checkbox:checked').forEach(function(checkbox) {
            symptoms.push(checkbox.id);
        });

        // Simulate prediction data (replace with your actual prediction logic)
        const predictionResult = {
            diagnosis: Math.random() > 0.5 ? "Possible Breast Cancer" : "Low Risk",
            confidence: (Math.random() * 100).toFixed(2) + "%"
        };

        // Store prediction data in localStorage
        localStorage.setItem('predictionResult', JSON.stringify(predictionResult));

        // Redirect to the results page
        window.location.href = '/Web_site/Draft/3rd_page.html';
    });
});
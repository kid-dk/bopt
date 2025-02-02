// Prefill data
const initialData = [
    [0.0, 0.0, 0.0, 430.700000, 0.040480],
    [0.5, 150.0, 0.0, 252.821507, 0.846667],
    [1.0, 150.0, 1.5, 76.741028, 0.930000],
    [0.0, 75.0, 0.0, 352.021355, 0.660667]
];

// Timer variables
let startTime;
let timerInterval;

// Load saved data from localStorage
function loadSavedData() {
    const savedData = localStorage.getItem('gprData');
    if (savedData) {
        return JSON.parse(savedData);
    }
    return initialData;
}

// Save data to localStorage
function saveData(data) {
    localStorage.setItem('gprData', JSON.stringify(data));
}

// Create the data table
function createDataTable() {
    const numPoints = document.getElementById('initial-data-points').value;
    if (numPoints <= 0) {
        alert('Please enter a valid number of initial data points.');
        return;
    }

    const tableSection = document.getElementById('data-table-section');
    tableSection.innerHTML = '';

    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    const headers = ["PEG (g)", "NaCl (mg)", "Glycerol (ml)", "ESD", "RF"];
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    const savedData = loadSavedData();

    for (let i = 0; i < numPoints; i++) {
        const tr = document.createElement('tr');
        headers.forEach((header, j) => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `data-${i}-${j}`;
            input.placeholder = header;
            if (savedData[i]) {
                input.value = savedData[i][j]; // Prefill with saved data
            }
            td.appendChild(input);
            tr.appendChild(td);
        });
        table.appendChild(tr);
    }

    tableSection.appendChild(table);
    document.getElementById('calculate-btn').style.display = 'block';
}

// Start the timer
function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 100);
}

// Stop the timer and calculate elapsed time
function stopTimer() {
    clearInterval(timerInterval);
    const elapsedTime = (Date.now() - startTime) / 1000;
    document.getElementById('elapsed-time').textContent = elapsedTime.toFixed(2);
}

// Update the timer display
function updateTimer() {
    const elapsedTime = (Date.now() - startTime) / 1000;
    document.getElementById('elapsed-time').textContent = elapsedTime.toFixed(2);
}

// Calculate function
async function calculate() {
    const numPoints = document.getElementById('initial-data-points').value;
    const batchSize = document.getElementById('batch-size').value;
    const initialData = [];

    for (let i = 0; i < numPoints; i++) {
        const row = [];
        for (let j = 0; j < 5; j++) {
            const value = parseFloat(document.getElementById(`data-${i}-${j}`).value);
            if (isNaN(value)) {
                alert(`Please enter a valid number for row ${i + 1}, column ${j + 1}.`);
                return;
            }
            row.push(value);
        }
        initialData.push(row);
    }

    // Save the data to localStorage
    saveData(initialData);

    // Show loading animation and start timer
    const loadingDiv = document.getElementById('loading');
    const calculateBtn = document.getElementById('calculate-btn');
    loadingDiv.style.display = 'block';
    calculateBtn.disabled = true;
    startTimer();

    try {
        const response = await fetch('https://bopt.onrender.com/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                initial_data: initialData,
                batch_size: parseInt(batchSize),
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        displayResult(data.result);
        document.getElementById('average-time').textContent = data.avg_response_time.toFixed(2);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
    } finally {
        // Hide loading animation and stop timer
        loadingDiv.style.display = 'none';
        calculateBtn.disabled = false;
        stopTimer();
    }
}

// Display the result
function displayResult(result) {
    const resultTable = document.getElementById('result-table');
    resultTable.innerHTML = '';

    const headerRow = document.createElement('tr');
    const headers = ["PEG (g)", "NaCl (mg)", "Glycerol (ml)"];
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    resultTable.appendChild(headerRow);

    result.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header].toFixed(2);
            tr.appendChild(td);
        });
        resultTable.appendChild(tr);
    });
}

// Fetch the average response time when the page loads
async function fetchAverageResponseTime() {
    try {
        const response = await fetch('https://bopt.onrender.com/average-response-time');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('average-time').textContent = data.avg_response_time.toFixed(2);
        }
    } catch (error) {
        console.error('Error fetching average response time:', error);
    }
}

// Initialize the table with pre-filled data and fetch average response time
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('initial-data-points').value = 4;
    createDataTable();
    fetchAverageResponseTime();  // Fetch average response time on page load
});
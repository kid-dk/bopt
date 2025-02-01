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

    for (let i = 0; i < numPoints; i++) {
        const tr = document.createElement('tr');
        headers.forEach((header, j) => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `data-${i}-${j}`;
            input.placeholder = header;
            td.appendChild(input);
            tr.appendChild(td);
        });
        table.appendChild(tr);
    }

    tableSection.appendChild(table);
    document.getElementById('calculate-btn').style.display = 'block';
}

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

    try {
        const response = await fetch('http://127.0.0.1:5000/calculate', {
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
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
    }
}

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
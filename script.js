document.getElementById("createTableBtn").addEventListener("click", function() {
    const numPoints = parseInt(document.getElementById("numPoints").value);
    if (isNaN(numPoints) || numPoints <= 0) {
        alert("Enter a valid number.");
        return;
    }
    createDataTable(numPoints);
});

function createDataTable(numPoints) {
    const container = document.getElementById("dataTableContainer");
    container.innerHTML = "";

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["PEG (g)", "NaCl (mg)", "Glycerol (ml)"].forEach(headerText => {
        const th = document.createElement("th");
        th.innerText = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (let i = 0; i < numPoints; i++) {
        const row = document.createElement("tr");
        for (let j = 0; j < 3; j++) {
            const cell = document.createElement("td");
            const input = document.createElement("input");
            input.type = "number";
            input.style.width = "80px";
            cell.appendChild(input);
            row.appendChild(cell);
        }
        tbody.appendChild(row);
    }
    table.appendChild(tbody);
    container.appendChild(table);
}

document.getElementById("generateBtn").addEventListener("click", function() {
    const batchSize = parseInt(document.getElementById("batchSize").value);
    if (isNaN(batchSize) || batchSize <= 0) {
        alert("Enter a valid batch size.");
        return;
    }

    fetch("https://bopt.onrender.com", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_size: batchSize })
    })
    .then(response => response.json())
    .then(data => displayOutputTable(data))
    .catch(error => console.error("Error:", error));
});

function displayOutputTable(candidates) {
    const container = document.getElementById("outputTableContainer");
    container.innerHTML = "";
    const table = document.createElement("table");

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["PEG (g)", "NaCl (mg)", "Glycerol (ml)"].forEach(headerText => {
        const th = document.createElement("th");
        th.innerText = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    candidates.forEach(candidate => {
        const row = document.createElement("tr");
        Object.values(candidate).forEach(value => {
            const cell = document.createElement("td");
            cell.innerText = value.toFixed(4);
            row.appendChild(cell);
        });
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    container.appendChild(table);
}

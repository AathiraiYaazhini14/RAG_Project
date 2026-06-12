const API_URL = "http://127.0.0.1:8000";
let currentDocId = null;

// Show selected filename
document.getElementById("fileInput").addEventListener("change", function() {
    const fileName = this.files[0]?.name || "No file chosen";
    document.getElementById("fileName").textContent = fileName;
});

// Upload file
async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const status = document.getElementById("uploadStatus");
    const btn = document.querySelector(".upload-section .primary-btn");

    if (!fileInput.files[0]) {
        status.innerHTML = '<span class="error">Please choose a file first.</span>';
        return;
    }

    // Show processing animation
    btn.disabled = true;
    btn.textContent = "Processing...";
    status.innerHTML = `
        <div class="thinking">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
            Processing document...
        </div>
    `;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
    currentDocId = data.doc_id;
    status.innerHTML = `<span class="success">Document processed — ${data.chunks_created} chunks created from ${data.filename}</span>`;
} else {
    const errorMsg = data.detail ? JSON.stringify(data.detail) : "Unknown error";
    status.innerHTML = `<span class="error">Error: ${errorMsg}</span>`;
            status.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
        }
    } catch (err) {
        status.innerHTML = '<span class="error">Could not connect to API. Is the backend running?</span>';
    } finally {
        btn.disabled = false;
        btn.textContent = "Upload & Process";
    }
}

// Ask question
async function askQuestion() {
    const question = document.getElementById("questionInput").value.trim();
    const answerBox = document.getElementById("answerBox");
    const btn = document.querySelector(".ask-section .primary-btn");

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    // Show thinking animation
    btn.disabled = true;
    btn.textContent = "Thinking...";
    answerBox.style.display = "block";
    answerBox.innerHTML = `
        <div class="thinking">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
            Searching documents and generating answer...
        </div>
    `;

    try {
        const searchScope = document.querySelector('input[name="searchScope"]:checked').value;
        const doc_id = searchScope === "current" ? currentDocId : null;
        const response = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question, top_k: 5, doc_id: doc_id })
            
        });

        const data = await response.json();

        if (response.ok) {
            let sourcesHTML = data.sources.map(source => `
                <div class="source-item">
                    <span class="filename">${source.filename}</span>
                    <span class="score">Score: ${source.score}</span>
                    <p class="excerpt">${source.excerpt}</p>
                </div>
            `).join("");

            answerBox.innerHTML = `
                <p class="answer-text">${data.answer}</p>
                <p class="sources-title">Sources</p>
                ${sourcesHTML}
            `;
        } else {
answerBox.innerHTML = `<p class="error">Error: ${JSON.stringify(data.detail)}</p>`;        }
    } catch (err) {
        answerBox.innerHTML = '<p class="error">Could not connect to API. Is the backend running?</p>';
    } finally {
        btn.disabled = false;
        btn.textContent = "Ask";
    }
}
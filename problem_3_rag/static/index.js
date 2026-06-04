document.addEventListener("DOMContentLoaded", () => {
    // --- Navigation ---
    const navRagBtn = document.getElementById("nav-rag-btn");
    const navLogsBtn = document.getElementById("nav-logs-btn");
    const viewRag = document.getElementById("view-rag");
    const viewLogs = document.getElementById("view-logs");
    const pageTitle = document.getElementById("page-title");
    const pageDesc = document.getElementById("page-desc");

    navRagBtn.addEventListener("click", () => {
        setActiveView("rag");
    });

    navLogsBtn.addEventListener("click", () => {
        setActiveView("logs");
    });

    function setActiveView(view) {
        if (view === "rag") {
            navRagBtn.classList.add("active");
            navLogsBtn.classList.remove("active");
            viewRag.classList.add("active");
            viewLogs.classList.remove("active");
            pageTitle.textContent = "Enterprise HR Policy Assistant";
            pageDesc.textContent = "Retrieval-Augmented Generation using local ChromaDB and Gemini API.";
        } else {
            navLogsBtn.classList.add("active");
            navRagBtn.classList.remove("active");
            viewLogs.classList.add("active");
            viewRag.classList.remove("active");
            pageTitle.textContent = "Transaction Log Analyzer";
            pageDesc.textContent = "High-performance streaming log parsing running in O(1) space complexity.";
        }
    }

    // --- Problem 3: RAG Assistant Logic ---
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    // Click handler for suggested queries
    document.querySelectorAll(".suggest-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            chatInput.value = query;
            submitRAGQuery(query);
        });
    });

    sendBtn.addEventListener("click", () => {
        const query = chatInput.value.trim();
        if (query) {
            submitRAGQuery(query);
        }
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const query = chatInput.value.trim();
            if (query) {
                submitRAGQuery(query);
            }
        }
    });

    async function submitRAGQuery(question) {
        // Clear input
        chatInput.value = "";
        
        // 1. Add User Message
        appendMessage("user", question);
        
        // 2. Add AI Typing Loader
        const loaderId = appendTypingLoader();
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // Send request to RAG backend
            const response = await fetch("/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                const errJson = await response.json();
                throw new Error(errJson.detail || "RAG pipeline failed.");
            }

            const data = await response.json();
            
            // Remove typing loader
            removeTypingLoader(loaderId);

            // 3. Add AI Answer with source documents
            appendMessage("ai", data.answer, data.sources);
            
        } catch (error) {
            removeTypingLoader(loaderId);
            appendMessage("ai", `⚠️ Error: ${error.message}. Please verify the RAG backend server logs.`);
        }
        
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendMessage(sender, text, sources = []) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender === "user" ? "user-msg" : "ai-msg");

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("msg-avatar");
        avatarDiv.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("msg-bubble");
        
        // Paragraph structure
        const p = document.createElement("p");
        // Simple markdown replacement for bolding **text**
        p.innerHTML = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        bubbleDiv.appendChild(p);

        // Append sources if available
        if (sources && sources.length > 0) {
            const uniqueId = `sources-${Math.random().toString(36).substr(2, 9)}`;
            
            const toggleBtn = document.createElement("button");
            toggleBtn.classList.add("sources-toggle-btn");
            toggleBtn.innerHTML = '<i class="fa-solid fa-folder-open"></i> Show retrieved policies';
            toggleBtn.addEventListener("click", () => {
                const container = document.getElementById(uniqueId);
                if (container.style.display === "block") {
                    container.style.display = "none";
                    toggleBtn.innerHTML = '<i class="fa-solid fa-folder-open"></i> Show retrieved policies';
                } else {
                    container.style.display = "block";
                    toggleBtn.innerHTML = '<i class="fa-solid fa-folder-closed"></i> Hide retrieved policies';
                }
            });
            
            const sourcesContainer = document.createElement("div");
            sourcesContainer.id = uniqueId;
            sourcesContainer.classList.add("sources-container");
            
            sources.forEach((source, index) => {
                const itemDiv = document.createElement("div");
                itemDiv.classList.add("source-item");
                
                const headerDiv = document.createElement("div");
                headerDiv.classList.add("source-header");
                const sourceName = source.metadata.source ? source.metadata.source.split(/[\\/]/).pop() : "hr_policy.txt";
                headerDiv.innerHTML = `<span>[Chunk ${index + 1}] Source: ${sourceName}</span>`;
                
                const contentDiv = document.createElement("div");
                contentDiv.classList.add("source-content");
                contentDiv.textContent = `"${source.page_content}"`;
                
                itemDiv.appendChild(headerDiv);
                itemDiv.appendChild(contentDiv);
                sourcesContainer.appendChild(itemDiv);
            });
            
            bubbleDiv.appendChild(toggleBtn);
            bubbleDiv.appendChild(sourcesContainer);
        }

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
    }

    function appendTypingLoader() {
        const loaderId = `loader-${Date.now()}`;
        const loaderDiv = document.createElement("div");
        loaderDiv.id = loaderId;
        loaderDiv.classList.add("message", "ai-msg");

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("msg-avatar");
        avatarDiv.innerHTML = '<i class="fa-solid fa-robot"></i>';

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("msg-bubble");
        
        const spinner = document.createElement("div");
        spinner.classList.add("spinner-container");
        spinner.innerHTML = '<div class="spinner-small"></div> <span>Searching ChromaDB & synthesizing response...</span>';
        
        bubbleDiv.appendChild(spinner);
        loaderDiv.appendChild(avatarDiv);
        loaderDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(loaderDiv);
        
        return loaderId;
    }

    function removeTypingLoader(id) {
        const loader = document.getElementById(id);
        if (loader) {
            loader.remove();
        }
    }

    // --- Problem 2: Log Analyzer Drag & Drop Upload ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const processLoader = document.getElementById("process-loader");
    const loaderStatus = document.getElementById("loader-status");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const analyzerResults = document.getElementById("analyzer-results");
    const flaggedCount = document.getElementById("flagged-count");
    const flaggedTableBody = document.getElementById("flagged-table-body");
    const downloadCsvBtn = document.getElementById("download-csv-btn");

    let currentFlaggedLogs = []; // Stores the current list of flagged logs for CSV export

    // Drag-over styling
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    // Handle file drop
    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processLogFile(files[0]);
        }
    });

    // Handle file select
    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            processLogFile(fileInput.files[0]);
        }
    });

    async function processLogFile(file) {
        // Reset UI
        dropZone.classList.add("hidden");
        processLoader.classList.remove("hidden");
        analyzerResults.classList.add("hidden");
        progressBarFill.style.width = "10%";
        loaderStatus.textContent = `Uploading ${file.name}...`;

        // Progress bar simulation
        let progressInterval = setInterval(() => {
            let width = parseInt(progressBarFill.style.width);
            if (width < 85) {
                progressBarFill.style.width = (width + 5) + "%";
            }
        }, 300);

        try {
            const formData = new FormData();
            formData.append("file", file);

            // Connects to the Problem 2 FastAPI server running on port 8000
            // Dynamic host enables running inside Docker/Virtual Machines
            const host = window.location.hostname;
            const uploadUrl = `http://${host}:8000/analyze-logs`;

            const response = await fetch(uploadUrl, {
                method: "POST",
                body: formData
            });

            clearInterval(progressInterval);
            progressBarFill.style.width = "100%";

            if (!response.ok) {
                const errDetail = await response.text();
                throw new Error(errDetail || "Failed to analyze logs.");
            }

            const data = await response.json();
            currentFlaggedLogs = data.flagged_users;

            // Update loader
            setTimeout(() => {
                processLoader.classList.add("hidden");
                dropZone.classList.remove("hidden");
                renderFlaggedResults(data.flagged_users);
            }, 500);

        } catch (error) {
            clearInterval(progressInterval);
            processLoader.classList.add("hidden");
            dropZone.classList.remove("hidden");
            alert(`Error processing log file: ${error.message}\nMake sure the FastAPI backend (port 8000) is running.`);
        }
    }

    function renderFlaggedResults(users) {
        flaggedTableBody.innerHTML = "";
        flaggedCount.textContent = `${users.length} Flagged`;

        if (users.length === 0) {
            const row = document.createElement("tr");
            row.innerHTML = '<td colspan="4" style="text-align: center; color: var(--text-muted);">No suspicious transactions (> $10,000) found in logs.</td>';
            flaggedTableBody.appendChild(row);
        } else {
            users.forEach(item => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${item.line_num}</td>
                    <td style="font-family: monospace; font-weight: 500; color: var(--accent-teal);">${item.user}</td>
                    <td style="font-weight: 600;">$${item.amount.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                    <td><span class="badge-flagged">Flagged</span></td>
                `;
                flaggedTableBody.appendChild(row);
            });
        }

        analyzerResults.classList.remove("hidden");
    }

    // Export CSV logic
    downloadCsvBtn.addEventListener("click", () => {
        if (currentFlaggedLogs.length === 0) return;
        
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "Line Number,User ID,Amount,Status\n";
        
        currentFlaggedLogs.forEach(row => {
            csvContent += `${row.line_num},"${row.user}",${row.amount},${row.status}\n`;
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "flagged_transactions.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});

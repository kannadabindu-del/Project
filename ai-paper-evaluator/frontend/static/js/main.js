// File upload display
document.getElementById("file")?.addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name;
    document.getElementById("fileName").textContent = fileName || "";
});

// Form submission loading
document.getElementById("uploadForm")?.addEventListener("submit", function () {
    document.getElementById("submitBtn").disabled = true;
    document.getElementById("submitBtn").textContent = "⏳ Evaluating...";
    document.getElementById("loading").style.display = "block";
});

// Drag and drop
const dropZone = document.getElementById("dropZone");
if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#3b82f6";
        dropZone.style.background = "rgba(59, 130, 246, 0.1)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "#475569";
        dropZone.style.background = "transparent";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "#475569";
        dropZone.style.background = "transparent";
        const fileInput = document.getElementById("file");
        fileInput.files = e.dataTransfer.files;
        document.getElementById("fileName").textContent = e.dataTransfer.files[0]?.name || "";
    });
}
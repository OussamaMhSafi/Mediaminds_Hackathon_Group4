const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const classification = document.getElementById("classification");
const confidence = document.getElementById("confidence");
const sourceList = document.getElementById("sourceList");

["dragenter", "dragover"].forEach(event =>
  dropzone.addEventListener(event, e => {
    e.preventDefault();
    dropzone.classList.add("hover");
  })
);

["dragleave", "drop"].forEach(event =>
  dropzone.addEventListener(event, e => {
    e.preventDefault();
    dropzone.classList.remove("hover");
  })
);

dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) processFile(file);
});

function processFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
  };
  reader.readAsDataURL(file);
  uploadToBackend(file);
}

function uploadToBackend(file) {
  const formData = new FormData();
  formData.append("file", file); 


  fetch("http://localhost:8000/analyze/", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    classification.textContent = data.decision.classification;
    classification.style.backgroundColor = data.decision.classification === "REAL" ? "#d4edda" : "#f8d7da";
    classification.style.color = data.decision.classification === "REAL" ? "#155724" : "#721c24";
    confidence.textContent = `Confidence: ${data.decision.confidence}%`;
    document.getElementById("explanationText").textContent = data.decision.explanation;
    sourceList.innerHTML = "";
    data.decision.sources.forEach(link => {
      const li = document.createElement("li");
      li.textContent = link;
      sourceList.appendChild(li);
    });
  })
  .catch(err => {
    classification.textContent = "Error";
    classification.style.backgroundColor = "#fbe9e7";
    classification.style.color = "#c0392b";
    confidence.textContent = "";
    sourceList.innerHTML = "";
    console.error(err);
  });
}

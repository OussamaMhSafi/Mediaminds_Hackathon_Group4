const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const reasoningText = document.getElementById("reasoningText");
const loader = document.getElementById("loader");

// Function to format the reasoning text with proper styling
function formatReasoningText(text) {
  if (!text) return "";
  
  // First, escape any HTML to prevent XSS
  let formattedText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  
  // Format bold text
  formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  
  // Handle URLs in the text
  // First, handle markdown-style links: [Title](URL)
  formattedText = formattedText.replace(/\[(.*?)\]\((https?:\/\/[^\s)]+)\)/g, 
    '<a href="$2" target="_blank">$1</a>');
  
  // Handle numbered source lists with URLs
  // This matches patterns like "1. http://example.com" or "2. Title http://example.com"
  formattedText = formattedText.replace(
    /(\d+\.\s*)([^:]*?)(https?:\/\/[^\s]+)/g, 
    function(match, number, title, url) {
      // If we already have a link tag, don't process further
      if (match.includes('<a href')) return match;
      
      // If there's text before the URL, keep it
      if (title.trim()) {
        return `${number}${title.trim()} <a href="${url}" target="_blank">${url}</a>`;
      } else {
        return `${number}<a href="${url}" target="_blank">${url}</a>`;
      }
    }
  );
  
  // Finally, handle any remaining plain URLs
  formattedText = formattedText.replace(
    /\b(https?:\/\/[a-z0-9\.\-]+\.[a-z]{2,}(?:\/[^\s]*)?)\b/gi,
    function(url) {
      // Skip if already in a link
      if (url.includes('<a href')) return url;
      return `<a href="${url}" target="_blank">${url}</a>`;
    }
  );
  
  return formattedText;
}

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
  // Show loader
  loader.classList.remove("hidden");
  
  // Display preview
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
  };
  reader.readAsDataURL(file);
  
  // Upload to backend
  uploadToBackend(file);
}

function uploadToBackend(file) {
  const formData = new FormData();
  formData.append("file", file); 

  // Configure server URL - default to localhost, but can be changed
  // You can change this URL if your server is running on a different address
  const serverUrl = "http://127.0.0.1:8000/analyze/";
  
  // Set a timeout for the fetch request
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000); // 60 second timeout (analysis might take time)
  
  fetch(serverUrl, {
    method: "POST",
    body: formData,
    signal: controller.signal
  })
  .then(res => {
    clearTimeout(timeoutId); // Clear the timeout
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}: ${res.statusText}`);
    }
    return res.json();
  })
  .then(data => {
    // Hide loader
    loader.classList.add("hidden");
    
    // Display agent's reasoning with proper formatting
    if (data.agent_reasoning) {
      // Format the text to render bold and links properly
      const formattedText = formatReasoningText(data.agent_reasoning);
      reasoningText.innerHTML = formattedText;
      
      // Debug info
      console.log("Original text:", data.agent_reasoning);
      console.log("Formatted text:", formattedText);
    } else {
      reasoningText.textContent = "No analysis result available.";
    }
  })
  .catch(err => {
    // Hide loader
    loader.classList.add("hidden");
    
    // Display error message
    let errorMessage = "";
    if (err.name === 'AbortError') {
      errorMessage = "Connection timed out. The server took too long to respond.";
    } else if (err.message.includes('Failed to fetch')) {
      errorMessage = "Could not connect to the server. Please check if the backend service is running and accessible.";
    } else {
      errorMessage = "An error occurred during analysis: " + err.message;
    }
    
    reasoningText.textContent = errorMessage;
    console.error(err);
  });
}
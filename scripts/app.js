// scripts/app.js
document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const inputField = document.getElementById("userInput");
  const resultBox = document.getElementById("result");

  if (analyzeBtn && inputField && resultBox) {
    analyzeBtn.addEventListener("click", async () => {
      const text = inputField.value.trim();
      if (!text) {
        resultBox.textContent = "Please enter some text.";
        return;
      }
      resultBox.textContent = "Analyzing...";
      const explanation = await EcoAI.explain(text, "text");
      resultBox.textContent = explanation;
    });
  }
});

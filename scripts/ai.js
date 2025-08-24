// scripts/ai.js
// Real AI integration using Hugging Face Inference API
// Replace with your free Hugging Face API key
const HUGGING_FACE_API_KEY = "YOUR_HF_API_KEY_HERE";

// Text model (free small model for demonstration)
const TEXT_MODEL = "google/flan-t5-small";

// Image model (image captioning to turn image → text)
const IMAGE_MODEL = "nlpconnect/vit-gpt2-image-captioning";

async function queryText(prompt) {
  const response = await fetch(
    `https://api-inference.huggingface.co/models/${TEXT_MODEL}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HUGGING_FACE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: prompt }),
    }
  );
  if (!response.ok) throw new Error("Text model request failed");
  const result = await response.json();
  return result[0]?.generated_text || "No response from AI.";
}

async function queryImage(imageBase64) {
  // Convert base64 to blob
  const binary = atob(imageBase64.split(",")[1]);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    array[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([array], { type: "image/jpeg" });

  const response = await fetch(
    `https://api-inference.huggingface.co/models/${IMAGE_MODEL}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HUGGING_FACE_API_KEY}`,
      },
      body: blob,
    }
  );
  if (!response.ok) throw new Error("Image model request failed");
  const result = await response.json();
  return result[0]?.generated_text || "No description available.";
}

// Main API exposed to rest of app
const EcoAI = {
  async explain(input, type = "text") {
    try {
      if (type === "text") {
        return await queryText(input);
      } else if (type === "image") {
        // Convert image to description text first
        const desc = await queryImage(input);
        // Then send description to text model for eco analysis
        return await queryText(
          `Analyze the environmental impact of this product: ${desc}`
        );
      }
    } catch (err) {
      console.error(err);
      return "Error analyzing. Please try again.";
    }
  },
};

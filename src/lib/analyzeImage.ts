/**
 * API handler for image analysis
 * Works on Vercel + local dev
 */

// Always define full URL inside the variable (no http:// inside code)
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL; 
// Example:
// VITE_BACKEND_URL=http://34.148.8.28:3000

export const analyzeImageViaProxy = async (imageFile: File) => {
  try {
    if (!BACKEND_URL) {
      throw new Error("VITE_BACKEND_URL is not defined");
    }

    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await fetch(`${BACKEND_URL}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        `API request failed with status ${response.status}: ${
          errorData.message || "Unknown error"
        }`
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error analyzing image:", error);
    throw error;
  }
};

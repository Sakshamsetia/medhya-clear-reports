/**
 * API handler for image analysis
 * This proxies requests to the external API
 */
const url = import.meta.env.VITE_BACKEND_URL
export const analyzeImageViaProxy = async (imageFile: File) => {
    console.log(url)
  try {
    // Create FormData and append the image file
    const formData = new FormData();
    formData.append('image', imageFile);

    // Call the API endpoint with multipart/form-data
    const response = await fetch(`http://${url}:3000/analyse`, {
        
      method: 'POST',
      // Don't set Content-Type header - browser will set it automatically with boundary
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        `API request failed with status ${response.status}: ${
          errorData.message || 'Unknown error'
        }`
      );
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

// /**
//  * Backend server for handling API requests and CORS
//  * Run this in development to proxy the analysis API
//  */

// import express from 'express';
// import cors from 'cors';

// const app = express();
// const PORT = 3001;

// // Middleware
// app.use(cors());
// app.use(express.json({ limit: '50mb' }));
// app.use(express.urlencoded({ limit: '50mb' }));

// // Health check
// app.get('/health', (req, res) => {
//   res.json({ status: 'ok' });
// });

// // Proxy endpoint for image analysis
// app.post('/analyse', async (req, res) => {
//   try {
//     const { image } = req.body;

//     if (!image) {
//       return res.status(400).json({ error: 'Image data is required' });
//     }

//     // Forward the request to the actual analysis API
//     const externalResponse = await fetch('http://34.148.8.28:3000/analyse', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ image }),
//     });

//     if (!externalResponse.ok) {
//       throw new Error(
//         `External API returned ${externalResponse.status}: ${externalResponse.statusText}`
//       );
//     }

//     const data = await externalResponse.json();
//     res.json(data);
//   } catch (error) {
//     console.error('Error in /api/analyse:', error);
//     res.status(500).json({
//       error: 'Failed to analyze image',
//       message: error instanceof Error ? error.message : 'Unknown error',
//     });
//   }
// });

// app.listen(PORT, () => {
//   console.log(`Backend server running on http://localhost:${PORT}`);
// });

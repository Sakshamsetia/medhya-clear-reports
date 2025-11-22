import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, Camera, Download, MessageSquare, Stethoscope, Terminal, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import MedicalLoader from "@/components/MedicalLoader";
import jsPDF from "jspdf";
import supabase from "@/lib/supabaseClient";
import { usePatientReport } from "@/context/PatientReportContext";
import { useNavigate } from "react-router-dom";

interface MedicalReport {
  name: string;
  gender: string;
  dateOfBirth: string;
  dateOfTest: string;
  diagnosis: string;
  explanation: string;
}

interface TerminalLine {
  type: "output" | "info" | "success" | "error";
  text: string;
  timestamp: string;
}

/* ------------------------------------------------------------------
   CAMERA COMPONENT - ENHANCED MOBILE VERSION
-------------------------------------------------------------------- */
const CameraCapture = ({
  onCapture,
  onClose,
}: {
  onCapture: (file: File, preview: string) => void;
  onClose: () => void;
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>("");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let mounted = true;
    let currentStream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        // Try with back camera first (better for mobile)
        let constraints: MediaStreamConstraints = {
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          },
          audio: false
        };

        try {
          currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
          // Fallback: try with any camera
          console.warn("Back camera not available, trying any camera:", err);
          constraints = {
            video: {
              width: { ideal: 1920 },
              height: { ideal: 1080 }
            },
            audio: false
          };
          currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        }

        if (!mounted) {
          // Component unmounted, stop the stream
          currentStream?.getTracks().forEach(track => track.stop());
          return;
        }

        setStream(currentStream);
        
        if (videoRef.current) {
          videoRef.current.srcObject = currentStream;
          
          // Wait for video to be ready
          videoRef.current.onloadedmetadata = () => {
            if (mounted) {
              videoRef.current?.play().then(() => {
                setIsReady(true);
              }).catch(err => {
                console.error("Error playing video:", err);
                setError("Failed to start camera preview");
              });
            }
          };
        }
      } catch (error) {
        console.error("Camera error:", error);
        if (mounted) {
          if (error instanceof DOMException) {
            if (error.name === "NotAllowedError") {
              setError("Camera access denied. Please allow camera permissions.");
            } else if (error.name === "NotFoundError") {
              setError("No camera found on this device.");
            } else if (error.name === "NotReadableError") {
              setError("Camera is already in use by another application.");
            } else {
              setError("Unable to access camera. Please try again.");
            }
          } else {
            setError("An unexpected error occurred while accessing the camera.");
          }
          
          // Auto close after showing error
          setTimeout(() => {
            if (mounted) onClose();
          }, 3000);
        }
      }
    };

    startCamera();

    // Cleanup function
    return () => {
      mounted = false;
      if (currentStream) {
        currentStream.getTracks().forEach(track => {
          track.stop();
        });
      }
    };
  }, []); // Empty dependency array

  const captureImage = () => {
    if (!videoRef.current || !isReady) {
      setError("Camera not ready yet. Please wait.");
      return;
    }

    const video = videoRef.current;
    const canvas = document.createElement("canvas");

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Handle case where video dimensions aren't available yet
    if (canvas.width === 0 || canvas.height === 0) {
      setError("Camera not ready. Please try again.");
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setError("Failed to create canvas context");
      return;
    }

    // Draw the video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob, then to File
    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], `medical_capture_${Date.now()}.jpg`, {
            type: "image/jpeg",
          });
          const preview = canvas.toDataURL("image/jpeg", 0.95);

          // Stop the camera stream
          if (stream) {
            stream.getTracks().forEach((track) => track.stop());
          }

          onCapture(file, preview);
        } else {
          setError("Failed to capture image. Please try again.");
        }
      },
      "image/jpeg",
      0.95
    );
  };

  const handleClose = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black z-[9999] flex flex-col">
      {/* Close button - top right */}
      <div className="absolute top-4 right-4 z-10">
        <Button 
          size="icon" 
          variant="ghost" 
          onClick={handleClose}
          className="bg-white/20 hover:bg-white/30 text-white rounded-full h-12 w-12"
        >
          <X className="h-6 w-6" />
        </Button>
      </div>

      {/* Video container */}
      <div className="flex-1 flex items-center justify-center relative overflow-hidden">
        {error ? (
          <div className="text-center p-6">
            <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-4">
              <p className="text-white text-lg">{error}</p>
            </div>
            <Button onClick={handleClose} variant="secondary">
              Close
            </Button>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
              style={{
                transform: 'scaleX(1)', // Remove mirror effect on mobile
              }}
            />
            
            {/* Camera ready indicator */}
            {!isReady && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="text-white text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                  <p>Starting camera...</p>
                </div>
              </div>
            )}

            {/* Capture frame guide */}
            {isReady && (
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-4 border-2 border-white/50 rounded-lg" />
              </div>
            )}
          </>
        )}
      </div>

      {/* Controls - bottom */}
      {!error && (
        <div className="bg-black/80 p-6 flex flex-col gap-4">
          <Button 
            size="lg" 
            onClick={captureImage}
            disabled={!isReady}
            className="w-full h-14 text-lg"
          >
            <Camera className="mr-2 h-5 w-5" />
            {isReady ? "Capture Photo" : "Preparing Camera..."}
          </Button>
          
          <Button 
            size="lg" 
            variant="outline" 
            onClick={handleClose}
            className="w-full h-14 text-lg"
          >
            Cancel
          </Button>

          <p className="text-center text-white/70 text-sm">
            Position the medical report within the frame
          </p>
        </div>
      )}
    </div>
  );
};

/* ------------------------------------------------------------------
   MAIN DASHBOARD
-------------------------------------------------------------------- */

export default function Dashboard() {
  const navigate = useNavigate();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [openCamera, setOpenCamera] = useState(false);
  const { report, setReport } = usePatientReport();
  const [saving, setSaving] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState<TerminalLine[]>([]);
  const [showTerminal, setShowTerminal] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Restore state on component mount
  useEffect(() => {
    if (report && !selectedImage) {
      const savedImage = sessionStorage.getItem('selectedImage');
      if (savedImage) {
        setSelectedImage(savedImage);
      }
    }
  }, [report, selectedImage]);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalOutput]);

  const addTerminalLine = (type: TerminalLine["type"], text: string) => {
    const newLine: TerminalLine = {
      type,
      text,
      timestamp: new Date().toLocaleTimeString(),
    };
    setTerminalOutput((prev) => [...prev, newLine]);
  };

  const getLineColor = (type: TerminalLine["type"]) => {
    switch (type) {
      case "output":
        return "text-green-400";
      case "info":
        return "text-blue-400";
      case "success":
        return "text-green-500 font-bold";
      case "error":
        return "text-red-500";
      default:
        return "text-gray-300";
    }
  };

  /* ------------------- FETCH PATIENT DATA FROM SUPABASE ------------------- */
  const fetchPatientData = async () => {
    try {
      const { data: { user }, error: userError } = await supabase.auth.getUser();

      if (userError || !user) {
        throw new Error("User not authenticated");
      }

      const { data: patientData, error: patientError } = await supabase
        .from('patients')
        .select('name')
        .eq('id', user.id)
        .single();

      if (patientError) {
        console.error("Supabase error details:", patientError);
        throw new Error(`Failed to fetch patient data: ${patientError.message}`);
      }

      return patientData;
    } catch (error) {
      console.error("Error fetching patient data:", error);
      toast({
        title: "Error",
        description: "Failed to fetch patient information",
        variant: "destructive",
      });
      return null;
    }
  };

  /* ------------------- ANALYZE IMAGE WITH REAL-TIME STREAMING ------------------- */
  const analyzeImageWithStream = async (file: File) => {
    setIsLoading(true);
    setShowTerminal(true);
    setTerminalOutput([]);
    addTerminalLine("info", "Starting analysis...");

    const formData = new FormData();
    formData.append("image", file);

    try {
      // First fetch patient data
      addTerminalLine("info", "Fetching patient information...");
      const patientData = await fetchPatientData();

      if (!patientData) {
        throw new Error("Failed to load patient data");
      }
      addTerminalLine("success", "Patient data loaded successfully");

      // Start streaming analysis
      addTerminalLine("info", "Connecting to analysis server at localhost:4000...");

      // Try to connect with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      let response;
      try {
        response = await fetch(import.meta.env.VITE_BACKEND_URL_2, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
      } catch (err) {
        clearTimeout(timeoutId);
        // If streaming endpoint fails, try regular endpoint
        addTerminalLine("info", "Stream endpoint unavailable, trying standard endpoint...");
        response = await fetch(import.meta.env.VITE_BACKEND_URL_1, {
          method: "POST",
          body: formData,
        });
      }

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const contentType = response.headers.get("content-type");

      // Check if it's a streaming response or JSON
      if (contentType?.includes("text/event-stream")) {
        // Streaming response
        addTerminalLine("success", "Connected! Starting image analysis with live output...");

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error("No reader available");
        }

        let analysisResult: any = null;
        let buffer = ""; // Buffer to handle incomplete chunks

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          // Decode and add to buffer
          buffer += decoder.decode(value, { stream: true });

          // Split by newlines but keep incomplete lines in buffer
          const lines = buffer.split("\n");

          // Keep the last incomplete line in buffer
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.trim().startsWith("data: ")) {
              try {
                const jsonStr = line.trim().slice(6);
                if (!jsonStr) continue; // Skip empty data

                const data = JSON.parse(jsonStr);

                switch (data.type) {
                  case "output":
                    addTerminalLine("output", data.data);
                    break;
                  case "info":
                    addTerminalLine("info", data.data);
                    break;
                  case "success":
                    addTerminalLine("success", data.data);
                    break;
                  case "error":
                    addTerminalLine("error", data.data);
                    break;
                  case "result":
                    analysisResult = data.data;
                    addTerminalLine("success", "Analysis complete! Processing results...");
                    break;
                  case "done":
                    // Analysis is done
                    break;
                  case "keepalive":
                    // Just a keepalive, ignore
                    break;
                }
              } catch (e) {
                // Only log if it's not an empty or whitespace line
                if (line.trim().length > 6) {
                  console.error("Parse error for line:", line, e);
                }
              }
            }
          }
        }

        // Process any remaining data in buffer
        if (buffer.trim().startsWith("data: ")) {
          try {
            const jsonStr = buffer.trim().slice(6);
            if (jsonStr) {
              const data = JSON.parse(jsonStr);
              if (data.type === "result") {
                analysisResult = data.data;
              }
            }
          } catch (e) {
            console.error("Parse error in final buffer:", e);
          }
        }

        if (!analysisResult) {
          throw new Error("No analysis result received");
        }

        // Create medical report from streaming result
        const medicalReport: MedicalReport = {
          name: patientData.name,
          gender: patientData.sex,
          dateOfBirth: patientData.dob,
          dateOfTest: new Date().toLocaleDateString(),
          diagnosis: analysisResult.Diagnosis || "No diagnosis available",
          explanation: analysisResult.Explanation || "No explanation available",
        };

        setReport(medicalReport);
        addTerminalLine("success", "✓ Report generated successfully!");

      } else {
        // Regular JSON response (fallback)
        addTerminalLine("info", "Using standard analysis mode...");
        const analysisResult = await response.json();

        addTerminalLine("output", "Processing image data...");
        addTerminalLine("output", "Running medical analysis...");
        addTerminalLine("success", "Analysis complete!");

        // Create medical report from regular result
        const medicalReport: MedicalReport = {
          name: patientData.name,
          gender: patientData.sex,
          dateOfBirth: patientData.dob,
          dateOfTest: new Date().toLocaleDateString(),
          diagnosis: analysisResult.Diagnosis || "No diagnosis available",
          explanation: analysisResult.Explanation || "No explanation available",
        };

        setReport(medicalReport);
        addTerminalLine("success", "✓ Report generated successfully!");
      }

      setTimeout(() => {
        setShowTerminal(false);
        setIsLoading(false);
      }, 2000);

      toast({
        title: "Report Generated",
        description: "Your medical report is ready.",
      });

    } catch (error) {
      console.error("Error analyzing image:", error);

      // Check if it's a connection error
      if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
        addTerminalLine("error", "✗ Cannot connect to analysis server on localhost:4000");
        addTerminalLine("error", "Please make sure the Flask server is running:");
        addTerminalLine("info", "Run: python py.py");
      } else {
        addTerminalLine("error", `Error: ${error instanceof Error ? error.message : "Unknown error"}`);
      }

      setTimeout(() => {
        setIsLoading(false);
      }, 5000);

      toast({
        title: "Connection Error",
        description: "Cannot connect to analysis server. Please ensure Flask backend is running on port 4000.",
        variant: "destructive",
      });
    }
  };

  /* ------------------- FILE / CAMERA IMAGE HANDLING ------------------- */
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Store the actual File object
    setImageFile(file);

    // Create preview URL
    const reader = new FileReader();
    reader.onloadend = () => {
      const imageData = reader.result as string;
      setSelectedImage(imageData);
      sessionStorage.setItem('selectedImage', imageData);
      analyzeImageWithStream(file);
    };
    reader.readAsDataURL(file);
  };

  const handleCameraCapture = (file: File, preview: string) => {
    setImageFile(file);
    setSelectedImage(preview);
    sessionStorage.setItem('selectedImage', preview);
    setOpenCamera(false);
    analyzeImageWithStream(file);
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(error);
    });
  };

  const handleDownloadReport = async () => {
    if (!report) return;

    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();

    let imgBase64 = null;
    if (imageFile) {
      imgBase64 = await fileToBase64(imageFile);
    } else if (selectedImage) {
      imgBase64 = selectedImage;
    }

    const primaryColor = "#0A4A78";
    const lightGray = "#f1f1f1";
    const textDark = "#333";
    const warningRed = "#DC2626";

    // Header
    pdf.setFillColor(primaryColor);
    pdf.rect(0, 0, pageWidth, 30, "F");
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(22);
    pdf.setTextColor("#fff");
    pdf.text("MEDICAL REPORT", pageWidth / 2, 18, { align: "center" });

    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(11);
    pdf.text("AI-Assisted Medical Documentation by MedhyaMed", pageWidth / 2, 27, { align: "center" });

    let y = 40;

    // AI DISCLAIMER BOX
    pdf.setFillColor(255, 245, 245); // Light red background
    pdf.setDrawColor(warningRed);
    pdf.setLineWidth(0.5);
    pdf.rect(10, y, pageWidth - 20, 20, "FD");
    
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(warningRed);
    pdf.text("⚠ AI DISCLAIMER", 15, y + 6);
    
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(9);
    pdf.setTextColor(textDark);
    const disclaimerText = "I am an AI, not a doctor. This analysis is for informational purposes only and should be verified by a qualified healthcare professional. Do not use this report as a substitute for professional medical advice, diagnosis, or treatment.";
    const disclaimerLines = pdf.splitTextToSize(disclaimerText, pageWidth - 30);
    pdf.text(disclaimerLines, 15, y + 12);

    y += 30;

    // Patient Information
    pdf.setFont("normal", "normal");
    pdf.setFillColor(lightGray);
    pdf.rect(10, y, pageWidth - 20, 10, "F");

    pdf.setFontSize(14);
    pdf.setTextColor(primaryColor);
    pdf.text("Patient Information", 15, y + 7);

    y += 20;
    pdf.setFontSize(11);
    pdf.setTextColor(textDark);

    pdf.text(`Name: ${report.name}`, 15, y);
    pdf.text(`Date of Test: ${report.dateOfTest}`, 15, y + 21);

    y += 35;

    // Diagnosis
    pdf.setFillColor(lightGray);
    pdf.rect(10, y, pageWidth - 20, 10, "F");
    pdf.setFontSize(14);
    pdf.setTextColor(primaryColor);
    pdf.text("Diagnosis", 15, y + 7);

    y += 15;
    pdf.setFontSize(11);
    pdf.setTextColor(textDark);

    const diagLines = pdf.splitTextToSize(report.diagnosis, pageWidth - 30);
    pdf.text(diagLines, 15, y);

    y += diagLines.length * 7 + 10;

    // Check if we need a new page
    if (y > pageHeight - 80) {
      pdf.addPage();
      y = 20;
    }

    // Explanation
    pdf.setFillColor(lightGray);
    pdf.rect(10, y, pageWidth - 20, 10, "F");
    pdf.setFontSize(14);
    pdf.setTextColor(primaryColor);
    pdf.text("Explanation", 15, y + 7);

    y += 15;
    pdf.setFontSize(11);
    pdf.setTextColor(textDark);

    const expLines = pdf.splitTextToSize(report.explanation, pageWidth - 30);
    pdf.text(expLines, 15, y);

    y += expLines.length * 7 + 15;

    // Attached Image
    if (imgBase64) {
      // Check if we need a new page for the image
      if (y + 170 > pageHeight - 10) {
        pdf.addPage();
        y = 20;
      }

      pdf.setFillColor(lightGray);
      pdf.rect(10, y, pageWidth - 20, 10, "F");
      pdf.setFontSize(14);
      pdf.setTextColor(primaryColor);
      pdf.text("Attached Image", 15, y + 7);

      y += 20;

      const imgWidth = pageWidth - 40;
      const imgHeight = 150;

      if (y + imgHeight > pageHeight - 10) {
        pdf.addPage();
        y = 20;
      }

      pdf.addImage(imgBase64, "JPEG", 20, y, imgWidth, imgHeight);
    }

    // Footer
    const footerY = pageHeight - 10;
    pdf.setFontSize(10);
    pdf.setTextColor("#777");
    pdf.text("© MedhyaMed — Confidential Medical Document", pageWidth / 2, footerY, { align: "center" });

    pdf.save(`medical_report_${report.name}_${report.dateOfTest}.pdf`);
  };

  const handleChatWithAI = () => {
    if (!report) {
      toast({
        title: "No Report",
        description: "Please generate a report first",
        variant: "destructive",
      });
      return;
    }
    navigate('/chats');
  };

  const handleAskDoctor = () => {
    if (!report) {
      toast({
        title: "No Report",
        description: "Please generate a report first",
        variant: "destructive",
      });
      return;
    }
    navigate('/doctors');
  };

  const resetUpload = () => {
    setSelectedImage(null);
    setImageFile(null);
    setReport(null);
    setIsLoading(false);
    setShowTerminal(false);
    setTerminalOutput([]);
    sessionStorage.removeItem('selectedImage');
  };

  const saveReport = async () => {
    if (!report) return alert("No report to save!");
    setSaving(true);

    try {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) return alert("User not logged in");

      const { error } = await supabase.from("records").insert([
        {
          user_id: user.id,
          report_json: report,
        },
      ]);

      if (error) throw error;
      toast({
        title: "Success",
        description: "Report saved successfully!",
      });
    } catch (err) {
      console.error("Error saving report:", err);
      toast({
        title: "Error",
        description: "Failed to save report.",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const [status, s_status] = useState(false)
  function thinking_click() {
    const box = document.getElementById("thinking_box");
    if (box) {
      if (status == true) {
        box.style.height = '0';
        s_status(false)
      } else {
        box.style.height = '150px';
        s_status(true)
      }
    }
  }

  const TypingIndicator: React.FC = () => {
    const [text, setText] = useState("Thinking");

    useEffect(() => {
      const timer = setTimeout(() => setText("This may take some time, don't reload or close your browser"), 10000);
      return () => clearTimeout(timer);
    }, []);

    return (
      <span className="flex items-center gap-1 text-gray-500 font-medium">
        {text}
        <span className="dot">.</span>
        <span className="dot">.</span>
        <span className="dot">.</span>
        <style>{`
          .dot {
            animation: blink 4s infinite;
          }
          .dot:nth-child(2) { animation-delay: 0.5s; }
          .dot:nth-child(3) { animation-delay: 1s; }

          @keyframes blink {
            0%, 20%, 50%, 80%, 100% { opacity: 0; }
            10%, 30%, 60%, 90% { opacity: 1; }
          }
        `}</style>
      </span>
    );
  };


  const TypingIndicatorA: React.FC = () => {
    const [text, setText] = useState("Analyzing Your Report");

    useEffect(() => {
      const typingMessages = [
        "Looking for a better answer for you…",
        "Analyzing your report…",
        "Evaluating your recent data…",
        "Generating insights from your test results…",
        "Cross-referencing similar cases…",
        "Reviewing key markers and trends…",
        "Processing patient history…",
        "Compiling diagnostic information…",
        "Searching for relevant medical literature…",
        "Checking potential health indicators…",
        "Comparing with previous reports…",
        "Identifying noteworthy patterns…",
        "Summarizing findings…",
        "Preparing a detailed analysis…",
        "Assessing risk factors…",
        "Validating your results…",
        "Correlating symptoms and reports…",
        "Evaluating laboratory metrics…",
        "Extracting actionable insights…",
        "Optimizing recommendations…",
        "Scanning for anomalies…",
        "Generating comprehensive summary…",
        "Interpreting your test data…",
        "Calculating likelihoods and probabilities…",
        "Filtering important results…",
        "Producing a readable report…",
        "Highlighting key findings…",
        "Reviewing your health indicators…",
        "Drafting insights for you…",
        "Finalizing analysis, almost done…",
        "Analyzing trends across metrics…",
        "Assessing overall patient condition…",
        "Examining possible correlations…",
        "Reviewing critical markers…",
        "Checking for abnormalities…",
        "Compiling diagnostic recommendations…",
        "Cross-checking lab results…",
        "Evaluating test reliability…",
        "Preparing actionable insights…",
        "Generating personalized advice…",
        "Summarizing test highlights…",
        "Analyzing for inconsistencies…",
        "Reviewing patient history context…",
        "Validating your health data…",
        "Producing easy-to-read summary…",
        "Highlighting abnormal results…",
        "Reviewing trends for accuracy…",
        "Generating final recommendations…",
        "Preparing report for you…",
        "Almost done analyzing your report…"
      ];

      const timer = setInterval(() => setText(typingMessages[Math.floor(Math.random() * 50)]), 6000);
      return () => clearInterval(timer);
    }, []);

    return (
      <span className="flex items-center gap-1 text-gray-500 font-medium">
        {text}
        <span className="dot">.</span>
        <span className="dot">.</span>
        <span className="dot">.</span>
        <style>{`
          .dot {
            animation: blink 4s infinite;
          }
          .dot:nth-child(2) { animation-delay: 0.5s; }
          .dot:nth-child(3) { animation-delay: 1s; }

          @keyframes blink {
            0%, 20%, 50%, 80%, 100% { opacity: 0; }
            10%, 30%, 60%, 90% { opacity: 1; }
          }
        `}</style>
      </span>
    );
  };


  return (
    <main className="py-12 px-4 sm:px-6 lg:px-8 ">
      <div className="container mx-auto max-w-6xl">
        {openCamera && (
          <CameraCapture
            onCapture={handleCameraCapture}
            onClose={() => setOpenCamera(false)}
          />
        )}

        {/* Upload Section */}
        {!selectedImage && !isLoading && !report && (
          <div className="text-center space-y-8">
            <Card className="max-w-2xl mx-auto p-8">
              <CardContent className="space-y-6">
                <Upload className="h-24 w-24 text-primary mx-auto" />
                <p className="text-lg text-muted-foreground">
                  Upload a medical report or take a photo
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
                  <Button size="lg" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="mr-2" /> Browse Files
                  </Button>

                  <Button
                    size="lg"
                    variant="outline"
                    onClick={() => setOpenCamera(true)}
                  >
                    <Camera className="mr-2" /> Take Photo
                  </Button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </CardContent>
            </Card>
          </div>
        )}

        {/* Loading State with Terminal */}
        {isLoading && showTerminal && (
          <div className="flex flex-col items-center justify-center space-y-6">
            <MedicalLoader />

            {/* Real-time Terminal Output */}
            <Card className="w-full max-w-4xl">
              <CardContent className="p-4">
                <div style={{ border: '1px solid white', borderBottomColor: 'grey' }} className="flex items-center gap-2 p-1">
                  <Terminal className="h-5 w-5 text-green-400" />
                  <span style={{ cursor: 'pointer' }} className="grey-white font-semibold" onClick={thinking_click}><TypingIndicatorA /></span>
                </div>
                <div id="thinking_box" style={{ transition: "all 0.25s ease-in-out" }}
                  ref={terminalRef}
                  className="h-0 overflow-y-hidden font-mono text-sm space-y-1 scrollbar-thin scrollbar-thumb-gray-600"
                >
                  {terminalOutput.map((line, idx) => (
                    <div key={idx} className="flex gap-2">
                      <span className="text-gray-500 text-xs">[{line.timestamp}]</span>
                      <span className={getLineColor(line.type)}>{line.text}</span>
                    </div>
                  ))}
                  {isLoading && (
                    <div className="flex items-center gap-2 text-yellow-400">
                      <span className="animate-pulse">●</span>
                      <span><TypingIndicator /></span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Report Display */}
        {report && !isLoading && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <h2 className="text-3xl font-bold text-foreground">Medical Report</h2>
              <div className="flex flex-wrap gap-2">
                <Button onClick={saveReport} disabled={saving}>
                  {saving ? "Saving..." : "Save Report"}
                </Button>
                <Button onClick={handleDownloadReport} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download
                </Button>
                <Button onClick={resetUpload} variant="outline">
                  New Report
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="hover-scale transition-all">
                <CardContent className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-primary border-b pb-2">
                    Patient Information
                  </h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Name:</span>
                      <span className="font-medium">{report.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Date of Test:</span>
                      <span className="font-medium">{report.dateOfTest}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover-scale transition-all">
                <CardContent className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-primary border-b pb-2">
                    Uploaded Medical Image
                  </h3>
                  {selectedImage && (
                    <img
                      src={selectedImage}
                      alt="Medical Report"
                      className="w-full max-h-96 object-contain rounded-lg border"
                    />
                  )}
                </CardContent>
              </Card>

              <Card className="lg:col-span-2 hover-scale transition-all">
                <CardContent className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-primary border-b pb-2">
                    Diagnosis
                  </h3>
                  <p className="text-foreground leading-relaxed">{report.diagnosis}</p>
                </CardContent>
              </Card>

              <Card className="lg:col-span-2 hover-scale transition-all">
                <CardContent className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-secondary border-b pb-2">
                    Detailed Explanation
                  </h3>
                  <p className="text-foreground leading-relaxed">{report.explanation}</p>
                </CardContent>
              </Card>
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
              <Card className="hover-scale transition-all cursor-pointer" onClick={handleChatWithAI}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-primary/10 rounded-lg">
                      <MessageSquare className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-foreground">Chat with AI</h3>
                      <p className="text-muted-foreground text-sm">
                        Ask questions about your report to our AI assistant
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="hover-scale transition-all cursor-pointer" onClick={handleAskDoctor}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-secondary/10 rounded-lg">
                      <Stethoscope className="h-8 w-8 text-secondary" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-foreground">Ask a Doctor</h3>
                      <p className="text-muted-foreground text-sm">
                        Connect with a real doctor for professional consultation
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
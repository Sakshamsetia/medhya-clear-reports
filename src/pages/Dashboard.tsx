"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, Camera, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import MedicalLoader from "@/components/MedicalLoader";
import jsPDF from "jspdf";
import ParticleBackground from '@/components/particle';
import supabase from "@/lib/supabaseClient";
import { analyzeImageViaProxy } from "@/lib/analyzeImage";

interface MedicalReport {
  name: string;
  gender: string;
  dateOfBirth: string;
  dateOfTest: string;
  diagnosis: string;
  explanation: string;
}

/* ------------------------------------------------------------------
   HELPER FUNCTION - CONVERT BASE64 TO FILE
-------------------------------------------------------------------- */
const base64ToFile = (base64String: string, filename: string): File => {
  // Remove data URL prefix if present
  const base64Data = base64String.split(',')[1] || base64String;
  const mimeType = base64String.match(/data:([^;]+);/)?.[1] || 'image/jpeg';
  
  // Convert base64 to binary
  const byteString = atob(base64Data);
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uint8Array = new Uint8Array(arrayBuffer);
  
  for (let i = 0; i < byteString.length; i++) {
    uint8Array[i] = byteString.charCodeAt(i);
  }
  
  // Create blob and convert to File
  const blob = new Blob([arrayBuffer], { type: mimeType });
  return new File([blob], filename, { type: mimeType });
};

/* ------------------------------------------------------------------
   CAMERA COMPONENT - WORKS ON MOBILE CAMERA + DESKTOP WEBCAM
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

  useEffect(() => {
    const startCamera = async () => {
      try {
        const camStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
        });

        setStream(camStream);
        if (videoRef.current) videoRef.current.srcObject = camStream;
      } catch (error) {
        console.error("Camera error:", error);
        onClose();
      }
    };

    startCamera();

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const captureImage = () => {
    const video = videoRef.current!;
    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(video, 0, 0);

    // Convert canvas to blob, then to File
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        const preview = canvas.toDataURL("image/jpeg");
        onCapture(file, preview);
      }
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-[999] flex flex-col items-center justify-center p-4">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full max-w-md rounded-lg shadow-xl border-2 border-white"
      />

      <div className="flex gap-4 mt-6">
        <Button size="lg" onClick={captureImage}>
          Capture
        </Button>
        <Button size="lg" variant="outline" onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------
   MAIN DASHBOARD
-------------------------------------------------------------------- */

export default function Dashboard() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [report, setReport] = useState<MedicalReport | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [openCamera, setOpenCamera] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  /* ------------------- FETCH PATIENT DATA FROM SUPABASE ------------------- */
  const fetchPatientData = async () => {
    try {
      // Get current user
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      
      if (userError || !user) {
        throw new Error("User not authenticated");
      }

      // Fetch patient data from patients table
      const { data: patientData, error: patientError } = await supabase
        .from('patients')
        .select('name,sex,dob')
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

  /* ------------------- SEND IMAGE TO API ------------------- */
  const analyzeImage = async (file: File) => {
    try {
      const data = await analyzeImageViaProxy(file);
      console.log("Analysis result:", data);
      return data;
    } catch (error) {
      console.error("Error analyzing image:", error);
      toast({
        title: "Analysis Error",
        description: error instanceof Error 
          ? error.message 
          : "Failed to analyze the image. Please try again.",
        variant: "destructive",
      });
      throw error;
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
      setSelectedImage(reader.result as string);
      processImage(file);
    };
    reader.readAsDataURL(file);
  };

  const handleCameraCapture = (file: File, preview: string) => {
    setImageFile(file);
    setSelectedImage(preview);
    setOpenCamera(false);
    processImage(file);
  };

  const processImage = async (file: File) => {
    setIsLoading(true);

    try {
      const [patientData, analysisResult] = await Promise.all([
        fetchPatientData(),
        analyzeImage(file)
      ]);

      if (!patientData) {
        throw new Error("Failed to load patient data");
      }

      if (!analysisResult) {
        throw new Error("Failed to analyze image");
      }

      const medicalReport: MedicalReport = {
        name: patientData.name,
        gender: patientData.sex,
        dateOfBirth: patientData.dob,
        dateOfTest: new Date().toLocaleDateString(),
        diagnosis: analysisResult.Diagnosis || "No diagnosis available",
        explanation: analysisResult.Explanation || "No explanation available",
      };

      setReport(medicalReport);
      setIsLoading(false);

      toast({
        title: "Report Generated",
        description: "Your medical report is ready.",
      });
    } catch (error) {
      console.error("Error processing image:", error);
      setIsLoading(false);
      
      toast({
        title: "Error",
        description: "Failed to generate report. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDownloadReport = () => {
    if (!report) return;

    const pdf = new jsPDF();
    
    // Title
    pdf.setFontSize(20);
    pdf.text("MEDICAL REPORT", 105, 20, { align: "center" });
    
    // Patient Information
    pdf.setFontSize(14);
    pdf.text("Patient Information", 20, 40);
    pdf.setFontSize(11);
    pdf.text(`Name: ${report.name}`, 20, 50);
    pdf.text(`Gender: ${report.gender}`, 20, 64);
    pdf.text(`Date of Birth: ${report.dateOfBirth}`, 20, 71);
    pdf.text(`Date of Test: ${report.dateOfTest}`, 20, 78);
    
    // Diagnosis
    pdf.setFontSize(14);
    pdf.text("Diagnosis", 20, 95);
    pdf.setFontSize(11);
    const diagnosisLines = pdf.splitTextToSize(report.diagnosis, 170);
    pdf.text(diagnosisLines, 20, 105);
    
    // Explanation
    const yPosition = 105 + (diagnosisLines.length * 7) + 10;
    pdf.setFontSize(14);
    pdf.text("Explanation", 20, yPosition);
    pdf.setFontSize(11);
    const explanationLines = pdf.splitTextToSize(report.explanation, 170);
    pdf.text(explanationLines, 20, yPosition + 10);
    
    pdf.save(`medical_report_${report.name}_${report.dateOfTest}.pdf`);
  };

  /* ------------------- RESET ------------------- */
  const resetUpload = () => {
    setSelectedImage(null);
    setImageFile(null);
    setReport(null);
    setIsLoading(false);
  };

  return (
    <main className="py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-background to-accent">
      <div className="container mx-auto max-w-6xl">
        {/* Camera Component */}
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

        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-8">
            <MedicalLoader />
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold text-foreground">Analyzing Your Report</h2>
              <p className="text-muted-foreground">Please wait while we process your medical data...</p>
            </div>
          </div>
        )}

        {/* Report Display */}
        {report && !isLoading && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex justify-between items-center">
              <h2 className="text-3xl font-bold text-foreground">Medical Report</h2>
              <div className="flex gap-2">
                <Button onClick={handleDownloadReport} className="gap-2">
                  <Download className="h-4 w-4" />
                  Download Report
                </Button>
                <Button onClick={resetUpload} variant="outline">
                  New Report
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Patient Information */}
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
                      <span className="text-muted-foreground">Gender:</span>
                      <span className="font-medium">{report.gender}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Date of Birth:</span>
                      <span className="font-medium">{report.dateOfBirth}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Date of Test:</span>
                      <span className="font-medium">{report.dateOfTest}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Uploaded Image */}
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

              {/* Diagnosis */}
              <Card className="lg:col-span-2 hover-scale transition-all">
                <CardContent className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-primary border-b pb-2">
                    Diagnosis
                  </h3>
                  <p className="text-foreground leading-relaxed">{report.diagnosis}</p>
                </CardContent>
              </Card>

              {/* Explanation */}
              <Card className="lg:col-span-2 hover-scale transition-all">
                <CardContent className="p-6 space-y-4">
                  <h3 className="text-xl font-semibold text-secondary border-b pb-2">
                    Detailed Explanation
                  </h3>
                  <p className="text-foreground leading-relaxed">{report.explanation}</p>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

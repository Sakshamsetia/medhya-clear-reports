"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, Camera, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import MedicalLoader from "@/components/MedicalLoader";
import jsPDF from "jspdf";
import ParticleBackground from '@/components/particle'

interface MedicalReport {
  name: string;
  age: number;
  gender: string;
  dateOfTest: string;
  bloodType: string;
  hemoglobin: string;
  wbc: string;
  platelets: string;
  glucose: string;
  cholesterol: string;
  diagnosis: string;
  recommendations: string;
}

/* ------------------------------------------------------------------
   CAMERA COMPONENT - WORKS ON MOBILE CAMERA + DESKTOP WEBCAM
-------------------------------------------------------------------- */
const CameraCapture = ({
  onCapture,
  onClose,
}: {
  onCapture: (img: string) => void;
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

    const photo = canvas.toDataURL("image/jpeg");
    onCapture(photo);
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
  const [report, setReport] = useState<MedicalReport | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [openCamera, setOpenCamera] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  /* ------------------- FILE / CAMERA IMAGE HANDLING ------------------- */
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
      processImage();
    };
    reader.readAsDataURL(file);
  };

  /* ------------------- SIMULATED PROCESSING ------------------- */
  const processImage = async () => {
    setIsLoading(true);

    await new Promise((res) => setTimeout(res, 2000)); // fake latency

    const mockReport: MedicalReport = {
      name: "John Doe",
      age: 35,
      gender: "Male",
      dateOfTest: new Date().toLocaleDateString(),
      bloodType: "O+",
      hemoglobin: "14.5 g/dL",
      wbc: "7,500 cells/μL",
      platelets: "250,000/μL",
      glucose: "95 mg/dL",
      cholesterol: "180 mg/dL",
      diagnosis: "All vital health parameters are normal.",
      recommendations:
        "Maintain a balanced diet, continue exercise, and get a check-up in 6 months.",
    };

    setReport(mockReport);
    setIsLoading(false);

    toast({
      title: "Report Generated",
      description: "Your medical report is ready.",
    });
  };

  /* ------------------- DOWNLOAD PDF ------------------- */
  const handleDownloadReport = () => {
    if (!report) return;

    const pdf = new jsPDF();
    pdf.setFontSize(18);
    pdf.text("MEDICAL REPORT", 20, 20);
    pdf.save("report.pdf");
  };

  /* ------------------- RESET ------------------- */
  const resetUpload = () => {
    setSelectedImage(null);
    setReport(null);
    setIsLoading(false);
  };

  /* ------------------- MAIN UI ------------------- */
  return (
    <main className="py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-background to-accent">
      <ParticleBackground/>
      {openCamera && (
        <CameraCapture
          onCapture={(img) => {
            setSelectedImage(img);
            setOpenCamera(false);
            processImage();
          }}
          onClose={() => setOpenCamera(false)}
        />
      )}

      <div className="container mx-auto max-w-6xl">
        {/* ------------------ START SCREEN ------------------ */}
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

        {/* ------------------ PROCESSING ------------------ */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6">
            <MedicalLoader />
            <p className="text-xl text-muted-foreground">Analyzing your report...</p>
          </div>
        )}

        {/* ------------------ REPORT SCREEN ------------------ */}
        {report && selectedImage && !isLoading && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex justify-between items-center">
              <h2 className="text-3xl font-bold">Medical Report</h2>

              <div className="flex gap-2">
                <Button onClick={handleDownloadReport}>
                  <Download className="mr-2 h-4 w-4" /> Download Report
                </Button>

                <Button variant="outline" onClick={resetUpload}>
                  New Report
                </Button>
              </div>
            </div>

            <Card className="p-6">
              <CardContent className="space-y-6">
                <img
                  src={selectedImage}
                  alt="Uploaded"
                  className="w-full max-h-96 object-contain rounded-lg border"
                />

                <div className="space-y-2">
                  <p><strong>Name:</strong> {report.name}</p>
                  <p><strong>Age:</strong> {report.age}</p>
                  <p><strong>Gender:</strong> {report.gender}</p>
                  <p><strong>Date:</strong> {report.dateOfTest}</p>
                  <p><strong>Diagnosis:</strong> {report.diagnosis}</p>
                  <p><strong>Recommendations:</strong> {report.recommendations}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  );
}

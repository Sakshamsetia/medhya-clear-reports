import { useState, useEffect } from "react";
import DoctorCard from "@/components/DoctorCard";
import MedicalLoader from "@/components/MedicalLoader";

interface Doctor {
  doctor_name: string;
  location: string;
  specialization: string;
  degree: string;
}

const Doctors = () => {
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchDoctors = async () => {
      setIsLoading(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock doctor data
      const mockDoctors: Doctor[] = [
        {
          doctor_name: "Dr. Sarah Johnson",
          location: "New York, NY",
          specialization: "Cardiologist",
          degree: "MD, FACC"
        },
        {
          doctor_name: "Dr. Michael Chen",
          location: "Los Angeles, CA",
          specialization: "Neurologist",
          degree: "MD, PhD"
        },
        {
          doctor_name: "Dr. Emily Rodriguez",
          location: "Chicago, IL",
          specialization: "Pediatrician",
          degree: "MD, FAAP"
        },
        {
          doctor_name: "Dr. James Williams",
          location: "Houston, TX",
          specialization: "Orthopedic Surgeon",
          degree: "MD, FAAOS"
        },
        {
          doctor_name: "Dr. Lisa Anderson",
          location: "Phoenix, AZ",
          specialization: "Dermatologist",
          degree: "MD, FAAD"
        },
        {
          doctor_name: "Dr. Robert Taylor",
          location: "Philadelphia, PA",
          specialization: "Oncologist",
          degree: "MD, FASCO"
        }
      ];
      
      setDoctors(mockDoctors);
      setIsLoading(false);
    };

    fetchDoctors();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-accent">
      <div className="container mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground">
              Our Medical Professionals
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Connect with experienced doctors across various specializations
            </p>
          </div>

          {isLoading ? (
            <div className="flex flex-col items-center justify-center min-h-[400px] space-y-8">
              <MedicalLoader />
              <p className="text-muted-foreground">Loading doctors...</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in">
              {doctors.map((doctor, index) => (
                <DoctorCard key={index} {...doctor} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Doctors;

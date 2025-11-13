import { useState, useEffect } from "react";
import DoctorCard from "@/components/DoctorCard";
import MedicalLoader from "@/components/MedicalLoader";
import supabase from "@/lib/supabaseClient";
import ParticleBackground from '@/components/particle'

// Shape expected by DoctorCard / component
interface Doctor {
  id: string;
  doctor_name: string;
  location?: string | null;
  specialization?: string | null;
  qualifications?: string | null;
}

const DoctorsPage = () => {
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchDoctors = async () => {
      setIsLoading(true);
      setErrorMsg(null);

      try {
        // Make sure we request `id` + the columns you want to map
        const { data, error } = await supabase
          .from("doctors")
          .select("id, name, location, specialization, qualifications");

        if (error) {
          console.error("Supabase error:", error);
          throw error;
        }

        if (!data || !Array.isArray(data)) {
          throw new Error("Unexpected response shape from the server");
        }

        // Map DB rows into the Doctor[] shape your UI expects
        const normalized: Doctor[] = data.map((r: any) => ({
          id: String(r.id),
          doctor_name: r.name ?? "", // map `name` -> `doctor_name`
          location: r.location ?? null,
          specialization: r.specialization ?? null,
          qualifications: r.qualifications ?? null,
        }));

        if (mounted) setDoctors(normalized);
      } catch (err: any) {
        console.error("Failed to fetch doctors:", err);
        if (mounted) {
          setDoctors([]);
          setErrorMsg(err?.message ?? "Failed to load doctors");
        }
      } finally {
        if (mounted) setIsLoading(false);
      }
    };

    fetchDoctors();

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-accent">
      <ParticleBackground/>
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
          ) : errorMsg ? (
            <div className="text-center text-red-500">{errorMsg}</div>
          ) : doctors.length === 0 ? (
            <div className="text-center text-muted-foreground">No doctors found</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in">
              {doctors.map((doctor) => (
                // DoctorCard will receive props doctor_name, location, specialization, qualifications
                <DoctorCard key={doctor.id} {...doctor} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DoctorsPage;

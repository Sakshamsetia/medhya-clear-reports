import { useEffect, useState } from "react";
import supabase from "@/lib/supabaseClient";

export default function useUserRole() {
  const [role, setRole] = useState<"patient" | "doctor" | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function checkRole() {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setRole(null);
        setLoading(false);
        return;
      }

      // Check patients
      const { data: isPatient } = await supabase
        .from("patients")
        .select("id")
        .eq("id", user.id)
        .single();

      if (isPatient) {
        setRole("patient");
        setLoading(false);
        return;
      }

      // Check doctors
      const { data: isDoctor } = await supabase
        .from("doctors")
        .select("id")
        .eq("id", user.id)
        .single();

      if (isDoctor) {
        setRole("doctor");
      }

      setLoading(false);
    }

    checkRole();
  }, []);

  return { role, loading };
}

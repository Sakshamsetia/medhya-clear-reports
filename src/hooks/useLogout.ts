import supabase from "@/lib/supabaseClient";

export async function logout() {
  await supabase.auth.signOut();
}

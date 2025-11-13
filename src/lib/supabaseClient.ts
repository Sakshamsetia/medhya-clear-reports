// lib/supabaseClient.ts
import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_KEY

if (!url || !anonKey) {
  throw new Error('Missing Supabase environment variables');
}
const supabase = createClient(url, anonKey);
export default supabase;
"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { usePatientReport } from "@/context/PatientReportContext";
import { motion, AnimatePresence } from "framer-motion";
import supabase from "@/lib/supabaseClient";

interface ChatMessage {
  role: "user" | "bot";
  text: string;
}

const Chat: React.FC = () => {
  const { report } = usePatientReport();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [botTyping, setBotTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  /* ------------------- FETCH MESSAGES ------------------- */
  const fetchMessages = async () => {
    try {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) return;

      const { data, error } = await supabase
        .from("chat_messages")
        .select("message")
        .eq("user_id", user.id)
        .order("created_at", { ascending: true });

      if (error) throw error;

      const chats = data?.map((row: any) => row.message) || [];
      setMessages(chats);
    } catch (err) {
      console.error("Error fetching messages:", err);
    }
  };

  useEffect(() => {
    fetchMessages();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* ------------------- SEND MESSAGE ------------------- */
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: ChatMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setBotTyping(true);

    try {
      // Get user
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) throw new Error("User not logged in");

      // Save user message in Supabase
      const { error: insertError } = await supabase.from("chat_messages").insert([
        { user_id: user.id, message: userMsg }
      ]);
      if (insertError) throw insertError;

      // Call your bot API
      const res = await axios.post(import.meta.env.VITE_BACKEND_CHAT , {
        message: input,
        reportJson: report,
      });

      const botMsg: ChatMessage = { role: "bot", text: res.data.reply };

      // Save bot response in Supabase
      const { error: botInsertError } = await supabase.from("chat_messages").insert([
        { user_id: user.id, message: botMsg }
      ]);
      if (botInsertError) throw botInsertError;

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("Error sending message:", err);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error contacting server." }
      ]);
    } finally {
      setBotTyping(false);
    }
  };

  /* ------------------- Typing Indicator Component ------------------- */
  const TypingIndicator: React.FC = () => {
    const [text, setText] = useState("Thinking");

    useEffect(() => {
      const timer = setTimeout(() => setText("Looking for a better answer for you"), 1800);
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

  return (
    <div id="chat-box" className="flex flex-col h-screen max-w-6xl mx-auto rounded-xl p-4">
      {/* Header */}
      <div className="text-center mb-6 mt-2">
        <h1 className="text-5xl font-bold text-blue-700">MedhyaExplainer</h1>
        <p className="text-lg mt-2">Hi, {report?.name || "Patient"} 👋</p>
        <p className="text-sm text-gray-600 mt-1">How can I help you today?</p>
      </div>

      {/* Chat Messages */}
      <div id= "chat-messages" className="flex-1 overflow-y-auto mb-4 px-2 space-y-2">
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg shadow ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white"
                    : "bg-gray-200 text-gray-800"
                }`}
              >
                {msg.text}
              </div>
            </motion.div>
          ))}

          {botTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex justify-start"
            >
              <div className="max-w-[60%] p-3 rounded-lg shadow bg-gray-200">
                <TypingIndicator />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 sticky bottom-0 bg-white p-2 rounded-xl">
        <input
          className="flex-1 p-3 rounded-lg border border-gray-300 "
          placeholder="Ask anything…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          className="px-5 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-all"
          onClick={sendMessage}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default Chat;

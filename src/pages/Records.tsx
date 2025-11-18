import { useEffect, useState } from "react";
import supabase from "@/lib/supabaseClient";
import { usePatientReport, MedicalReport } from "@/context/PatientReportContext";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Stethoscope, X } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function Records() {
    const { report, setReport } = usePatientReport();
    const navigate = useNavigate();
    const [records, setRecords] = useState<MedicalReport[]>([]);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [selectedReport, setSelectedReport] = useState<MedicalReport | null>(null);

    /* ------------------- FETCH USER RECORDS ------------------- */
    const fetchRecords = async () => {
        setLoading(true);
        try {
            const { data: { user }, error: userError } = await supabase.auth.getUser();
            if (userError || !user) {
                setRecords([]);
                return;
            }

            const { data, error } = await supabase
                .from("records")
                .select("report_json")
                .eq("user_id", user.id)
                .order("created_at", { ascending: false });

            if (error) throw error;

            const reports: MedicalReport[] = data?.map((r: any) => r.report_json) || [];
            setRecords(reports);
        } catch (err) {
            console.error("Error fetching records:", err);
            alert("Failed to fetch records.");
        } finally {
            setLoading(false);
        }
    };

    /* ------------------- SAVE CURRENT REPORT ------------------- */
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
            alert("Report saved successfully!");
            fetchRecords();
        } catch (err) {
            console.error("Error saving report:", err);
            alert("Failed to save report.");
        } finally {
            setSaving(false);
        }
    };

    /* ------------------- HANDLE ACTIONS ------------------- */
    const handleChatWithAI = () => {
        if (!selectedReport) return;
        // Set the selected report as the current report in context
        setReport(selectedReport);
        // Close modal and navigate
        setSelectedReport(null);
        navigate('/chats');
    };

    const handleAskDoctor = () => {
        if (!selectedReport) return;
        // Set the selected report as the current report in context
        setReport(selectedReport);
        // Close modal and navigate
        setSelectedReport(null);
        navigate('/ask-doctor');
    };

    useEffect(() => {
        fetchRecords();
    }, []);


    return (
        <main className="py-12 px-4 sm:px-6 lg:px-8 min-h-screen relative">
            <div className="container mx-auto max-w-6xl space-y-6">
                {/* Header */}
                <div className="flex justify-between items-center">
                    <h1 className="text-3xl font-bold text-foreground">Medical Records</h1>
                    {report && (
                        <Button onClick={saveReport} disabled={saving}>
                            {saving ? "Saving..." : "Save Current Report"}
                        </Button>
                    )}
                </div>

                {/* Loading State */}
                {loading && (
                    <div className="flex justify-center items-center py-12">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
                    </div>
                )}

                {/* No Records */}
                {!loading && records.length === 0 && (
                    <Card className="p-12">
                        <p className="text-center text-gray-500 text-lg">No records found.</p>
                        <p className="text-center text-gray-400 text-sm mt-2">
                            Upload and analyze a medical report to create your first record.
                        </p>
                    </Card>
                )}

                {/* Records Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {records.map((rec, idx) => (
                        <Card
                            key={idx}
                            className="hover:scale-105 transition-all cursor-pointer shadow-lg hover:shadow-xl"
                            onClick={() => setSelectedReport(rec)}
                        >
                            <CardContent className="p-4 space-y-2">
                                <h3 className="text-lg font-semibold text-primary border-b pb-1">
                                    {rec.name}'s Report
                                </h3>
                                <p className="text-sm"><strong>Gender:</strong> {rec.gender}</p>
                                <p className="text-sm"><strong>DOB:</strong> {rec.dateOfBirth}</p>
                                <p className="text-sm"><strong>Test Date:</strong> {rec.dateOfTest}</p>
                                <p className="text-xs text-gray-500 mt-2 line-clamp-2">
                                    {rec.diagnosis}
                                </p>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>

            {/* ------------------- Selected Report Modal ------------------- */}
            <AnimatePresence>
                {selectedReport && (
                    <motion.div
                        className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setSelectedReport(null)}
                    >
                        <motion.div
                            className="bg-white rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col relative"
                            initial={{ y: 50, opacity: 0, scale: 0.95 }}
                            animate={{ y: 0, opacity: 1, scale: 1 }}
                            exit={{ y: 50, opacity: 0, scale: 0.95 }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Header */}
                            <div className="flex justify-between items-center border-b p-6">
                                <h2 className="text-2xl font-bold text-primary">
                                    {selectedReport.name}'s Full Report
                                </h2>
                                <button
                                    className="text-gray-500 hover:text-gray-700 transition-colors"
                                    onClick={() => setSelectedReport(null)}
                                >
                                    <X className="h-6 w-6" />
                                </button>
                            </div>

                            {/* Content - Scrollable */}
                            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                                {/* Patient Information */}
                                <Card>
                                    <CardContent className="p-4 space-y-2">
                                        <h3 className="text-lg font-semibold text-primary border-b pb-2">
                                            Patient Information
                                        </h3>
                                        <div className="grid grid-cols-2 gap-3">
                                            <div>
                                                <p className="text-xs text-gray-500">Name</p>
                                                <p className="font-medium">{selectedReport.name}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Gender</p>
                                                <p className="font-medium">{selectedReport.gender}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Date of Birth</p>
                                                <p className="font-medium">{selectedReport.dateOfBirth}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Test Date</p>
                                                <p className="font-medium">{selectedReport.dateOfTest}</p>
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Diagnosis */}
                                <Card>
                                    <CardContent className="p-4 space-y-2">
                                        <h3 className="text-lg font-semibold text-primary border-b pb-2">
                                            Diagnosis
                                        </h3>
                                        <p className="text-foreground leading-relaxed">
                                            {selectedReport.diagnosis}
                                        </p>
                                    </CardContent>
                                </Card>

                                {/* Explanation */}
                                <Card>
                                    <CardContent className="p-4 space-y-2">
                                        <h3 className="text-lg font-semibold text-primary border-b pb-2">
                                            Detailed Explanation
                                        </h3>
                                        <p className="text-foreground leading-relaxed">
                                            {selectedReport.explanation}
                                        </p>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Action Buttons - Fixed Footer */}
                            <div className="border-t p-6 bg-gray-50">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <Card
                                        className="hover:scale-105 transition-all cursor-pointer hover:shadow-lg border-2 border-transparent hover:border-primary">
                                        <CardContent className="p-4">
                                            <div className="flex items-center gap-3" onClick={handleChatWithAI}>
                                                <div className="p-3 bg-primary/10 rounded-lg">
                                                    <MessageSquare className="h-6 w-6 text-primary" />
                                                </div>
                                                <div>
                                                    <h3 className="text-base font-semibold text-foreground">
                                                        Chat with AI
                                                    </h3>
                                                    <p className="text-xs text-muted-foreground">
                                                        Ask questions about this report
                                                    </p>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>

                                    <Card
                                        className="hover:scale-105 transition-all cursor-pointer hover:shadow-lg border-2 border-transparent hover:border-secondary"
                                        onClick={handleAskDoctor}
                                    >
                                        <CardContent className="p-4">
                                            <div className="flex items-center gap-3">
                                                <div className="p-3 bg-secondary/10 rounded-lg">
                                                    <Stethoscope className="h-6 w-6 text-secondary" />
                                                </div>
                                                <div>
                                                    <h3 className="text-base font-semibold text-foreground">
                                                        Ask a Doctor
                                                    </h3>
                                                    <p className="text-xs text-muted-foreground">
                                                        Get professional consultation
                                                    </p>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </main>
    );
}
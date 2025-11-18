import { createContext, useContext, useState, ReactNode } from "react";

export interface MedicalReport {
  name: string;
  gender: string;
  dateOfBirth: string;
  dateOfTest: string;
  diagnosis: string;
  explanation: string;
}

interface PatientReportContextType {
  report: MedicalReport | null;
  setReport: (report: MedicalReport | null) => void;
}

const PatientReportContext = createContext<PatientReportContextType>({
  report: null,
  setReport: () => {},
});

export const PatientReportProvider = ({ children }: { children: ReactNode }) => {
  const [report, setReport] = useState<MedicalReport | null>(null);

  return (
    <PatientReportContext.Provider value={{ report, setReport }}>
      {children}
    </PatientReportContext.Provider>
  );
};

// Custom hook to easily use the context
export const usePatientReport = () => useContext(PatientReportContext);

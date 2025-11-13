import { Card, CardContent } from "@/components/ui/card";
import { MapPin, GraduationCap, Stethoscope } from "lucide-react";

interface DoctorCardProps {
  doctor_name: string;
  location: string;
  specialization: string;
  degree: string;
}

const DoctorCard = ({ doctor_name, location, specialization, degree }: DoctorCardProps) => {
  return (
    <Card className="hover-scale transition-all">
      <CardContent className="p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Stethoscope className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">{doctor_name}</h3>
            <p className="text-sm text-muted-foreground">{specialization}</p>
          </div>
        </div>
        
        <div className="space-y-2 pt-2 border-t">
          <div className="flex items-center gap-2 text-sm">
            <MapPin className="h-4 w-4 text-primary" />
            <span className="text-muted-foreground">{location}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <GraduationCap className="h-4 w-4 text-primary" />
            <span className="text-muted-foreground">{degree}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default DoctorCard;

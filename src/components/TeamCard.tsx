import { Card, CardContent } from "@/components/ui/card";
import { User } from "lucide-react";

interface TeamCardProps {
  name: string;
  image?: string;
}

const TeamCard = ({ name, image }: TeamCardProps) => {
  return (
    <Card className="bg-accent border-0 shadow-md transition-all duration-300 hover:scale-105 hover:shadow-lg overflow-hidden">
      <CardContent className="p-0">
        <div className="aspect-square bg-muted flex items-center justify-center">
          {image ? (
            <img src={image} alt={name} className="w-full h-full object-cover" />
          ) : (
            <User className="h-24 w-24 text-muted-foreground" />
          )}
        </div>
        <div className="p-6 text-center">
          <h3 className="text-xl font-semibold text-foreground">{name}</h3>
        </div>
      </CardContent>
    </Card>
  );
};

export default TeamCard;

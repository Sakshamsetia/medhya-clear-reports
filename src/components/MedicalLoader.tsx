const MedicalLoader = () => {
  return (
    <div className="relative w-32 h-32">
      {/* Outer rotating ring */}
      <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
      
      {/* Rotating primary ring */}
      <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary border-r-primary animate-spin" />
      
      {/* Inner rotating ring */}
      <div className="absolute inset-4 rounded-full border-4 border-transparent border-b-secondary border-l-secondary animate-spin" 
           style={{ animationDirection: "reverse", animationDuration: "1s" }} />
      
      {/* Center pulse */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-8 h-8 bg-primary rounded-full animate-pulse" />
      </div>
    </div>
  );
};

export default MedicalLoader;

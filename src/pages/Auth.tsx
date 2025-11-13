import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import authBackground from "@/assets/auth-background.jpg";

const Auth = () => {
  const navigate = useNavigate();
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [signupType, setSignupType] = useState<"patient" | "doctor">("patient");

  // Patient signup fields
  const [patientName, setPatientName] = useState("");
  const [patientEmail, setPatientEmail] = useState("");
  const [patientPassword, setPatientPassword] = useState("");
  const [patientLocation, setPatientLocation] = useState("");

  // Doctor signup fields
  const [doctorName, setDoctorName] = useState("");
  const [doctorEmail, setDoctorEmail] = useState("");
  const [doctorPassword, setDoctorPassword] = useState("");
  const [doctorLocation, setDoctorLocation] = useState("");
  const [doctorSpecialization, setDoctorSpecialization] = useState("");
  const [doctorQualifications, setDoctorQualifications] = useState("");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Login:", { email: loginEmail, password: loginPassword });
    navigate("/dashboard");
  };

  const handlePatientSignup = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Patient Signup:", {
      name: patientName,
      email: patientEmail,
      password: patientPassword,
      location: patientLocation,
    });
    navigate("/dashboard");
  };

  const handleDoctorSignup = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Doctor Signup:", {
      name: doctorName,
      email: doctorEmail,
      password: doctorPassword,
      location: doctorLocation,
      specialization: doctorSpecialization,
      qualifications: doctorQualifications,
    });
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-2">
      {/* Left side - Forms */}
      <div className="flex items-center justify-center p-8 bg-background">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center space-y-2">
            <Link to="/" className="inline-flex items-center gap-2 mb-4">
              <Activity className="h-10 w-10 text-primary" />
              <span className="text-2xl font-bold">Medhyamed</span>
            </Link>
            <h1 className="text-3xl font-bold text-foreground">Welcome</h1>
            <p className="text-muted-foreground">Sign in or create an account to continue</p>
          </div>

          <Tabs defaultValue="login" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">Login</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>

            <TabsContent value="login">
              <Card>
                <CardHeader>
                  <CardTitle>Login</CardTitle>
                  <CardDescription>Enter your credentials to access your account</CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleLogin} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="login-email">Email</Label>
                      <Input
                        id="login-email"
                        type="email"
                        placeholder="your@email.com"
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="login-password">Password</Label>
                      <Input
                        id="login-password"
                        type="password"
                        placeholder="••••••••"
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        required
                      />
                    </div>
                    <Button type="submit" className="w-full">
                      Login
                    </Button>
                  </form>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="signup">
              <Card>
                <CardHeader>
                  <CardTitle>Create Account</CardTitle>
                  <CardDescription>Choose your account type and fill in your details</CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs value={signupType} onValueChange={(v) => setSignupType(v as "patient" | "doctor")}>
                    <TabsList className="grid w-full grid-cols-2 mb-6">
                      <TabsTrigger value="patient">Patient</TabsTrigger>
                      <TabsTrigger value="doctor">Doctor</TabsTrigger>
                    </TabsList>

                    <TabsContent value="patient">
                      <form onSubmit={handlePatientSignup} className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="patient-name">Name</Label>
                          <Input
                            id="patient-name"
                            placeholder="John Doe"
                            value={patientName}
                            onChange={(e) => setPatientName(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="patient-email">Email</Label>
                          <Input
                            id="patient-email"
                            type="email"
                            placeholder="your@email.com"
                            value={patientEmail}
                            onChange={(e) => setPatientEmail(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="patient-password">Password</Label>
                          <Input
                            id="patient-password"
                            type="password"
                            placeholder="••••••••"
                            value={patientPassword}
                            onChange={(e) => setPatientPassword(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="patient-location">Location</Label>
                          <Input
                            id="patient-location"
                            placeholder="City, Country"
                            value={patientLocation}
                            onChange={(e) => setPatientLocation(e.target.value)}
                            required
                          />
                        </div>
                        <Button type="submit" className="w-full">
                          Sign Up as Patient
                        </Button>
                      </form>
                    </TabsContent>

                    <TabsContent value="doctor">
                      <form onSubmit={handleDoctorSignup} className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="doctor-name">Name</Label>
                          <Input
                            id="doctor-name"
                            placeholder="Dr. Jane Smith"
                            value={doctorName}
                            onChange={(e) => setDoctorName(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="doctor-email">Email</Label>
                          <Input
                            id="doctor-email"
                            type="email"
                            placeholder="doctor@hospital.com"
                            value={doctorEmail}
                            onChange={(e) => setDoctorEmail(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="doctor-password">Password</Label>
                          <Input
                            id="doctor-password"
                            type="password"
                            placeholder="••••••••"
                            value={doctorPassword}
                            onChange={(e) => setDoctorPassword(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="doctor-location">Location</Label>
                          <Input
                            id="doctor-location"
                            placeholder="City, Country"
                            value={doctorLocation}
                            onChange={(e) => setDoctorLocation(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="doctor-specialization">Specialization</Label>
                          <Input
                            id="doctor-specialization"
                            placeholder="e.g., Cardiology"
                            value={doctorSpecialization}
                            onChange={(e) => setDoctorSpecialization(e.target.value)}
                            required
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="doctor-qualifications">Qualifications</Label>
                          <Input
                            id="doctor-qualifications"
                            placeholder="e.g., MBBS, MD"
                            value={doctorQualifications}
                            onChange={(e) => setDoctorQualifications(e.target.value)}
                            required
                          />
                        </div>
                        <Button type="submit" className="w-full">
                          Sign Up as Doctor
                        </Button>
                      </form>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Right side - Image */}
      <div className="hidden lg:block relative">
        <img
          src={authBackground}
          alt="Medical Professional"
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-primary/10" />
      </div>
    </div>
  );
};

export default Auth;

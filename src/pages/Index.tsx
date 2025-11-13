import { FileText, Stethoscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FeatureCard from "@/components/FeatureCard";
import heroImage from "@/assets/hero-medical.jpg";
import ParticleBackground from '@/components/particle'

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <ParticleBackground/>
      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 ">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6 animate-fade-in">
              <h1 className="text-5xl md:text-6xl font-bold leading-tight">
                <span className="text-foreground">Medhyamed</span>
                <br />
                <span className="text-primary">WHERE REPORTS SPEAK CLEARLY</span>
              </h1>
              <p className="text-xl text-muted-foreground leading-relaxed">
                Transform complex medical reports into clear, understandable insights powered by AI technology.
              </p>
              <div className="flex gap-4">
                <Link to="/auth">
                  <Button size="lg" className="text-base px-8">
                    Get Started
                  </Button>
                </Link>
                <Link to="/about">
                  <Button size="lg" variant="outline" className="text-base px-8">
                    Learn More
                  </Button>
                </Link>
              </div>
            </div>
            <div className="lg:order-last order-first animate-scale-in">
              <img
                src={heroImage}
                alt="Medical Technology"
                className="rounded-2xl shadow-strong w-full"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <FeatureCard
              icon={FileText}
              title="Medhya Explainer"
              description="Get instant simplified version of your medical report through AI. Understand complex medical terminology in plain language that makes sense."
            />
            <FeatureCard
              icon={Stethoscope}
              title="Medhya Scribe"
              description="Generate instant report through AI for Doctors. Streamline your documentation process and focus more on patient care."
            />
          </div>
        </div>
      </section>

      {/* Report Generation Info Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-accent">
        <div className="container mx-auto max-w-4xl text-center space-y-6">
          <h2 className="text-4xl font-bold text-foreground">
            AI-Powered Report Generation
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            Our advanced AI technology analyzes medical reports and generates clear, comprehensive explanations
            that help patients understand their health better. For healthcare professionals, we provide intelligent
            documentation tools that save time and improve accuracy.
          </p>
          <p className="text-lg text-muted-foreground leading-relaxed">
            Whether you're a patient seeking clarity or a doctor needing efficient documentation,
            Medhyamed bridges the gap between complex medical information and easy understanding.
          </p>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;

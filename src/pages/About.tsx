import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import TeamCard from "@/components/TeamCard";
import ParticleBackground from '@/components/particle'

const About = () => {
  const teamMembers = [
    { name: "Team Member 1" },
    { name: "Team Member 2" },
    { name: "Team Member 3" },
    { name: "Team Member 4" },
    { name: "Team Member 5" },
    { name: "Team Member 6" },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <ParticleBackground/>
      <main className="flex-1 py-20 px-4 sm:px-6 lg:px-8">
        <div className="container mx-auto">
          <div className="text-center mb-16 space-y-4 animate-fade-in">
            <h1 className="text-5xl font-bold text-foreground">Our Team</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Meet the dedicated professionals behind Medhyamed, working to make medical information accessible to everyone.
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-6 md:gap-8 animate-scale-in">
            {teamMembers.map((member, index) => (
              <TeamCard key={index} name={member.name} />
            ))}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default About;

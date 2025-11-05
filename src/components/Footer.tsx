import { Activity } from "lucide-react";

const Footer = () => {
  return (
    <footer className="border-t bg-accent">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Activity className="h-6 w-6 text-primary" />
              <span className="text-lg font-bold">Medhyamed</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Transforming medical reports into clear, understandable insights for better healthcare decisions.
            </p>
          </div>

          <div className="space-y-4">
            <h3 className="font-semibold text-foreground">Quick Links</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a href="/about" className="hover:text-primary transition-colors">
                  About Us
                </a>
              </li>
              <li>
                <a href="/auth" className="hover:text-primary transition-colors">
                  Get Started
                </a>
              </li>
            </ul>
          </div>

          <div className="space-y-4">
            <h3 className="font-semibold text-foreground">Contact</h3>
            <p className="text-sm text-muted-foreground">
              Have questions? Reach out to us and we'll help you understand your medical reports better.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Medhyamed. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

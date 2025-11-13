import { Outlet } from "react-router-dom";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function DashboardLayout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <SidebarProvider>
        <div className="flex flex-1 w-full">
          <AppSidebar />
          <main className="flex-1 relative">
            <div className="sticky top-16 z-40 bg-background/95 backdrop-blur border-b px-4 py-2">
              <SidebarTrigger />
            </div>
            <Outlet />
          </main>
        </div>
      </SidebarProvider>
      <Footer />
    </div>
  );
}

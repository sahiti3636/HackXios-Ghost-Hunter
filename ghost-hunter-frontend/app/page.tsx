"use client"

import { useEffect, useState } from "react"
import ScrollGlobe from "@/components/ui/landing-page"
import { LightSpeed } from "@/components/ui/light-speed"
import { ParallaxComponent } from "@/components/ui/parallax-scrolling"
import Lenis from "@studio-freight/lenis"
import gsap from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"

export default function GhostHunterHome() {
  const [activeSection, setActiveSection] = useState(0)

  useEffect(() => {
    // Determine timing for transitions
    let timeout: NodeJS.Timeout

    if (activeSection === 1) {
      // LightSpeed: 4 seconds (to allow effect to play out)
      timeout = setTimeout(() => {
        setActiveSection(2)
      }, 4000)
    }

    return () => clearTimeout(timeout)
  }, [activeSection])

  useEffect(() => {
    // Initialize Lenis for smooth scrolling (global)
    const lenis = new Lenis()

    // Sync Lenis with GSAP ScrollTrigger
    lenis.on("scroll", ScrollTrigger.update)

    gsap.ticker.add((time) => {
      lenis.raf(time * 1000)
    })

    gsap.ticker.lagSmoothing(0)

    return () => {
      lenis.destroy()
      gsap.ticker.remove((time) => {
        lenis.raf(time * 1000)
      })
    }
  }, [])

  return (
    <div className="w-full bg-black min-h-screen relative">
      {/* Section 1: Globe - Use absolute for fade out, but it needs to be relative when active to allow scrolling? 
          Actually, if we want to scroll, it CANNOT be absolute inset-0. 
          When activeSection === 0, it should be relative. 
          When activeSection !== 0, if we hide it, we can make it absolute or hidden.
      */}
      <div
        className={`w-full transition-opacity duration-1000 ease-in-out z-30 ${activeSection === 0 ? "relative opacity-100" : "absolute inset-0 opacity-0 pointer-events-none h-screen overflow-hidden"
          }`}
      >
        <section className="w-full">
          <ScrollGlobe onDiveIn={() => {
            setActiveSection(1);
            // Optional: Scroll to top if desired for next section, or rely on layout shift
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }} />
        </section>
      </div>

      {/* Section 2: Light-Speed Effect - Fixed to cover screen regardless of previous scroll position */}
      <div
        className={`fixed inset-0 w-full h-full transition-opacity duration-1000 ease-in-out z-40 ${activeSection === 1 ? "opacity-100" : "opacity-0 pointer-events-none"
          }`}
      >
        <section className="h-screen w-full relative overflow-hidden">
          <LightSpeed className="w-full h-full bg-black" />
        </section>
      </div>

      {/* Section 3: Parallax with Ghost Hunter - Relative to start fresh flow? */}
      <div
        className={`w-full transition-opacity duration-1000 ease-in-out z-10 ${activeSection === 2 ? "relative opacity-100" : "absolute inset-0 opacity-0 h-0 overflow-hidden"
          }`}
      >
        <section className="min-h-screen w-full">
          {activeSection === 2 && <ParallaxComponent />}
        </section>
      </div>
    </div>
  )
}

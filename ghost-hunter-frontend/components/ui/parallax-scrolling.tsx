"use client"

import { useEffect, useRef } from "react"
import gsap from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"

import { useRouter } from "next/navigation"

export function ParallaxComponent() {
  const router = useRouter()
  const parallaxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger)

    const triggerElement = parallaxRef.current?.querySelector("[data-parallax-layers]")

    if (triggerElement) {
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: triggerElement,
          start: "0% 0%",
          end: "100% 0%",
          scrub: 0,
        },
      })

      const layers = [
        { layer: "1", yPercent: 100 }, // Increased from 70
        { layer: "2", yPercent: 75 },  // Increased from 55
        { layer: "3", yPercent: 50 },  // Increased from 40
        { layer: "4", yPercent: 20 },  // Increased from 10
      ]

      layers.forEach((layerObj, idx) => {
        tl.to(
          triggerElement.querySelectorAll(`[data-parallax-layer="${layerObj.layer}"]`),
          {
            yPercent: layerObj.yPercent,
            ease: "none",
          },
          idx === 0 ? undefined : "<",
        )
      })
    }

    return () => {
      ScrollTrigger.getAll().forEach((st) => st.kill())
      gsap.killTweensOf(triggerElement)
    }
  }, [])

  return (
    <div className="parallax" ref={parallaxRef}>
      <section className="parallax__header relative w-full h-[150vh] overflow-hidden">
        <div className="parallax__visuals relative w-full h-full">
          {/* Background image layer */}
          <div
            data-parallax-layer="1"
            className="absolute inset-0 w-full h-full bg-cover bg-center"
            style={{
              backgroundImage: "url('/ghost-hunter-bg.png')",
              backgroundPosition: "center",
              backgroundSize: "cover",
            }}
          />

          {/* Overlay gradient */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/40" />

          {/* Ghost Hunter Title */}
          <div
            data-parallax-layer="3"
            className="parallax__layer-title absolute inset-0 flex items-center justify-center top-[-10%]"
          >
            <h1 className="text-[15vw] leading-none font-black text-white text-center drop-shadow-2xl tracking-tighter uppercase">
              GHOST HUNTER
            </h1>
          </div>

          {/* Parallax fade effect */}
          <div className="parallax__fade absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent" />
        </div>
      </section>

      <section className="parallax__content relative w-full min-h-screen bg-black flex flex-col items-center justify-center gap-8 px-4">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="120"
          height="120"
          viewBox="0 0 160 160"
          fill="none"
          className="text-cyan-400"
        >
          <path
            d="M94.8284 53.8578C92.3086 56.3776 88 54.593 88 51.0294V0H72V59.9999C72 66.6273 66.6274 71.9999 60 71.9999H0V87.9999H51.0294C54.5931 87.9999 56.3777 92.3085 53.8579 94.8283L18.3431 130.343L29.6569 141.657L65.1717 106.142C67.684 103.63 71.9745 105.396 72 108.939V160L88.0001 160L88 99.9999C88 93.3725 93.3726 87.9999 100 87.9999H160V71.9999H108.939C105.407 71.9745 103.64 67.7091 106.12 65.1938L106.142 65.1716L141.657 29.6568L130.343 18.3432L94.8284 53.8578Z"
            fill="currentColor"
          ></path>
        </svg>

        <div className="text-center space-y-4 max-w-2xl">
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Ready to Hunt?</h2>
          <p className="text-xl md:text-2xl text-gray-300 mb-8">Click below to begin your supernatural investigation</p>
          <button
            onClick={() => router.push("/hunt")}
            className="relative px-10 py-4 text-xl font-bold text-black bg-cyan-400 rounded-lg hover:bg-cyan-300 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-400/50 transform hover:scale-105"
          >
            Use Ghost Hunter
          </button>
        </div>
      </section>
    </div>
  )
}

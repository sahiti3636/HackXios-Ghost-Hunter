"use client";

import React, { useState } from "react";
import { CalendarIcon, Crosshair, Map as MapIcon, PlayCircle, Settings, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { ProcessingView } from "./processing-view";
import { useRouter } from "next/navigation";

// Mock Data for MPAs
const EXISTING_MPAS = [
    { id: "mpa-1", name: "Galápagos Marine Reserve", area: "133,000 km²" },
    { id: "mpa-2", name: "Papahānaumokuākea", area: "1,508,870 km²" },
    { id: "mpa-3", name: "Great Barrier Reef", area: "344,400 km²" },
    { id: "mpa-4", name: "Palau National Marine Sanctuary", area: "500,000 km²" },
];

export function RegionSetup() {
    const router = useRouter();
    const [mode, setMode] = useState<"draw" | "select">("draw");
    const [selectedMpa, setSelectedMpa] = useState<string>("");
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [startDate, setStartDate] = useState("");
    const [endDate, setEndDate] = useState("");

    // Simulate polygon drawing state
    const [hasPolygon, setHasPolygon] = useState(false); // In a real app, this would check the map state

    const handleAnalysisComplete = () => {
        // Redirect to a random result ID
        router.push("/results/analysis-8829");
    };

    const startAnalysis = () => {
        setIsAnalyzing(true);
    };

    const canAnalyze = (mode === "draw" && hasPolygon) || (mode === "select" && selectedMpa);

    if (isAnalyzing) {
        return <ProcessingView onComplete={handleAnalysisComplete} />;
    }

    return (
        <div className="flex flex-col h-screen w-full bg-black text-white overflow-hidden font-sans">
            {/* Top Bar */}
            <header className="h-14 border-b border-white/10 bg-black/50 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-50">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center">
                        <Crosshair className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold tracking-wider text-lg">GHOST<span className="text-cyan-400">HUNTER</span></span>
                    <div className="h-4 w-[1px] bg-white/20 mx-3" />
                    <span className="text-sm text-gray-400 font-medium">New Analysis</span>
                </div>
                <div className="flex items-center gap-4">
                    <button className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white">
                        <Settings className="w-5 h-5" />
                    </button>
                    <button className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white">
                        <User className="w-5 h-5" />
                    </button>
                </div>
            </header>

            <div className="flex-1 flex overflow-hidden relative">
                {/* Left Panel: Region Controls */}
                <aside className="w-80 border-r border-white/10 bg-black/80 backdrop-blur-sm flex flex-col z-40 relative shadow-2xl">
                    <div className="p-6 space-y-8 flex-1 overflow-y-auto">

                        {/* 1. Region Selection */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 text-cyan-400 mb-2">
                                <MapIcon className="w-4 h-4" />
                                <h3 className="text-sm font-bold uppercase tracking-widest">Target Region</h3>
                            </div>

                            <div className="grid grid-cols-2 gap-2 p-1 bg-white/5 rounded-lg border border-white/10">
                                <button
                                    onClick={() => setMode("draw")}
                                    className={cn(
                                        "py-2 px-3 text-sm font-medium rounded-md transition-all",
                                        mode === "draw"
                                            ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                                            : "text-gray-400 hover:text-white hover:bg-white/5"
                                    )}
                                >
                                    Draw Area
                                </button>
                                <button
                                    onClick={() => setMode("select")}
                                    className={cn(
                                        "py-2 px-3 text-sm font-medium rounded-md transition-all",
                                        mode === "select"
                                            ? "bg-cyan-500/20 text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                                            : "text-gray-400 hover:text-white hover:bg-white/5"
                                    )}
                                >
                                    Select MPA
                                </button>
                            </div>

                            {mode === "draw" ? (
                                <div className="space-y-3 p-4 border border-dashed border-white/20 rounded-lg bg-white/5">
                                    <p className="text-xs text-gray-400">Click and drag on the map to define a custom surveillance polygon.</p>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setHasPolygon(!hasPolygon)}
                                            className={cn("text-xs px-3 py-1 rounded border transition-colors", hasPolygon ? "border-green-500 text-green-400 bg-green-500/10" : "border-white/20 text-gray-300 hover:bg-white/10")}
                                        >
                                            {hasPolygon ? "Polygon Set (Mock)" : "Draw Polygon (Mock)"}
                                        </button>
                                    </div>
                                    {hasPolygon && (
                                        <div className="flex items-center gap-2 text-xs text-green-400">
                                            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                                            <span>Area: 425 km²</span>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <label className="text-xs text-gray-400">Select Protected Area</label>
                                    <select
                                        value={selectedMpa}
                                        onChange={(e) => setSelectedMpa(e.target.value)}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 transition-colors"
                                    >
                                        <option value="" disabled>Choose an MPA...</option>
                                        {EXISTING_MPAS.map(mpa => (
                                            <option key={mpa.id} value={mpa.id}>{mpa.name}</option>
                                        ))}
                                    </select>
                                    {selectedMpa && (
                                        <div className="p-3 bg-cyan-900/20 border border-cyan-500/30 rounded-lg">
                                            <div className="text-xs text-cyan-200">
                                                <span className="font-bold">{EXISTING_MPAS.find(m => m.id === selectedMpa)?.name}</span>
                                                <br />
                                                <span className="text-cyan-400/70">Area: {EXISTING_MPAS.find(m => m.id === selectedMpa)?.area}</span>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* 2. Timeframe Selection */}
                        <div className="space-y-4 pt-4 border-t border-white/10">
                            <div className="flex items-center gap-2 text-cyan-400 mb-2">
                                <CalendarIcon className="w-4 h-4" />
                                <h3 className="text-sm font-bold uppercase tracking-widest">Timeframe</h3>
                            </div>

                            <div className="grid grid-cols-1 gap-3">
                                <div className="space-y-1">
                                    <label className="text-xs text-gray-500">Start Date</label>
                                    <input
                                        type="date"
                                        value={startDate}
                                        onChange={(e) => setStartDate(e.target.value)}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-cyan-500 transition-colors [color-scheme:dark]"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs text-gray-500">End Date</label>
                                    <input
                                        type="date"
                                        value={endDate}
                                        onChange={(e) => setEndDate(e.target.value)}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-cyan-500 transition-colors [color-scheme:dark]"
                                    />
                                </div>
                            </div>
                            <p className="text-[10px] text-gray-500 leading-relaxed">
                                * Analysis is based on available Sentinel-1 SAR imagery revisit cycles in this region.
                            </p>
                        </div>

                    </div>

                    {/* Action Footer */}
                    <div className="p-6 border-t border-white/10 bg-black/40">
                        <button
                            onClick={startAnalysis}
                            disabled={!canAnalyze}
                            className={cn(
                                "w-full py-4 px-6 rounded-lg font-bold tracking-wider flex items-center justify-center gap-3 transition-all duration-300 relative overflow-hidden group",
                                canAnalyze
                                    ? "bg-cyan-600 hover:bg-cyan-500 text-white shadow-[0_0_20px_rgba(8,145,178,0.4)] hover:shadow-[0_0_30px_rgba(6,182,212,0.6)]"
                                    : "bg-white/5 text-gray-500 cursor-not-allowed border border-white/5"
                            )}
                        >
                            {canAnalyze && (
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                            )}
                            <PlayCircle className={cn("w-5 h-5", canAnalyze && "animate-pulse")} />
                            <span>INITIATE SCAN</span>
                        </button>
                    </div>
                </aside>

                {/* Main Content Area (Map Placeholder) */}
                <main className="flex-1 relative bg-[#050505] overflow-hidden">
                    {/* Grid Background Effect */}
                    <div className="absolute inset-0 z-0 opacity-20 pointer-events-none"
                        style={{
                            backgroundImage: 'linear-gradient(#1a1a1a 1px, transparent 1px), linear-gradient(90deg, #1a1a1a 1px, transparent 1px)',
                            backgroundSize: '40px 40px'
                        }}
                    />

                    {/* Placeholder Content Center */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 z-0 select-none">
                        <div className="w-24 h-24 rounded-full border border-white/10 flex items-center justify-center mb-4">
                            <Crosshair className="w-8 h-8 opacity-20" />
                        </div>
                        <p className="text-sm font-mono uppercase tracking-[0.2em] opacity-40">Global Map Render Output</p>

                        {/* Mock Polygon Visualization */}
                        {mode === "draw" && hasPolygon && (
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-48 border-2 border-cyan-500/50 bg-cyan-500/10 rounded-sm animate-pulse flex items-center justify-center">
                                <div className="px-2 py-1 bg-black/80 text-[10px] text-cyan-400 border border-cyan-500/30 rounded font-mono">
                                    TARGET: 425 km²
                                </div>

                                {/* Corner accents */}
                                <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-cyan-400" />
                                <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-cyan-400" />
                                <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-cyan-400" />
                                <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-cyan-400" />
                            </div>
                        )}
                    </div>

                    {/* Map UI Overlays */}
                    <div className="absolute top-6 right-6 flex flex-col gap-2 z-10">
                        <button className="w-10 h-10 bg-black/60 backdrop-blur border border-white/10 rounded-lg flex items-center justify-center text-white hover:bg-white/10 transition-colors">
                            <span className="text-xl">+</span>
                        </button>
                        <button className="w-10 h-10 bg-black/60 backdrop-blur border border-white/10 rounded-lg flex items-center justify-center text-white hover:bg-white/10 transition-colors">
                            <span className="text-xl">-</span>
                        </button>
                    </div>

                    <div className="absolute bottom-6 right-6 px-4 py-2 bg-black/60 backdrop-blur border border-white/10 rounded-lg text-xs font-mono text-gray-400">
                        SCALE: 1:50,000 | Lat: 34.0522 Long: -118.2437
                    </div>
                </main>
            </div>
        </div>
    );
}

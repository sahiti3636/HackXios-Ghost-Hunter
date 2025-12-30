"use client";

import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { Check, Loader2, Satellite, Wifi } from "lucide-react";

interface ProcessingViewProps {
    onComplete: () => void;
}

const STEPS = [
    { id: 1, label: "Fetching satellite data", detail: "Source: Sentinel-1 SAR imagery • 2.4GB processed" },
    { id: 2, label: "Detecting vessels", detail: "AIS Correlation complete • 412 targets identified" },
    { id: 3, label: "Filtering dark vessels", detail: "Cross-referencing RF Signal Analysis..." },
    { id: 4, label: "Behavioral analysis", detail: "Applying anomaly detection models..." },
];

export function ProcessingView({ onComplete }: ProcessingViewProps) {
    const [currentStep, setCurrentStep] = useState(0);
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        // Mock progress simulation
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    clearInterval(interval);
                    return 100;
                }
                return prev + 0.8; // ~3-4 seconds total
            });
        }, 30);

        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        // Step progression logic linked to progress bar
        if (progress > 15) setCurrentStep(1);
        if (progress > 50) setCurrentStep(2);
        if (progress > 80) setCurrentStep(3);

        if (progress >= 100) {
            setTimeout(() => {
                onComplete();
            }, 800); // Small delay at 100% before redirect
        }
    }, [progress, onComplete]);

    return (
        <div className="fixed inset-0 bg-black z-[100] flex items-center justify-center p-4">
            {/* Background Tech Effects */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-3xl animate-pulse" />
                <div className="absolute top-0 w-full h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
                <div className="absolute bottom-0 w-full h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
            </div>

            <div className="w-full max-w-2xl bg-[#0a0a0a] border border-white/10 rounded-2xl overflow-hidden relative shadow-2xl z-10">
                {/* Header */}
                <div className="p-6 border-b border-white/5 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="relative">
                            <div className="w-3 h-3 bg-cyan-500 rounded-full animate-ping absolute inset-0" />
                            <div className="w-3 h-3 bg-cyan-500 rounded-full relative" />
                        </div>
                        <h2 className="text-xl font-bold tracking-wide text-white">
                            ANALYSIS IN PROGRESS
                        </h2>
                    </div>
                    <div className="font-mono text-cyan-400 text-lg">{Math.round(progress)}%</div>
                </div>

                {/* Main Content */}
                <div className="p-8 space-y-8">
                    {/* Progress Bar */}
                    <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.8)] transition-all duration-100 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>

                    {/* Step List */}
                    <div className="space-y-6">
                        {STEPS.map((step, idx) => {
                            const isActive = idx === currentStep;
                            const isCompleted = idx < currentStep;
                            const isPending = idx > currentStep;

                            return (
                                <div
                                    key={step.id}
                                    className={cn(
                                        "flex items-start gap-4 transition-all duration-500",
                                        isPending && "opacity-30 blur-[1px]",
                                        isActive && "opacity-100 scale-[1.02]",
                                        isCompleted && "opacity-60"
                                    )}
                                >
                                    <div className={cn(
                                        "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border transition-colors duration-300",
                                        isCompleted ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400" :
                                            isActive ? "bg-cyan-500/10 border-cyan-500 text-white animate-pulse" :
                                                "bg-white/5 border-white/10 text-gray-500"
                                    )}>
                                        {isCompleted ? <Check className="w-4 h-4" /> :
                                            isActive ? <Loader2 className="w-4 h-4 animate-spin" /> :
                                                <div className="w-2 h-2 rounded-full bg-current" />}
                                    </div>
                                    <div className="space-y-1">
                                        <h4 className={cn("font-medium text-lg leading-none", isActive ? "text-cyan-100" : "text-gray-400")}>
                                            {step.label}
                                        </h4>
                                        <p className="text-sm text-gray-500 font-mono">
                                            {isActive ? <span className="typing-effect">{step.detail}</span> : step.detail}
                                        </p>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Footer Console Log look-alike */}
                <div className="bg-black p-4 border-t border-white/10 font-mono text-xs h-32 overflow-hidden text-gray-500 opacity-60">
                    <div className="space-y-1">
                        <div>[SYSTEM] Initializing neural filter v4.2...</div>
                        {progress > 10 && <div>[SAT_LINK] Connected to Sentinel-1B feed...</div>}
                        {progress > 25 && <div>[IMG_PROC] Resampling SAR grid... OK</div>}
                        {progress > 45 && <div>[AIS_DATA] 1240 records correlated</div>}
                        {progress > 60 && <div className="text-yellow-500/80">[WARN] Anomaly detected at 34.05, -118.24</div>}
                        {progress > 80 && <div className="text-cyan-500">[AI_CORE] Generating suspect profile...</div>}
                        <div className="animate-pulse">_</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

import React from 'react';
import { RegionSetup } from '@/components/hunt/region-setup';

export default function HuntPage() {
    return (
        <div className="min-h-screen bg-black text-white selection:bg-cyan-500/30">
            <RegionSetup />
        </div>
    );
}

import { useEffect, useRef, useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Scissors, Download } from 'lucide-react';
import { HighlightDetector } from '@/lib/highlight-detector';
import type { Highlight } from '@shared/schema';

interface VideoEditorProps {
  src?: string;
  highlights?: Highlight[];
  onHighlightDetected?: (highlight: Highlight) => void;
}

export function VideoEditor({ src, highlights = [], onHighlightDetected }: VideoEditorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [detector, setDetector] = useState<HighlightDetector | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (!videoRef.current || !src) return;

    const newDetector = new HighlightDetector(videoRef.current, {
      volumeThreshold: 0.4,
      motionThreshold: 0.3,
      sceneThreshold: 0.5
    });

    setDetector(newDetector);
    newDetector.start();

    return () => {
      newDetector.destroy();
    };
  }, [src]);

  const handleExport = async () => {
    if (!highlights.length) return;

    // Sort highlights by timestamp
    const sortedHighlights = [...highlights].sort((a, b) => a.timestamp - b.timestamp);

    // Generate EDL (Edit Decision List) format
    const edl = sortedHighlights.map((h, i) => {
      const start = h.timestamp;
      const end = i < highlights.length - 1 ? highlights[i + 1].timestamp : start + 5;
      return `${i + 1} BL C 　${formatTimecode(start)} ${formatTimecode(end)}`;
    }).join('\n');

    // Download EDL file
    const blob = new Blob([edl], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'highlights.edl';
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatTimecode = (seconds: number): string => {
    const hh = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const mm = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const ss = Math.floor(seconds % 60).toString().padStart(2, '0');
    const ff = Math.floor((seconds % 1) * 25).toString().padStart(2, '0');
    return `${hh}:${mm}:${ss}:${ff}`;
  };

  return (
    <Card className="w-full p-4 bg-[#1E1E1E] border-[#4E4376]">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-foreground">Video Editor</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={!highlights.length}
          >
            <Download className="w-4 h-4 mr-2" />
            Export EDL
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={isProcessing}
            onClick={() => setIsProcessing(!isProcessing)}
          >
            <Scissors className="w-4 h-4 mr-2" />
            {isProcessing ? 'Processing...' : 'Process Video'}
          </Button>
        </div>
      </div>
      
      <video
        ref={videoRef}
        src={src}
        className="w-full rounded-lg"
        controls
      />

      <div className="mt-4">
        <h4 className="text-sm font-medium mb-2">Detected Highlights:</h4>
        <div className="space-y-2">
          {highlights.map((highlight, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-2 rounded bg-[#2E2E2E]"
            >
              <span>
                {formatTimecode(highlight.timestamp)} - {highlight.type}
              </span>
              <span className="text-sm opacity-70">
                {Math.round(highlight.confidence * 100)}% confidence
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

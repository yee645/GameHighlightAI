import { InsertVideo } from "@shared/schema";

export async function getVideoDuration(file: File): Promise<string> {
  return new Promise((resolve) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      window.URL.revokeObjectURL(video.src);
      const duration = Math.round(video.duration);
      const minutes = Math.floor(duration / 60);
      const seconds = duration % 60;
      resolve(`${minutes}:${seconds.toString().padStart(2, "0")}`);
    };
    video.src = URL.createObjectURL(file);
  });
}

export async function processVideoForHighlights(file: File) {
  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  
  const highlights: InsertVideo["highlights"] = [];
  
  // Process video frames for motion detection
  video.addEventListener("play", () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    
    let lastImageData: ImageData | null = null;
    
    const processFrame = () => {
      if (video.paused || video.ended) return;
      
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        if (lastImageData) {
          const diff = detectMotion(imageData, lastImageData);
          if (diff > 0.3) {
            highlights.push({
              timestamp: video.currentTime,
              type: "motion",
              confidence: diff
            });
          }
        }
        
        lastImageData = imageData;
      }
      
      requestAnimationFrame(processFrame);
    };
    
    processFrame();
  });
  
  return highlights;
}

function detectMotion(current: ImageData, previous: ImageData): number {
  const diff = new Uint8ClampedArray(current.data.length);
  let diffCount = 0;
  
  for (let i = 0; i < current.data.length; i += 4) {
    const rDiff = Math.abs(current.data[i] - previous.data[i]);
    const gDiff = Math.abs(current.data[i + 1] - previous.data[i + 1]);
    const bDiff = Math.abs(current.data[i + 2] - previous.data[i + 2]);
    
    if (rDiff > 30 || gDiff > 30 || bDiff > 30) {
      diffCount++;
    }
  }
  
  return diffCount / (current.width * current.height);
}

export function downloadHighlightsCsv(highlights: NonNullable<InsertVideo["highlights"]>) {
  const csvContent = "timestamp,type,confidence\n" + 
    highlights.map(h => `${h.timestamp},${h.type},${h.confidence}`).join("\n");
  
  const blob = new Blob([csvContent], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "highlights.csv";
  a.click();
  window.URL.revokeObjectURL(url);
}

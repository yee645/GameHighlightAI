import type { InsertVideo } from "@shared/schema";

interface HighlightDetectorOptions {
  volumeThreshold?: number; // 0-1 scale
  motionThreshold?: number; // 0-1 scale
  sceneThreshold?: number; // 0-1 scale
}

export class HighlightDetector {
  private readonly video: HTMLVideoElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly audioContext: AudioContext;
  private readonly analyser: AnalyserNode;
  private readonly options: Required<HighlightDetectorOptions>;
  private lastImageData: ImageData | null = null;
  private highlights: NonNullable<InsertVideo["highlights"]> = [];

  constructor(
    video: HTMLVideoElement,
    options: HighlightDetectorOptions = {}
  ) {
    this.video = video;
    this.canvas = document.createElement('canvas');
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');
    this.ctx = ctx;

    // Set up canvas dimensions
    this.canvas.width = 320; // Reduced size for performance
    this.canvas.height = 240;

    // Initialize Web Audio
    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaElementSource(video);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    source.connect(this.analyser);
    this.analyser.connect(this.audioContext.destination);

    // Default options
    this.options = {
      volumeThreshold: options.volumeThreshold ?? 0.4,
      motionThreshold: options.motionThreshold ?? 0.3,
      sceneThreshold: options.sceneThreshold ?? 0.5
    };

    // Bind event handlers
    this.processFrame = this.processFrame.bind(this);
    this.analyzeAudio = this.analyzeAudio.bind(this);
  }

  public start(): void {
    // Start processing frames when video plays
    this.video.addEventListener('play', () => {
      this.processFrame();
      this.analyzeAudio();
    });
  }

  public getHighlights(): NonNullable<InsertVideo["highlights"]> {
    return [...this.highlights].sort((a, b) => a.timestamp - b.timestamp);
  }

  private processFrame(): void {
    if (this.video.paused || this.video.ended) return;

    // Draw current frame to canvas
    this.ctx.drawImage(
      this.video, 
      0, 
      0, 
      this.canvas.width, 
      this.canvas.height
    );

    const currentImageData = this.ctx.getImageData(
      0, 
      0, 
      this.canvas.width, 
      this.canvas.height
    );

    if (this.lastImageData) {
      // Detect motion
      const motionScore = this.detectMotion(currentImageData, this.lastImageData);
      if (motionScore > this.options.motionThreshold) {
        this.addHighlight({
          timestamp: this.video.currentTime,
          type: "motion",
          confidence: motionScore
        });
      }

      // Detect scene changes
      const sceneScore = this.detectSceneChange(currentImageData, this.lastImageData);
      if (sceneScore > this.options.sceneThreshold) {
        this.addHighlight({
          timestamp: this.video.currentTime,
          type: "scene",
          confidence: sceneScore
        });
      }
    }

    this.lastImageData = currentImageData;
    requestAnimationFrame(this.processFrame);
  }

  private analyzeAudio(): void {
    if (this.video.paused || this.video.ended) return;

    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(dataArray);

    // Calculate average volume
    const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
    const normalizedVolume = average / 255;

    if (normalizedVolume > this.options.volumeThreshold) {
      this.addHighlight({
        timestamp: this.video.currentTime,
        type: "volume",
        confidence: normalizedVolume
      });
    }

    requestAnimationFrame(this.analyzeAudio);
  }

  private detectMotion(current: ImageData, previous: ImageData): number {
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

  private detectSceneChange(current: ImageData, previous: ImageData): number {
    let totalDiff = 0;
    const pixelCount = current.data.length / 4;
    
    for (let i = 0; i < current.data.length; i += 4) {
      const rDiff = Math.abs(current.data[i] - previous.data[i]);
      const gDiff = Math.abs(current.data[i + 1] - previous.data[i + 1]);
      const bDiff = Math.abs(current.data[i + 2] - previous.data[i + 2]);
      
      totalDiff += (rDiff + gDiff + bDiff) / (3 * 255); // Normalize to 0-1
    }
    
    return totalDiff / pixelCount;
  }

  private addHighlight(highlight: InsertVideo["highlights"][number]): void {
    // Prevent duplicate highlights within 1 second
    const isDuplicate = this.highlights.some(h => 
      Math.abs(h.timestamp - highlight.timestamp) < 1 && 
      h.type === highlight.type
    );

    if (!isDuplicate) {
      this.highlights.push(highlight);
    }
  }

  public destroy(): void {
    this.audioContext.close();
    this.lastImageData = null;
    this.highlights = [];
  }
}

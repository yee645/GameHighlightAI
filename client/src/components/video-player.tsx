import { useEffect, useRef } from "react";
import videojs from "video.js";
import "video.js/dist/video-js.css";
import { Card } from "./ui/card";

interface VideoPlayerProps {
  src?: string;
  onTimeUpdate?: (time: number) => void;
  currentTime?: number;
}

export function VideoPlayer({ src, onTimeUpdate, currentTime }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const playerRef = useRef<any>(null);

  useEffect(() => {
    if (!videoRef.current) return;

    playerRef.current = videojs(videoRef.current, {
      controls: true,
      fluid: true,
      playbackRates: [0.5, 1, 1.5, 2],
      controlBar: {
        children: [
          'playToggle',
          'volumePanel',
          'currentTimeDisplay',
          'timeDivider',
          'durationDisplay',
          'progressControl',
          'playbackRateMenuButton',
          'fullscreenToggle',
        ],
      },
    }, () => {
      console.log('Video player initialized');
    });

    return () => {
      if (playerRef.current) {
        playerRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    if (!playerRef.current) return;

    if (src) {
      console.log('Loading video source:', src);
      playerRef.current.src({ src, type: "video/mp4" });
    }

    if (onTimeUpdate) {
      playerRef.current.on("timeupdate", () => {
        const currentTime = playerRef.current.currentTime();
        console.log('Time update:', currentTime);
        onTimeUpdate(currentTime);
      });
    }

    // Add error handling
    playerRef.current.on('error', () => {
      console.error('Video player error:', playerRef.current.error());
    });

  }, [src, onTimeUpdate]);

  // Handle external time updates
  useEffect(() => {
    if (playerRef.current && currentTime !== undefined && 
        Math.abs(playerRef.current.currentTime() - currentTime) > 0.5) {
      console.log('Seeking to:', currentTime);
      playerRef.current.currentTime(currentTime);
    }
  }, [currentTime]);

  return (
    <Card className="w-full overflow-hidden bg-[#1E1E1E] border-[#4E4376]">
      <div data-vjs-player>
        <video
          ref={videoRef}
          className="video-js vjs-theme-city vjs-big-play-centered"
        />
      </div>
    </Card>
  );
}
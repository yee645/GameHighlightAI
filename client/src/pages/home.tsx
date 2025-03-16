import { useState, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { VideoPlayer } from "@/components/video-player";
import { VideoEditor } from "@/components/video-editor";
import { Timeline } from "@/components/timeline";
import { useToast } from "@/hooks/use-toast";
import { Upload, Download, Play } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { getVideoDuration, downloadHighlightsCsv } from "@/lib/video";
import type { Video } from "@shared/schema";
import type { Highlight } from "@/types";


// ML service endpoint URL - use Replit domain
const ML_SERVICE_URL = `https://${window.location.hostname.replace('0-', '1-')}`;

export default function Home() {
  const [videoUrl, setVideoUrl] = useState<string>();
  const [currentTime, setCurrentTime] = useState(0);
  const { toast } = useToast();

  const { data: video } = useQuery<Video>({
    queryKey: ["/api/videos/current"],
    enabled: !!videoUrl,
  });

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const duration = await getVideoDuration(file);

      // First, get ML predictions
      const formData = new FormData();
      formData.append("video", file);

      console.log("Sending request to ML service:", ML_SERVICE_URL);
      try {
        const mlResponse = await fetch(`${ML_SERVICE_URL}/api/predict`, {
          method: "POST",
          body: formData,
          credentials: 'include',
          mode: 'cors'
        });

        if (!mlResponse.ok) {
          const errorText = await mlResponse.text();
          throw new Error(`ML service error: ${errorText}`);
        }

        const mlHighlights = await mlResponse.json();
        console.log("Received highlights:", mlHighlights);

        if (!mlHighlights.highlights || !Array.isArray(mlHighlights.highlights)) {
          throw new Error("Invalid highlights data received from ML service");
        }

        // Upload to main server
        const uploadFormData = new FormData();
        uploadFormData.append("file", file);
        uploadFormData.append("name", file.name);
        uploadFormData.append("duration", duration);
        uploadFormData.append("highlights", JSON.stringify(mlHighlights.highlights));

        await apiRequest("POST", "/api/videos", uploadFormData);
      } catch (error) {
        console.error("Full error details:", error);
        throw error;
      }
    },
    onSuccess: () => {
      toast({
        title: "Video uploaded",
        description: "Your video has been processed successfully with ML highlights.",
      });
    },
    onError: (error) => {
      console.error("Upload error:", error);
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to process video",
        variant: "destructive",
      });
    },
  });

  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setVideoUrl(URL.createObjectURL(file));
    uploadMutation.mutate(file);
  }, [uploadMutation]);

  const handleSeek = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleExport = useCallback(() => {
    if (!video?.highlights) return;
    downloadHighlightsCsv(video.highlights);
  }, [video]);

  const loadTestVideo = useCallback(async () => {
    try {
      console.log("Loading test video from:", `${ML_SERVICE_URL}/api/test-video`);
      const response = await fetch(`${ML_SERVICE_URL}/api/test-video`, {
        credentials: 'include',
        mode: 'cors'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to load test video: ${errorText}`);
      }

      const blob = await response.blob();
      const file = new File([blob], 'test_video.mp4', { type: 'video/mp4' });

      setVideoUrl(URL.createObjectURL(file));
      uploadMutation.mutate(file);
    } catch (error) {
      console.error('Test video load error:', error);
      toast({
        title: "Test video load failed",
        description: error instanceof Error ? error.message : "Failed to load test video",
        variant: "destructive",
      });
    }
  }, [uploadMutation]);

  const handleHighlightDetected = useCallback((highlight: Highlight) => {
    console.log("New highlight detected:", highlight);
  }, []);

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-foreground">AI Video Highlight Detection</h1>
          <div className="flex gap-4">
            <Button
              variant="outline"
              onClick={() => document.getElementById("file-input")?.click()}
              disabled={uploadMutation.isPending}
            >
              <Upload className="mr-2 h-4 w-4" />
              {uploadMutation.isPending ? "Processing..." : "Upload Video"}
            </Button>
            <Button
              variant="secondary"
              onClick={loadTestVideo}
              disabled={uploadMutation.isPending}
            >
              <Play className="mr-2 h-4 w-4" />
              Load Test Video
            </Button>
            <Button
              variant="secondary"
              onClick={handleExport}
              disabled={!video?.highlights}
            >
              <Download className="mr-2 h-4 w-4" />
              Export Highlights
            </Button>
          </div>
          <input
            id="file-input"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <VideoPlayer
              src={videoUrl}
              onTimeUpdate={setCurrentTime}
              currentTime={currentTime}
            />
            <VideoEditor
              src={videoUrl}
              highlights={video?.highlights}
              onHighlightDetected={handleHighlightDetected}
            />
          </div>

          <Card className="p-6 bg-[#1E1E1E] border-[#4E4376]">
            <h2 className="text-xl font-semibold mb-6 text-foreground">Detected Highlights</h2>
            {video && video.highlights && video.highlights.length > 0 ? (
              <Timeline
                duration={parseFloat(video.duration)}
                highlights={video.highlights}
                currentTime={currentTime}
                onSeek={handleSeek}
              />
            ) : (
              <div className="text-center text-gray-500">
                {uploadMutation.isPending ? "Processing video..." : "No highlights detected"}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
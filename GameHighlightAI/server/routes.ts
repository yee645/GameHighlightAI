import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { insertVideoSchema } from "@shared/schema";

const upload = multer({ storage: multer.memoryStorage() });

export async function registerRoutes(app: Express): Promise<Server> {
  app.post("/api/videos", upload.single("file"), async (req, res) => {
    try {
      const videoData = {
        name: req.body.name,
        duration: req.body.duration,
        highlights: JSON.parse(req.body.highlights)
      };

      const parsed = insertVideoSchema.parse(videoData);
      const video = await storage.createVideo(parsed);
      res.json(video);
    } catch (error) {
      res.status(400).json({ message: "Invalid video data" });
    }
  });

  app.get("/api/videos/current", async (req, res) => {
    const videos = await storage.getVideos();
    // Return most recently uploaded video
    const current = videos[videos.length - 1];
    if (!current) {
      res.status(404).json({ message: "No videos found" });
      return;
    }
    res.json(current);
  });

  const httpServer = createServer(app);
  return httpServer;
}

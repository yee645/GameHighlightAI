import { videos, type Video, type InsertVideo, type Highlight } from "@shared/schema";

export interface IStorage {
  createVideo(video: InsertVideo): Promise<Video>;
  getVideos(): Promise<Video[]>;
}

export class MemStorage implements IStorage {
  private videos: Video[];
  private currentId: number;

  constructor() {
    this.videos = [];
    this.currentId = 1;
  }

  async createVideo(insertVideo: InsertVideo): Promise<Video> {
    const video: Video = {
      id: this.currentId++,
      name: insertVideo.name,
      duration: insertVideo.duration,
      uploadedAt: new Date(),
      highlights: insertVideo.highlights as Highlight[] | null,
    };
    this.videos.push(video);
    return video;
  }

  async getVideos(): Promise<Video[]> {
    return this.videos;
  }
}

export const storage = new MemStorage();
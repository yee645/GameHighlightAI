import { pgTable, text, serial, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Define the highlight type first for better type inference
export const highlightSchema = z.object({
  timestamp: z.number(),
  type: z.enum(["volume", "motion", "scene", "ml"]),
  confidence: z.number(),
});

export type Highlight = z.infer<typeof highlightSchema>;

export const videos = pgTable("videos", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  duration: text("duration").notNull(),
  uploadedAt: timestamp("uploaded_at").defaultNow().notNull(),
  highlights: jsonb("highlights").$type<Highlight[]>(),
});

export const insertVideoSchema = createInsertSchema(videos).omit({
  id: true,
  uploadedAt: true,
});

export type InsertVideo = z.infer<typeof insertVideoSchema>;
export type Video = typeof videos.$inferSelect;
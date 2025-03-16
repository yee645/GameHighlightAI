import { useRef, useEffect } from "react";
import * as d3 from "d3";
import { Card } from "./ui/card";
import type { Highlight } from "@shared/schema";

interface TimelineProps {
  duration: number;
  highlights: Highlight[];
  currentTime: number;
  onSeek: (time: number) => void;
}

export function Timeline({ duration, highlights, currentTime, onSeek }: TimelineProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const width = svgRef.current.clientWidth;
    const height = 100;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const xScale = d3.scaleLinear()
      .domain([0, duration])
      .range([margin.left, width - margin.right]);

    // Draw timeline base
    svg.append("line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", height / 2)
      .attr("y2", height / 2)
      .attr("stroke", "#4E4376")
      .attr("stroke-width", 2);

    // Draw time marks
    const numMarks = Math.min(duration, 10); // Adjust based on duration
    for (let i = 0; i <= numMarks; i++) {
      const x = margin.left + (i * (width - margin.left - margin.right) / numMarks);
      const timeValue = (i * duration / numMarks);
      const minutes = Math.floor(timeValue / 60);
      const seconds = Math.floor(timeValue % 60);
      const timeText = `${minutes}:${seconds.toString().padStart(2, '0')}`;

      svg.append("line")
        .attr("x1", x)
        .attr("x2", x)
        .attr("y1", height / 2 - 5)
        .attr("y2", height / 2 + 5)
        .attr("stroke", "#4E4376");

      svg.append("text")
        .attr("x", x)
        .attr("y", height / 2 + 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#F5F7FA")
        .attr("font-size", "12px")
        .text(timeText);
    }

    // Draw highlights
    highlights.forEach((highlight) => {
      const highlightColor = highlight.type === "ml" ? "#E74C3C" : 
                           highlight.type === "motion" ? "#2B5876" :
                           highlight.type === "scene" ? "#4E4376" : "#F5F7FA";

      svg.append("circle")
        .attr("cx", xScale(highlight.timestamp))
        .attr("cy", height / 2)
        .attr("r", 6)
        .attr("fill", highlightColor)
        .append("title")
        .text(`${highlight.type} (${Math.round(highlight.confidence * 100)}%)`);
    });

    // Draw current time marker with label
    const marker = svg.append("g")
      .attr("transform", `translate(${xScale(currentTime)}, 0)`);

    marker.append("line")
      .attr("x1", 0)
      .attr("x2", 0)
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .attr("stroke", "#F5F7FA")
      .attr("stroke-width", 2);

    const minutes = Math.floor(currentTime / 60);
    const seconds = Math.floor(currentTime % 60);
    marker.append("text")
      .attr("x", 0)
      .attr("y", margin.top - 5)
      .attr("text-anchor", "middle")
      .attr("fill", "#F5F7FA")
      .attr("font-size", "12px")
      .text(`${minutes}:${seconds.toString().padStart(2, '0')}`);

    // Add click handler for seeking
    svg.on("click", (event) => {
      const [x] = d3.pointer(event);
      const time = xScale.invert(x);
      onSeek(Math.max(0, Math.min(duration, time)));
    });

  }, [duration, highlights, currentTime, onSeek]);

  return (
    <Card className="w-full p-4 bg-[#1E1E1E] border-[#4E4376]">
      <svg
        ref={svgRef}
        className="w-full h-[100px]"
        style={{ backgroundColor: "transparent" }}
      />
    </Card>
  );
}
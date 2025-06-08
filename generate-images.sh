#!/bin/bash

# Script to generate PNG images from PlantUML and Mermaid diagram sources
# Generated images are placed in the images/ folder

cd "$(dirname "$0")"

echo "Generating PNG images from diagram sources..."

# Generate PNG files from PlantUML diagrams
if [ -f "plantuml.jar" ]; then
    echo "Generating PlantUML diagrams..."
    java -jar plantuml.jar -tpng -o ../images diagrams/*.puml
    echo "PlantUML diagrams generated successfully"
else
    echo "Warning: plantuml.jar not found, skipping PlantUML generation"
fi

# Generate PNG files from Mermaid diagrams (if mmdc is available)
if command -v mmdc &> /dev/null; then
    echo "Generating Mermaid diagrams..."
    for mmd_file in diagrams/*.mmd; do
        if [ -f "$mmd_file" ]; then
            filename=$(basename "$mmd_file" .mmd)
            mmdc -i "$mmd_file" -o "images/${filename}.png" --scale 4
        fi
    done
    echo "Mermaid diagrams generated successfully"
else
    echo "Warning: mmdc not found, skipping Mermaid generation"
    echo "Install with: npm install -g @mermaid-js/mermaid-cli"
fi

echo "Image generation complete!"
echo "Generated images are in: images/"
echo "Source diagrams are in: diagrams/"
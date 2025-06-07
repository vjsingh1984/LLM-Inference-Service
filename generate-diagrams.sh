#!/bin/bash

echo "🎨 Generating project diagrams..."

# Generate PlantUML diagrams
echo "📊 Generating PlantUML diagrams..."
if command -v java &> /dev/null && [ -f plantuml.jar ]; then
    java -jar plantuml.jar -tpng images/class_diagram.puml
    java -jar plantuml.jar -tpng images/api_sequence.puml  
    java -jar plantuml.jar -tpng images/deployment.puml
    echo "✅ PlantUML diagrams generated"
else
    echo "⚠️  PlantUML not available - skipping UML diagrams"
fi

# Generate Mermaid diagrams
echo "🐟 Generating Mermaid diagrams..."
if command -v mmdc &> /dev/null; then
    # Use puppeteer config to handle sandbox issues
    PUPPETEER_CONFIG="puppeteer-config.json"
    
    # Check if puppeteer config exists, create if not
    if [ ! -f "$PUPPETEER_CONFIG" ]; then
        echo '{"args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]}' > "$PUPPETEER_CONFIG"
    fi
    
    # Generate architecture diagram
    if mmdc -i images/architecture.mmd -o images/architecture.png -w 1024 -H 768 -p "$PUPPETEER_CONFIG" 2>/dev/null; then
        echo "✅ Architecture diagram generated"
    else
        echo "⚠️  Architecture diagram generation failed"
    fi
    
    # Generate package structure diagram
    if mmdc -i images/package_structure.mmd -o images/package_structure.png -w 800 -H 600 -p "$PUPPETEER_CONFIG" 2>/dev/null; then
        echo "✅ Package structure diagram generated"
    else
        echo "⚠️  Package structure diagram generation failed"
    fi
    
    # Generate request flow diagram
    if mmdc -i images/request_flow.mmd -o images/request_flow.png -w 800 -H 500 -p "$PUPPETEER_CONFIG" 2>/dev/null; then
        echo "✅ Request flow diagram generated"
    else
        echo "⚠️  Request flow diagram generation failed"
    fi
else
    echo "⚠️  Mermaid CLI not available - skipping Mermaid diagrams"
fi

echo "🎯 Diagram generation complete!"
echo ""
echo "📋 Generated files:"
ls -la images/*.png
echo ""
echo "📋 Next steps:"
echo "  1. Review generated diagrams in images/ directory"
echo "  2. Update diagrams by editing .mmd or .puml source files"
echo "  3. Re-run this script to regenerate"
echo "  4. Commit updated diagrams to git"
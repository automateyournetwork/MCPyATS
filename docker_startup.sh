#!/bin/bash

# Load environment variables from .env
if [ -f .env ]; then
  source .env
  echo "Loaded environment variables from .env"
else
  echo ".env file not found. Using default values."
fi

####################
#                  #
# Build containers #
#                  #
####################

echo "Building github-mcp image..."
docker build -t github-mcp ./mcp_servers/github
if [ $? -ne 0 ]; then echo "Error building github-mcp image."; exit 1; fi
echo "github-mcp image built successfully."

echo "Building google-maps-mcp image..."
docker build -t google-maps-mcp ./mcp_servers/google_maps
if [ $? -ne 0 ]; then echo "Error building google-maps-mcp image."; exit 1; fi
echo "google-maps-mcp image built successfully."

echo "Building sequentialthinking-mcp image..."
docker build -t sequentialthinking-mcp ./mcp_servers/sequentialthinking
if [ $? -ne 0 ]; then echo "Error building sequentialthinking-mcp image."; exit 1; fi
echo "sequentialthinking-mcp image built successfully."

echo "Building slack-mcp image..."
docker build -t slack-mcp ./mcp_servers/slack
if [ $? -ne 0 ]; then echo "Error building slack-mcp image."; exit 1; fi
echo "slack-mcp image built successfully."

echo "Building excalidraw-mcp image..."
docker build -t excalidraw-mcp ./mcp_servers/excalidraw
if [ $? -ne 0 ]; then echo "Error building excalidraw-mcp image."; exit 1; fi
echo "excalidraw-mcp image built successfully."

echo "Building filesystem-mcp image..."
docker build -t filesystem-mcp ./mcp_servers/filesystem
if [ $? -ne 0 ]; then echo "Error building filesystem-mcp image."; exit 1; fi
echo "filesystem-mcp image built successfully."

echo "Building netbox-mcp image..."
docker build -t netbox-mcp ./mcp_servers/netbox
if [ $? -ne 0 ]; then echo "Error building netbox-mcp image."; exit 1; fi
echo "netbox-mcp image built successfully."

echo "Building google-search-mcp image..."
docker build -t google-search-mcp ./mcp_servers/google_search
if [ $? -ne 0 ]; then echo "Error building google-search-mcp image."; exit 1; fi
echo "google-search-mcp image built successfully."

echo "Building sericenow-mcp image..."
docker build -t servicenow-mcp ./mcp_servers/servicenow
if [ $? -ne 0 ]; then echo "Error building servicenow-mcp image."; exit 1; fi
echo "servicenow-mcp image built successfully."

echo "Building email-mcp image..."
docker build -t email-mcp ./mcp_servers/email
if [ $? -ne 0 ]; then echo "Error building email-mcp image."; exit 1; fi
echo "email-mcp image built successfully."

echo "Building pyats-mcp image..."
docker build -t pyats-mcp ./mcp_servers/pyats_mcp_server
if [ $? -ne 0 ]; then echo "Error building pyats-mcp image."; exit 1; fi
echo "pyats-mcp image built successfully."

echo "Building chatgpt-mcp image..."
docker build -t chatgpt-mcp ./mcp_servers/chatgpt
if [ $? -ne 0 ]; then echo "Error building chatgpt-mcp image."; exit 1; fi
echo "chatgpt-mcp image built successfully."

echo "Building quickchart-mcp image..."
docker build -t quickchart-mcp ./mcp_servers/quickchart
if [ $? -ne 0 ]; then echo "Error building quickchart-mcp image."; exit 1; fi
echo "quickchart-mcp image built successfully."

echo "Building vegalite-mcp image..."
docker build -t vegalite-mcp ./mcp_servers/vegalite
echo "vegalite-mcp image built successfully"

echo "Building mermaid-mcp image..."
docker build -t mermaid-mcp ./mcp_servers/mermaid
echo "mermaid-mcp image built successfully"

echo "Building rfc-mcp image..."
docker build -t rfc-mcp ./mcp_servers/rfc
echo "rfc-mcp image built successfully"

echo "Building nist-mcp image..."
docker build -t nist-mcp ./mcp_servers/nist
echo "nist-mcp image built successfully"

echo "Building subnet-calculator-mcp image..."
docker build -t subnet-calculator-mcp ./mcp_servers/subnet_calculator
echo "subnet-calculator-mcp image built successfully"

echo "Building drawio-mcp image..."
docker build -t drawio-mcp ./mcp_servers/drawio_mcp
echo "drawio-mcp image built successfully"

echo "Building a2a image..."
docker build -t a2a ./a2a
echo "a2a image built successfully"

echo "Building streamlit-app image..."
docker build -t streamlit-app ./streamlit
if [ $? -ne 0 ]; then echo "Error building streamlit-app image."; exit 1; fi
echo "streamlit-app image built successfully."

echo "Building langgraph container (mcpyats)..."
docker build -t mcpyats -f ./mcpyats/Dockerfile ./mcpyats
if [ $? -ne 0 ]; then echo "Error building mcpyats image."; exit 1; fi
echo "mcpyats image built successfully."

echo "Building local drawio."
docker build -t drawio ./drawio
echo "local drawio image built successfully."

echo "Building local ise-mcp"
docker build -t ise-mcp ./mcp_servers/ise_mcp
echo "local ise-mcp image built successfully."

echo "Building local wikipedia-mcp"
docker build -t wikipedia-mcp ./mcp_servers/wikipedia
echo "local wikipedia-mcp image built successfully."

echo "Building local aci-mcp"
docker build -t aci-mcp ./mcp_servers/aci_mcp
echo "local aci-mcp image built successfully."

#######
#     #
# RUN #
#     #
#######

echo "Starting github-mcp container..."
docker run -dit --name github-mcp -e GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_GITHUB_TOKEN}" github-mcp
echo "github-mcp container started."

echo "Starting google-maps-mcp container..."
docker run -dit --name google-maps-mcp -e GOOGLE_MAPS_API_KEY="${GOOGLE_MAPS_API_KEY:-YOUR_GOOGLE_MAPS_API_KEY}" google-maps-mcp
echo "google-maps-mcp container started."

echo "Starting sequentialthinking-mcp container..."
docker run -dit --name sequentialthinking-mcp sequentialthinking-mcp
echo "sequentialthinking-mcp container started."

echo "Starting slack-mcp container..."
docker run -dit --name slack-mcp -e SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-YOUR_SLACK_BOT_TOKEN}" -e SLACK_TEAM_ID="${SLACK_TEAM_ID:-YOUR_SLACK_TEAM_ID}" slack-mcp
echo "slack-mcp container started."

echo "Starting excalidraw-mcp container..."
docker run -dit --name excalidraw-mcp excalidraw-mcp
echo "excalidraw-mcp container started."

docker run -dit \
  --name filesystem-mcp \
  -v "/home/johncapobianco/MCPyATS/shared_output:/projects" \
  filesystem-mcp

echo "Starting netbox-mcp container..."
docker run -d --name netbox-mcp -e NETBOX_URL="${NETBOX_URL:-YOUR_SELECTOR_URL}" -e NETBOX_TOKEN="${NETBOX_TOKEN:-NETBOX_TOKEN}" netbox-mcp python3 server.py --restart unless-stopped
echo "netbox-mcp container started."

echo "Starting google-search-mcp container..."
docker run -dit --name google-search-mcp google-search-mcp
echo "google-search-mcp container started."

echo "Starting service now-mcp container..."
docker run -d --name servicenow-mcp \
 --env-file .env \
 servicenow-mcp python3 server.py --restart unless-stopped
echo "servicenow-mcp container started."

echo "Starting email-mcp container..."
docker run -dit --name email-mcp --env-file .env --dns 8.8.8.8 email-mcp
echo "email-mcp container started."

echo "Starting cahtgpt-mcp container..."
docker run -dit --name chatgpt-mcp \
 --env-file .env \
 chatgpt-mcp python3 server.py --restart unless-stopped
echo "chatgpt-mcp container started."

echo "Starting pyats-mcp container..."
docker run -d --name pyats-mcp \
  -e PYATS_TESTBED_PATH="/app/testbed.yaml" \
  -v "$(pwd)/mcp_servers/pyats_mcp_server/testbed.yaml:/app/testbed.yaml" \
  pyats-mcp
echo "pyats-mcp container started."

echo "Starting quickchart-mcp container..."
docker run -dit --name quickchart-mcp quickchart-mcp
echo "quickchart-mcp container started."

echo "Starting vegalite-mcp container..."
docker run -dit --name vegalite-mcp \
  -v "/home/johncapobianco/MCPyATS/shared_output:/output" \
  vegalite-mcp
echo "vegalite-mcp container started."

echo "Starting mermaid-mcp container..."
docker run -dit --name mermaid-mcp \
  -v "/home/johncapobianco/MCPyATS/shared_output:/output" \
  -e CONTENT_IMAGE_SUPPORTED=false \
  mermaid-mcp
echo "mermaid-mcp container started."

echo "Starting rfc-mcp container..."
docker run -dit --name rfc-mcp rfc-mcp
echo "rfc-mcp container started."

echo "Starting nist-mcp container..."
docker run -dit \
  --name nist-mcp \
  --env-file .env \
  --dns 8.8.8.8 \
  nist-mcp 
echo "nist-mcp container started."

echo "Starting subnet-calculator-mcp container..."
docker run -dit --name subnet-calculator-mcp subnet-calculator-mcp
echo "subnet-calculator-mcp container started."

echo "Starting drawio-mcp container..."
docker run -dit --name drawio-mcp -p 3000:3000 -p 11434:11434 drawio-mcp
echo "✅ drawio-mcp container running with both STDIO + WebSocket"

# Check if last MCP containers are running
if ! docker ps | grep -q "drawio-mcp"; then
    echo "drawio-mcp container not found."
    exit 1
fi

echo "Starting ise-mcp container..."
docker run -dit --env-file .env --name ise-mcp ise-mcp
echo "ise-mcp container started."

echo "Starting wikipedia-mcp container..."
docker run -dit --name wikipedia-mcp wikipedia-mcp
echo "wikipedia-mcp container started."

echo "Starting aci-mcp container..."
docker run -dit --env-file .env --name aci-mcp aci-mcp
echo "aci-mcp container started."

sleep 2

docker run -p 10000:10000 \
  -dit \
  --name a2a \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -e LANGGRAPH_URL=http://host.docker.internal:2024 \
  -e PUBLIC_BASE_URL=https://83e7-76-68-137-181.ngrok-free.app \
  -v $(pwd)/a2a:/a2a \
  -v /home/johncapobianco/MCPyATS/shared_output:/output \
  a2a

docker run -dit \
  --name mcpyats \
  -p 2024:2024 \
  --env-file .env \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd)/a2a:/a2a \
  -v /home/johncapobianco/MCPyATS/shared_output:/output \
  mcpyats
echo "mcpyats container started."

echo "Starting streamlit-app container..."
docker run -d --name streamlit-app -p 8501:8501 streamlit-app
echo "streamlit-app container started at http://localhost:8501"

echo "Starting local drawio container..."
docker run -d -p 8080:80 --name drawio-local drawio
echo "local drawio container started at http://localhost:8080"

echo "All containers started."
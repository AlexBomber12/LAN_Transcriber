#!/usr/bin/env bash
set -euo pipefail

# Delete Mise APT source if present
sudo rm -f /etc/apt/sources.list.d/mise_jdx_dev.list
sudo rm -f /etc/apt/trusted.gpg.d/mise.gpg

# Remove the mise package if installed
sudo apt-get -y purge mise || true

# Refresh package lists
sudo apt-get update

# Final message
echo "Mise repository and package removed \xE2\x9C\x94"

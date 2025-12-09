#!/bin/bash
install_package() { 
    for package in "$@"; do 
        echo "Installing APT package: $package"; 
        # Use --only-upgrade to prevent attempting to install core packages again if possible
        # The installation uses the list filtered by xargs
        apt-get install -y --no-install-recommends "$package"; 
    done;
}
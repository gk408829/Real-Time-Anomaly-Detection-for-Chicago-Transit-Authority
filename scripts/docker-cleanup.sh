#!/bin/bash

# Docker cleanup script to manage disk space

echo "🧹 Docker Cleanup Script"
echo "========================"

# Show current usage
echo "📊 Current Docker disk usage:"
docker system df

echo ""
echo "🔍 Detailed breakdown:"
echo "Images:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "Containers:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}"

echo ""
echo "🧹 Cleanup options:"
echo "1. Clean build cache only (safe)"
echo "2. Remove unused images (safe)"
echo "3. Remove stopped containers (safe)"
echo "4. Full cleanup (removes everything unused)"
echo "5. Nuclear option (removes ALL Docker data)"
echo "6. Show what would be removed (dry run)"
echo ""

read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo "🗑️  Cleaning build cache..."
        docker builder prune -f
        ;;
    2)
        echo "🗑️  Removing unused images..."
        docker image prune -f
        ;;
    3)
        echo "🗑️  Removing stopped containers..."
        docker container prune -f
        ;;
    4)
        echo "🗑️  Full cleanup (unused images, containers, networks, build cache)..."
        docker system prune -f
        ;;
    5)
        echo "☢️  NUCLEAR OPTION: This will remove ALL Docker data!"
        read -p "Are you absolutely sure? Type 'YES' to confirm: " confirm
        if [ "$confirm" = "YES" ]; then
            echo "💥 Removing all Docker data..."
            docker system prune -a -f --volumes
        else
            echo "❌ Cancelled"
        fi
        ;;
    6)
        echo "👀 Dry run - what would be removed:"
        echo ""
        echo "Unused build cache:"
        docker builder prune --dry-run
        echo ""
        echo "Unused images:"
        docker image prune --dry-run
        echo ""
        echo "Unused containers:"
        docker container prune --dry-run
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "📊 Disk usage after cleanup:"
docker system df

echo ""
echo "💡 Tips to manage Docker space:"
echo "  • Run this cleanup script regularly"
echo "  • Use .dockerignore to exclude unnecessary files"
echo "  • Use multi-stage builds to reduce final image size"
echo "  • Remove old containers: docker rm \$(docker ps -aq)"
echo "  • Remove old images: docker rmi \$(docker images -q)"
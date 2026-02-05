#!/bin/bash
# Script de nettoyage Docker pour Wakee Airflow
# Libère RAM et espace disque

echo "🧹 Wakee Docker Cleanup"
echo "======================="

# 1. Arrête Airflow
echo "⏸️  Stopping Airflow..."
docker compose down

# 2. Nettoie les logs volumineux
echo "🗑️  Cleaning logs..."
find ./logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
find ./logs -name "*.log" -size +100M -delete 2>/dev/null || true

# 3. Nettoie le cache Python
echo "🐍 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 4. Nettoie les images Docker non utilisées
echo "🐳 Cleaning Docker images..."
docker image prune -f

# 5. Nettoie les containers arrêtés
echo "📦 Cleaning stopped containers..."
docker container prune -f

# 6. Nettoie les volumes non utilisés (ATTENTION)
echo "💾 Cleaning unused volumes..."
docker volume prune -f

# 7. Nettoie le build cache
echo "🔨 Cleaning build cache..."
docker builder prune -f

# 8. Stats finales
echo ""
echo "📊 Docker disk usage after cleanup:"
docker system df

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "To restart Airflow:"
echo "  docker compose up -d"

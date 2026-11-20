# Allineato ai requisiti Node di Angular 22 (^22.22.3 || ^24.15.0)
FROM node:24-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npx ng build --configuration production

# Stage runtime statico con Nginx (~10-15 MiB di RAM invece di ~675 MiB di `ng serve`)
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/dist/frontend/browser /usr/share/nginx/html
# Script eseguito automaticamente da /docker-entrypoint.sh di nginx per iniettare GS_API_URL in assets/config.js
COPY docker-entrypoint.d/40-config.sh /docker-entrypoint.d/40-config.sh
RUN chmod +x /docker-entrypoint.d/40-config.sh

EXPOSE 4200
CMD ["nginx", "-g", "daemon off;"]

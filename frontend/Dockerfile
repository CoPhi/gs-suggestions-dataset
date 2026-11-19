# Allineato ai requisiti Node di Angular 22 (^22.22.3 || ^24.15.0) e all'ambiente di sviluppo
FROM node:24-alpine

WORKDIR /app

COPY package*.json ./
# Installazione deterministica dal lockfile senza CLI globale non pinnata (evita drift tra major)
RUN npm ci

COPY . .

EXPOSE 4200
CMD ["npx", "ng", "serve", "--host", "0.0.0.0", "--port", "4200"]
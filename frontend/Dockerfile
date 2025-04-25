FROM node:22

WORKDIR /app/frontend

COPY . .

RUN npm install -g @angular/cli

RUN npm install -g

CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4200"]

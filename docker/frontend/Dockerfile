FROM node:20-alpine

WORKDIR /app

# Kopiowanie plików definicji pakietów
COPY package*.json ./

# Instalacja zależności
RUN npm install

# Kopiowanie reszty kodu (przydatne przy buildzie produkcyjnym, w dev nadpisze to wolumen z compose.yaml)
COPY . .

# Wystawienie portu używanego przez Vite
EXPOSE 5173

# Uruchomienie deweloperskiego serwera
CMD ["npm", "run", "dev"]

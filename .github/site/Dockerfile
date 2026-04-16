FROM nginx:alpine

RUN mkdir -p /app /data/coverage /data/bench

COPY index.html /app/index.html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 8080
